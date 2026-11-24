package pipeline

import (
	"encoding/json"
	"fmt"
	"sort"

	"github.com/sjwhitworth/golearn/base"
)

// OneHotEncoder expands every non-class CategoricalAttribute column into one
// FloatAttribute indicator column per category observed in the training data.
//
//   - The vocabulary is fixed at Fit time from the values actually observed
//     in the training rows (sorted alphabetically for determinism). It never
//     changes afterwards, which is what makes train/predict consistent and
//     cross-validation leak-free.
//   - At Transform time a value that was never seen during Fit produces an
//     all-zero indicator row for that column: it is neither an error nor
//     silently attributed to a known category. Each occurrence is counted and
//     reported via UnseenCounts.
//   - Indicator columns are named "<column>_<value>". If that name collides
//     with any other output column (an existing feature, a class column, or
//     a previously generated indicator), the suffix "_2", "_3", ... is
//     appended until the name is unique. The rule is deterministic: the same
//     input schema always yields the same output names.
//   - A categorical column with an empty vocabulary on the training data
//     (e.g. all its cells are missing) makes Fit fail with a descriptive
//     error rather than producing a silently degenerate table.
//
// InverseTransform is not supported: an indicator row cannot be mapped back
// to a category unambiguously (the all-zero row has no category at all).
type OneHotEncoder struct {
	schema []colDef
	// cols holds one fitted entry per non-class categorical column, in
	// schema order.
	cols []oneHotColumn
	// passthrough records, for each non-class non-categorical column, its
	// name; order matches schema order.
	fitted bool
	// unseen counts values not present in the fitted vocabulary, per
	// column, accumulated during the most recent Transform call.
	unseen map[string]int
}

type oneHotColumn struct {
	// Name of the source categorical column.
	Name string `json:"name"`
	// Vocab: categories observed at Fit time, sorted.
	Vocab []string `json:"vocab"`
	// OutNames: generated indicator column names, aligned with Vocab.
	OutNames []string `json:"out_names"`
	// index maps category -> position in Vocab for O(1) lookup.
	index map[string]int
}

// NewOneHotEncoder creates an unfitted OneHotEncoder.
func NewOneHotEncoder() *OneHotEncoder {
	return &OneHotEncoder{}
}

// Type returns the serialisation identifier of this stage.
func (o *OneHotEncoder) Type() string { return "one-hot-encoder" }

// String returns a human-readable summary.
func (o *OneHotEncoder) String() string {
	return fmt.Sprintf("OneHotEncoder(%d column(s))", len(o.cols))
}

// Fit fixes the vocabulary of every non-class categorical column from the
// values observed in train (training data only).
func (o *OneHotEncoder) Fit(train base.FixedDataGrid) error {
	o.schema = schemaOf(train)
	o.cols = nil
	o.unseen = make(map[string]int)

	_, rows := train.Size()

	// Collect the vocabulary of each categorical column from observed rows.
	for _, a := range nonClassAttrs(train) {
		cat, ok := a.(*base.CategoricalAttribute)
		if !ok {
			continue
		}
		spec, err := train.GetAttribute(a)
		if err != nil {
			return err
		}
		nVals := len(cat.GetValues())
		seen := make(map[string]bool)
		var vocab []string
		for r := 0; r < rows; r++ {
			raw := train.Get(spec, r)
			idx := base.UnpackBytesToU64(raw)
			if idx >= uint64(nVals) {
				// Cell never set (missing): not a category, skipped.
				continue
			}
			v := cat.GetValues()[idx]
			if !seen[v] {
				seen[v] = true
				vocab = append(vocab, v)
			}
		}
		if len(vocab) == 0 {
			return fmt.Errorf("one-hot-encoder: column %q has an empty vocabulary on the training data (no observed categories)", cat.GetName())
		}
		sort.Strings(vocab)
		index := make(map[string]int, len(vocab))
		for i, v := range vocab {
			index[v] = i
		}
		o.cols = append(o.cols, oneHotColumn{Name: cat.GetName(), Vocab: vocab, index: index})
	}

	// Generate output names with the deterministic collision rule. Fixed
	// names (all non-generated output columns) are reserved first.
	used := make(map[string]bool)
	for _, a := range train.AllAttributes() {
		if _, isCat := a.(*base.CategoricalAttribute); isCat && !isClassAttr(train, a) {
			continue // replaced by generated indicator columns
		}
		used[a.GetName()] = true
	}
	for ci := range o.cols {
		c := &o.cols[ci]
		c.OutNames = make([]string, len(c.Vocab))
		for i, v := range c.Vocab {
			name := fmt.Sprintf("%s_%s", c.Name, v)
			if used[name] {
				for suffix := 2; ; suffix++ {
					cand := fmt.Sprintf("%s_%d", name, suffix)
					if !used[cand] {
						name = cand
						break
					}
				}
			}
			used[name] = true
			c.OutNames[i] = name
		}
	}
	o.fitted = true
	return nil
}

func isClassAttr(data base.FixedDataGrid, a base.Attribute) bool {
	for _, c := range data.AllClassAttributes() {
		if c.Equals(a) {
			return true
		}
	}
	return false
}

// Vocabulary returns the fitted vocabulary of a column, or nil if the column
// is unknown. Mainly useful for inspection and tests.
func (o *OneHotEncoder) Vocabulary(column string) []string {
	for _, c := range o.cols {
		if c.Name == column {
			ret := make([]string, len(c.Vocab))
			copy(ret, c.Vocab)
			return ret
		}
	}
	return nil
}

// OutputNames returns the generated indicator column names of a column,
// aligned with Vocabulary(column).
func (o *OneHotEncoder) OutputNames(column string) []string {
	for _, c := range o.cols {
		if c.Name == column {
			ret := make([]string, len(c.OutNames))
			copy(ret, c.OutNames)
			return ret
		}
	}
	return nil
}

// UnseenCounts reports, per source column, how many values in the most recent
// Transform call were not part of the fitted vocabulary (and were therefore
// encoded as all-zero indicator rows).
func (o *OneHotEncoder) UnseenCounts() map[string]int {
	ret := make(map[string]int, len(o.unseen))
	for k, v := range o.unseen {
		ret[k] = v
	}
	return ret
}

// Transform expands the fitted categorical columns into indicator columns.
// Numeric and other non-categorical columns pass through unchanged, class
// columns are preserved, row order and count are kept.
func (o *OneHotEncoder) Transform(data base.FixedDataGrid) (base.FixedDataGrid, error) {
	if !o.fitted {
		return nil, ErrNotFitted
	}
	if err := checkSchema(o.Type(), o.schema, data); err != nil {
		return nil, err
	}

	// Reset the unseen-value counters for this call.
	o.unseen = make(map[string]int)

	// Build the output structure: non-categorical columns keep their
	// Attribute, categorical columns are replaced by their indicator
	// FloatAttributes, class columns are appended and re-marked.
	out := base.NewDenseInstances()
	passThrough := make(map[string]base.AttributeSpec) // out attr name -> out spec (set below)
	type catOut struct {
		col   *oneHotColumn
		specs []base.AttributeSpec
	}
	catOuts := make(map[string]*catOut)

	for _, a := range nonClassAttrs(data) {
		if _, isCat := a.(*base.CategoricalAttribute); isCat {
			c := o.findColumn(a.GetName())
			if c == nil {
				return nil, fmt.Errorf("one-hot-encoder: no fitted vocabulary for column %q", a.GetName())
			}
			co := &catOut{col: c}
			for _, n := range c.OutNames {
				co.specs = append(co.specs, out.AddAttribute(base.NewFloatAttribute(n)))
			}
			catOuts[c.Name] = co
		} else {
			spec := out.AddAttribute(a)
			passThrough[a.GetName()] = spec
		}
	}
	for _, c := range data.AllClassAttributes() {
		out.AddAttribute(c)
		if err := out.AddClassAttribute(c); err != nil {
			return nil, err
		}
	}

	_, rows := data.Size()
	if err := out.Extend(rows); err != nil {
		return nil, err
	}

	// Fill the rows.
	for _, a := range nonClassAttrs(data) {
		inSpec, err := data.GetAttribute(a)
		if err != nil {
			return nil, err
		}
		if cat, isCat := a.(*base.CategoricalAttribute); isCat {
			co := catOuts[cat.GetName()]
			nVals := len(cat.GetValues())
			for r := 0; r < rows; r++ {
				idx := base.UnpackBytesToU64(data.Get(inSpec, r))
				if idx >= uint64(nVals) {
					// Missing cell: treated as an unseen value.
					o.unseen[cat.GetName()]++
					continue
				}
				v := cat.GetValues()[idx]
				vi, ok := co.col.index[v]
				if !ok {
					// Unseen category: all-zero row, counted and reported.
					o.unseen[cat.GetName()]++
					continue
				}
				out.Set(co.specs[vi], r, base.PackFloatToBytes(1.0))
			}
		} else {
			outSpec := passThrough[a.GetName()]
			for r := 0; r < rows; r++ {
				out.Set(outSpec, r, data.Get(inSpec, r))
			}
		}
	}
	// Copy class columns verbatim.
	for _, c := range data.AllClassAttributes() {
		inSpec, err := data.GetAttribute(c)
		if err != nil {
			return nil, err
		}
		outSpec, err := out.GetAttribute(c)
		if err != nil {
			return nil, err
		}
		for r := 0; r < rows; r++ {
			out.Set(outSpec, r, data.Get(inSpec, r))
		}
	}
	return out, nil
}

func (o *OneHotEncoder) findColumn(name string) *oneHotColumn {
	for i := range o.cols {
		if o.cols[i].Name == name {
			return &o.cols[i]
		}
	}
	return nil
}

// InverseTransform always fails: indicator columns cannot be mapped back to
// categories (an all-zero row has no category at all).
func (o *OneHotEncoder) InverseTransform(base.FixedDataGrid) (base.FixedDataGrid, error) {
	return nil, fmt.Errorf("one-hot-encoder: indicator columns cannot be mapped back to categories: %w", ErrNoInverse)
}

// oneHotState is the serialised form of a fitted OneHotEncoder.
type oneHotState struct {
	Schema  []colDef       `json:"schema"`
	Columns []oneHotColumn `json:"columns"`
}

func (o *OneHotEncoder) marshalState() ([]byte, error) {
	return json.Marshal(oneHotState{Schema: o.schema, Columns: o.cols})
}

func (o *OneHotEncoder) unmarshalState(raw []byte) error {
	var st oneHotState
	if err := json.Unmarshal(raw, &st); err != nil {
		return err
	}
	for i := range st.Columns {
		idx := make(map[string]int, len(st.Columns[i].Vocab))
		for j, v := range st.Columns[i].Vocab {
			idx[v] = j
		}
		st.Columns[i].index = idx
	}
	o.schema = st.Schema
	o.cols = st.Columns
	o.unseen = make(map[string]int)
	o.fitted = true
	return nil
}

func init() {
	registerStage("one-hot-encoder", func() Transformer { return NewOneHotEncoder() })
}
