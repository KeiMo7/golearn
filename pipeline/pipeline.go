// Package pipeline provides a reusable, serialisable feature-engineering
// chain for GoLearn.
//
// A Pipeline is an ordered list of Transformer stages (imputation, scaling,
// one-hot encoding, ...). Each stage is fitted on training data only and can
// then transform any data with the same schema, so exactly the same
// transformation is applied during training and prediction. Fitted state can
// be saved to disk and restored in another process with identical results.
//
// Design rules (stable across versions):
//
//   - Fitting only ever sees the data passed to Fit. In cross-validation each
//     fold must re-fit its own Pipeline on the fold's training rows; use
//     CrossValidate which does this for you.
//   - Class (target) Attributes are never transformed. They are passed
//     through unchanged when present, and may be absent at prediction time.
//   - Row order and row count are always preserved.
//   - Missing numeric values are represented as NaN inside FloatAttribute
//     columns. Stages never silently reorder the steps declared by the user.
//   - Stages are applied strictly in declaration order; the output of one
//     stage is the input of the next.
package pipeline

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"

	"github.com/sjwhitworth/golearn/base"
)

// FormatVersion is the serialisation version written by Save and accepted by
// LoadPipeline. It is bumped whenever the on-disk format changes.
const FormatVersion = 1

// ErrNotFitted is returned when Transform or InverseTransform is called on a
// Transformer (or Pipeline) that has not been fitted yet.
var ErrNotFitted = errors.New("pipeline: transformer is not fitted")

// ErrNoInverse is returned by stages which cannot undo their transformation
// (imputation, one-hot encoding).
var ErrNoInverse = errors.New("pipeline: this transformer does not support inverse transformation")

// colDef describes one non-class input column: its name and its Attribute
// type (base.Float64Type, base.CategoricalType, base.BinaryType).
type colDef struct {
	Name string `json:"name"`
	Type int    `json:"type"`
}

// nonClassAttrs returns the non-class Attributes of data in a stable,
// deterministic order (the order of AllAttributes). It does not use
// base.NonClassAttributes, whose result order is explicitly unspecified
// (it is built from a map) and would make schemas non-deterministic.
func nonClassAttrs(data base.FixedDataGrid) []base.Attribute {
	classSet := make(map[base.Attribute]bool)
	for _, c := range data.AllClassAttributes() {
		classSet[c] = true
	}
	ret := make([]base.Attribute, 0)
	for _, a := range data.AllAttributes() {
		if !classSet[a] {
			ret = append(ret, a)
		}
	}
	return ret
}

// nonClassFloatAttrs returns the non-class FloatAttributes of data in a
// stable order.
func nonClassFloatAttrs(data base.FixedDataGrid) []base.Attribute {
	ret := make([]base.Attribute, 0)
	for _, a := range nonClassAttrs(data) {
		if _, ok := a.(*base.FloatAttribute); ok {
			ret = append(ret, a)
		}
	}
	return ret
}

// schemaOf computes the schema of a data set: the ordered name/type pairs of
// all non-class Attributes. Class Attributes are deliberately excluded so a
// fitted Transformer also accepts prediction-time data without a target
// column.
func schemaOf(data base.FixedDataGrid) []colDef {
	attrs := nonClassAttrs(data)
	ret := make([]colDef, 0, len(attrs))
	for _, a := range attrs {
		ret = append(ret, colDef{Name: a.GetName(), Type: a.GetType()})
	}
	return ret
}

// checkSchema verifies that data has exactly the non-class columns (names and
// types, in order) recorded at Fit time. A mismatch means the table has
// different dimensions or column types than the training data and the
// transformation would be meaningless.
func checkSchema(stage string, want []colDef, data base.FixedDataGrid) error {
	got := schemaOf(data)
	if len(got) != len(want) {
		return fmt.Errorf("pipeline: %s: dimension mismatch: fitted on %d non-class column(s) but input has %d",
			stage, len(want), len(got))
	}
	for i := range want {
		if got[i] != want[i] {
			return fmt.Errorf("pipeline: %s: schema mismatch at column %d: fitted on %q (type %d) but input has %q (type %d)",
				stage, i, want[i].Name, want[i].Type, got[i].Name, got[i].Type)
		}
	}
	return nil
}

// Transformer is one stage of a Pipeline. Implementations must fit their
// state from the training data passed to Fit and only that data.
type Transformer interface {
	// Fit learns the stage's state (statistics, vocabularies) from train.
	Fit(train base.FixedDataGrid) error
	// Transform applies the fitted stage to data. It returns ErrNotFitted
	// if Fit has not been called, and a descriptive error if the schema of
	// data does not match the schema seen during Fit.
	Transform(data base.FixedDataGrid) (base.FixedDataGrid, error)
	// InverseTransform undoes the transformation where this is meaningful
	// (numeric scaling). Stages which cannot be inverted return ErrNoInverse.
	InverseTransform(data base.FixedDataGrid) (base.FixedDataGrid, error)
	// Type returns the stable type identifier used in serialised form.
	Type() string
	// String returns a human-readable summary.
	String() string
	// marshalState serialises the fitted state (JSON).
	marshalState() ([]byte, error)
	// unmarshalState restores the fitted state produced by marshalState.
	unmarshalState([]byte) error
}

// stageFactory builds a fresh, unfitted Transformer of a registered type.
var stageRegistry = map[string]func() Transformer{}

func registerStage(t string, f func() Transformer) {
	stageRegistry[t] = f
}

// Pipeline is an ordered chain of Transformers. The output of stage i is the
// input of stage i+1; stages are never reordered internally.
type Pipeline struct {
	stages []Transformer
	fitted bool
}

// NewPipeline creates a Pipeline from the given stages, applied in the order
// they are declared.
func NewPipeline(stages ...Transformer) *Pipeline {
	return &Pipeline{stages: stages}
}

// Stages returns the stages of this Pipeline in application order.
func (p *Pipeline) Stages() []Transformer {
	ret := make([]Transformer, len(p.stages))
	copy(ret, p.stages)
	return ret
}

// Fit fits every stage in declaration order on train, chaining each stage's
// output into the next stage's Fit. Only train is ever inspected.
func (p *Pipeline) Fit(train base.FixedDataGrid) error {
	cur := train
	for i, s := range p.stages {
		if err := s.Fit(cur); err != nil {
			return fmt.Errorf("pipeline: fitting stage %d (%s): %w", i, s.Type(), err)
		}
		next, err := s.Transform(cur)
		if err != nil {
			return fmt.Errorf("pipeline: transforming with stage %d (%s) during fit: %w", i, s.Type(), err)
		}
		cur = next
	}
	p.fitted = true
	return nil
}

// Transform applies every fitted stage in declaration order. It returns
// ErrNotFitted when the Pipeline (or one of its stages) has not been fitted.
func (p *Pipeline) Transform(data base.FixedDataGrid) (base.FixedDataGrid, error) {
	if !p.fitted {
		return nil, ErrNotFitted
	}
	cur := data
	var err error
	for i, s := range p.stages {
		cur, err = s.Transform(cur)
		if err != nil {
			return nil, fmt.Errorf("pipeline: stage %d (%s): %w", i, s.Type(), err)
		}
	}
	return cur, nil
}

// FitTransform is Fit followed by Transform on the same data.
func (p *Pipeline) FitTransform(train base.FixedDataGrid) (base.FixedDataGrid, error) {
	if err := p.Fit(train); err != nil {
		return nil, err
	}
	return p.Transform(train)
}

// InverseTransform applies the inverse of every stage in reverse order. If
// any stage does not support inversion (imputation, one-hot encoding) it
// returns an error wrapping ErrNoInverse.
func (p *Pipeline) InverseTransform(data base.FixedDataGrid) (base.FixedDataGrid, error) {
	if !p.fitted {
		return nil, ErrNotFitted
	}
	cur := data
	var err error
	for i := len(p.stages) - 1; i >= 0; i-- {
		cur, err = p.stages[i].InverseTransform(cur)
		if err != nil {
			return nil, fmt.Errorf("pipeline: inverse of stage %d (%s): %w", i, p.stages[i].Type(), err)
		}
	}
	return cur, nil
}

// String returns a human-readable summary of the Pipeline.
func (p *Pipeline) String() string {
	s := fmt.Sprintf("Pipeline(%d stage(s), fitted=%v:", len(p.stages), p.fitted)
	for i, st := range p.stages {
		if i > 0 {
			s += " ->"
		}
		s += " " + st.String()
	}
	return s + ")"
}

// pipelineEnvelope is the on-disk serialisation format.
type pipelineEnvelope struct {
	Version int             `json:"version"`
	Stages  []stageEnvelope `json:"stages"`
}

type stageEnvelope struct {
	Type  string          `json:"type"`
	State json.RawMessage `json:"state"`
}

// Save writes the fitted Pipeline (including every stage's state) to w in a
// versioned JSON format.
func (p *Pipeline) Save(w io.Writer) error {
	if !p.fitted {
		return ErrNotFitted
	}
	env := pipelineEnvelope{Version: FormatVersion}
	for _, s := range p.stages {
		raw, err := s.marshalState()
		if err != nil {
			return fmt.Errorf("pipeline: serialising stage %s: %v", s.Type(), err)
		}
		env.Stages = append(env.Stages, stageEnvelope{Type: s.Type(), State: raw})
	}
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	return enc.Encode(env)
}

// LoadPipeline reads a Pipeline previously written with Save. The format
// version must match FormatVersion exactly; mismatched versions are rejected
// with a descriptive error rather than parsed on a best-effort basis.
func LoadPipeline(r io.Reader) (*Pipeline, error) {
	var env pipelineEnvelope
	if err := json.NewDecoder(r).Decode(&env); err != nil {
		return nil, fmt.Errorf("pipeline: cannot parse serialised pipeline: %v", err)
	}
	if env.Version != FormatVersion {
		return nil, fmt.Errorf("pipeline: unsupported format version %d (this build supports version %d); refusing to load",
			env.Version, FormatVersion)
	}
	p := &Pipeline{}
	for i, se := range env.Stages {
		factory, ok := stageRegistry[se.Type]
		if !ok {
			return nil, fmt.Errorf("pipeline: unknown stage type %q at position %d", se.Type, i)
		}
		stage := factory()
		if err := stage.unmarshalState(se.State); err != nil {
			return nil, fmt.Errorf("pipeline: deserialising stage %d (%s): %v", i, se.Type, err)
		}
		p.stages = append(p.stages, stage)
	}
	p.fitted = true
	return p, nil
}

// copyStructure builds a DenseInstances with the same Attributes (including
// class Attributes and their markers) as src, but no rows allocated. The
// returned specs align with base.ResolveAllAttributes(src) order.
func copyStructure(src base.FixedDataGrid) (*base.DenseInstances, []base.AttributeSpec) {
	ret := base.NewDenseInstances()
	specs := make([]base.AttributeSpec, 0)
	for _, a := range src.AllAttributes() {
		specs = append(specs, ret.AddAttribute(a))
	}
	for _, c := range src.AllClassAttributes() {
		if err := ret.AddClassAttribute(c); err != nil {
			panic(fmt.Sprintf("pipeline: cannot mark class attribute: %v", err))
		}
	}
	return ret, specs
}

// floatValue reads the float64 stored at (spec, row).
func floatValue(data base.FixedDataGrid, spec base.AttributeSpec, row int) float64 {
	return base.UnpackBytesToFloat(data.Get(spec, row))
}
