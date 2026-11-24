package pipeline

import (
	"encoding/json"
	"fmt"
	"math"

	"github.com/sjwhitworth/golearn/base"
)

// MeanImputer fills missing (NaN) values of every non-class FloatAttribute
// column with the mean of that column computed on the training data.
//
// Degradation rule: a column that is entirely NaN on the training data has no
// computable mean; it is imputed with 0.0 and recorded in AllMissingCols so
// the behaviour is visible rather than silent. No NaN or Inf is ever
// produced.
//
// InverseTransform is not supported: imputed values cannot be distinguished
// from observed ones after the fact.
type MeanImputer struct {
	schema []colDef
	fills  map[string]float64
	// AllMissingCols lists the columns that were entirely missing at Fit
	// time and are imputed with 0.0.
	AllMissingCols []string
	fitted         bool
}

// NewMeanImputer creates an unfitted MeanImputer.
func NewMeanImputer() *MeanImputer {
	return &MeanImputer{}
}

// Type returns the serialisation identifier of this stage.
func (im *MeanImputer) Type() string { return "mean-imputer" }

// String returns a human-readable summary.
func (im *MeanImputer) String() string {
	return fmt.Sprintf("MeanImputer(%d column(s))", len(im.fills))
}

// Fit computes the per-column means from train (training data only).
func (im *MeanImputer) Fit(train base.FixedDataGrid) error {
	im.schema = schemaOf(train)
	im.fills = make(map[string]float64)
	im.AllMissingCols = nil

	_, rows := train.Size()
	for _, a := range nonClassFloatAttrs(train) {
		spec, err := train.GetAttribute(a)
		if err != nil {
			return err
		}
		sum := 0.0
		n := 0
		for r := 0; r < rows; r++ {
			v := floatValue(train, spec, r)
			if math.IsNaN(v) {
				continue
			}
			sum += v
			n++
		}
		if n == 0 {
			// Documented degradation: all-missing column -> fill 0.0
			im.fills[a.GetName()] = 0.0
			im.AllMissingCols = append(im.AllMissingCols, a.GetName())
		} else {
			im.fills[a.GetName()] = sum / float64(n)
		}
	}
	im.fitted = true
	return nil
}

// Transform replaces NaN values in the fitted float columns with the training
// means. All other columns (including class columns) pass through unchanged;
// row order and count are preserved.
func (im *MeanImputer) Transform(data base.FixedDataGrid) (base.FixedDataGrid, error) {
	if !im.fitted {
		return nil, ErrNotFitted
	}
	if err := checkSchema(im.Type(), im.schema, data); err != nil {
		return nil, err
	}

	out, outSpecs := copyStructure(data)
	_, rows := data.Size()
	if err := out.Extend(rows); err != nil {
		return nil, err
	}

	inSpecs := base.ResolveAllAttributes(data)
	// Copy every column, replacing NaN in the fitted float columns.
	for i, inSpec := range inSpecs {
		name := inSpec.GetAttribute().GetName()
		fill, isFloatCol := im.fills[name]
		isFloat := inSpec.GetAttribute().GetType() == base.Float64Type
		for r := 0; r < rows; r++ {
			v := data.Get(inSpec, r)
			if isFloat && isFloatCol && math.IsNaN(base.UnpackBytesToFloat(v)) {
				out.Set(outSpecs[i], r, base.PackFloatToBytes(fill))
			} else {
				out.Set(outSpecs[i], r, v)
			}
		}
	}
	return out, nil
}

// InverseTransform always fails: imputation cannot be undone.
func (im *MeanImputer) InverseTransform(base.FixedDataGrid) (base.FixedDataGrid, error) {
	return nil, fmt.Errorf("mean-imputer: imputed values cannot be recovered: %w", ErrNoInverse)
}

// imputerState is the serialised form of a fitted MeanImputer.
type imputerState struct {
	Schema         []colDef           `json:"schema"`
	Fills          map[string]float64 `json:"fills"`
	AllMissingCols []string           `json:"all_missing_cols,omitempty"`
}

func (im *MeanImputer) marshalState() ([]byte, error) {
	return json.Marshal(imputerState{
		Schema:         im.schema,
		Fills:          im.fills,
		AllMissingCols: im.AllMissingCols,
	})
}

func (im *MeanImputer) unmarshalState(raw []byte) error {
	var st imputerState
	if err := json.Unmarshal(raw, &st); err != nil {
		return err
	}
	if st.Fills == nil {
		st.Fills = make(map[string]float64)
	}
	im.schema = st.Schema
	im.fills = st.Fills
	im.AllMissingCols = st.AllMissingCols
	im.fitted = true
	return nil
}

func init() {
	registerStage("mean-imputer", func() Transformer { return NewMeanImputer() })
}
