package pipeline

import (
	"encoding/json"
	"fmt"
	"math"

	"github.com/sjwhitworth/golearn/base"
)

// StandardScaler standardises every non-class FloatAttribute column to zero
// mean and unit variance using statistics computed on the training data:
//
//	y = (x - mean) / std
//
// where std is the population standard deviation. NaN values pass through
// unchanged and are excluded from the statistics, so place a MeanImputer
// earlier in the chain if the data can contain missing values.
//
// Zero-variance rule: a column whose training variance is 0 is transformed to
// 0.0 everywhere (and inverted back to the training mean). No NaN or Inf is
// ever produced.
//
// InverseTransform restores the original scale: x = y * std + mean.
type StandardScaler struct {
	schema []colDef
	means  map[string]float64
	stds   map[string]float64
	fitted bool
}

// NewStandardScaler creates an unfitted StandardScaler.
func NewStandardScaler() *StandardScaler {
	return &StandardScaler{}
}

// Type returns the serialisation identifier of this stage.
func (s *StandardScaler) Type() string { return "standard-scaler" }

// String returns a human-readable summary.
func (s *StandardScaler) String() string {
	return fmt.Sprintf("StandardScaler(%d column(s))", len(s.means))
}

// Fit computes per-column means and standard deviations from train.
func (s *StandardScaler) Fit(train base.FixedDataGrid) error {
	s.schema = schemaOf(train)
	s.means = make(map[string]float64)
	s.stds = make(map[string]float64)

	_, rows := train.Size()
	for _, a := range nonClassFloatAttrs(train) {
		spec, err := train.GetAttribute(a)
		if err != nil {
			return err
		}
		sum, sumSq := 0.0, 0.0
		n := 0
		for r := 0; r < rows; r++ {
			v := floatValue(train, spec, r)
			if math.IsNaN(v) {
				continue
			}
			sum += v
			sumSq += v * v
			n++
		}
		mean, std := 0.0, 0.0
		if n > 0 {
			mean = sum / float64(n)
			variance := sumSq/float64(n) - mean*mean
			if variance > 0 {
				std = math.Sqrt(variance)
			}
		}
		s.means[a.GetName()] = mean
		s.stds[a.GetName()] = std
	}
	s.fitted = true
	return nil
}

// Means returns the fitted per-column training means (for inspection/tests).
func (s *StandardScaler) Means() map[string]float64 {
	ret := make(map[string]float64, len(s.means))
	for k, v := range s.means {
		ret[k] = v
	}
	return ret
}

// Stds returns the fitted per-column training standard deviations.
func (s *StandardScaler) Stds() map[string]float64 {
	ret := make(map[string]float64, len(s.stds))
	for k, v := range s.stds {
		ret[k] = v
	}
	return ret
}

// Transform standardises the fitted float columns. Other columns (including
// class columns) pass through unchanged; row order and count are preserved.
func (s *StandardScaler) Transform(data base.FixedDataGrid) (base.FixedDataGrid, error) {
	if !s.fitted {
		return nil, ErrNotFitted
	}
	if err := checkSchema(s.Type(), s.schema, data); err != nil {
		return nil, err
	}
	return s.apply(data, false)
}

// InverseTransform maps standardised values back to the original scale.
func (s *StandardScaler) InverseTransform(data base.FixedDataGrid) (base.FixedDataGrid, error) {
	if !s.fitted {
		return nil, ErrNotFitted
	}
	if err := checkSchema(s.Type(), s.schema, data); err != nil {
		return nil, err
	}
	return s.apply(data, true)
}

func (s *StandardScaler) apply(data base.FixedDataGrid, inverse bool) (base.FixedDataGrid, error) {
	out, outSpecs := copyStructure(data)
	_, rows := data.Size()
	if err := out.Extend(rows); err != nil {
		return nil, err
	}

	inSpecs := base.ResolveAllAttributes(data)
	for i, inSpec := range inSpecs {
		attr := inSpec.GetAttribute()
		mean, isFloatCol := s.means[attr.GetName()]
		std := s.stds[attr.GetName()]
		isFloat := attr.GetType() == base.Float64Type
		for r := 0; r < rows; r++ {
			v := data.Get(inSpec, r)
			if !(isFloat && isFloatCol) {
				out.Set(outSpecs[i], r, v)
				continue
			}
			f := base.UnpackBytesToFloat(v)
			if math.IsNaN(f) {
				out.Set(outSpecs[i], r, v) // NaN passes through untouched
				continue
			}
			var y float64
			if !inverse {
				if std > 0 {
					y = (f - mean) / std
				} else {
					y = 0.0 // zero-variance column: documented rule
				}
			} else {
				y = f*std + mean // std == 0 naturally restores the mean
			}
			out.Set(outSpecs[i], r, base.PackFloatToBytes(y))
		}
	}
	return out, nil
}

// standardScalerState is the serialised form of a fitted StandardScaler.
type standardScalerState struct {
	Schema []colDef           `json:"schema"`
	Means  map[string]float64 `json:"means"`
	Stds   map[string]float64 `json:"stds"`
}

func (s *StandardScaler) marshalState() ([]byte, error) {
	return json.Marshal(standardScalerState{Schema: s.schema, Means: s.means, Stds: s.stds})
}

func (s *StandardScaler) unmarshalState(raw []byte) error {
	var st standardScalerState
	if err := json.Unmarshal(raw, &st); err != nil {
		return err
	}
	s.schema = st.Schema
	s.means = st.Means
	s.stds = st.Stds
	s.fitted = true
	return nil
}

// MinMaxScaler scales every non-class FloatAttribute column into the range
// [0, 1] using the minimum and maximum observed on the training data:
//
//	y = (x - min) / (max - min)
//
// NaN values pass through unchanged and are excluded from the statistics.
//
// Zero-range rule: a constant training column (max == min) is transformed to
// 0.0 everywhere (and inverted back to the training minimum). No NaN or Inf
// is ever produced. Values outside the training range at transform time are
// scaled linearly and may fall outside [0, 1]; this is intentional.
//
// InverseTransform restores the original scale: x = y * (max - min) + min.
type MinMaxScaler struct {
	schema []colDef
	mins   map[string]float64
	maxs   map[string]float64
	fitted bool
}

// NewMinMaxScaler creates an unfitted MinMaxScaler.
func NewMinMaxScaler() *MinMaxScaler {
	return &MinMaxScaler{}
}

// Type returns the serialisation identifier of this stage.
func (m *MinMaxScaler) Type() string { return "minmax-scaler" }

// String returns a human-readable summary.
func (m *MinMaxScaler) String() string {
	return fmt.Sprintf("MinMaxScaler(%d column(s))", len(m.mins))
}

// Fit computes per-column minima and maxima from train.
func (m *MinMaxScaler) Fit(train base.FixedDataGrid) error {
	m.schema = schemaOf(train)
	m.mins = make(map[string]float64)
	m.maxs = make(map[string]float64)

	_, rows := train.Size()
	for _, a := range nonClassFloatAttrs(train) {
		spec, err := train.GetAttribute(a)
		if err != nil {
			return err
		}
		min, max := 0.0, 0.0
		n := 0
		for r := 0; r < rows; r++ {
			v := floatValue(train, spec, r)
			if math.IsNaN(v) {
				continue
			}
			if n == 0 || v < min {
				min = v
			}
			if n == 0 || v > max {
				max = v
			}
			n++
		}
		m.mins[a.GetName()] = min
		m.maxs[a.GetName()] = max
	}
	m.fitted = true
	return nil
}

// Transform scales the fitted float columns into [0, 1]. Other columns
// (including class columns) pass through unchanged.
func (m *MinMaxScaler) Transform(data base.FixedDataGrid) (base.FixedDataGrid, error) {
	if !m.fitted {
		return nil, ErrNotFitted
	}
	if err := checkSchema(m.Type(), m.schema, data); err != nil {
		return nil, err
	}
	return m.apply(data, false)
}

// InverseTransform maps scaled values back to the original scale.
func (m *MinMaxScaler) InverseTransform(data base.FixedDataGrid) (base.FixedDataGrid, error) {
	if !m.fitted {
		return nil, ErrNotFitted
	}
	if err := checkSchema(m.Type(), m.schema, data); err != nil {
		return nil, err
	}
	return m.apply(data, true)
}

func (m *MinMaxScaler) apply(data base.FixedDataGrid, inverse bool) (base.FixedDataGrid, error) {
	out, outSpecs := copyStructure(data)
	_, rows := data.Size()
	if err := out.Extend(rows); err != nil {
		return nil, err
	}

	inSpecs := base.ResolveAllAttributes(data)
	for i, inSpec := range inSpecs {
		attr := inSpec.GetAttribute()
		min, isFloatCol := m.mins[attr.GetName()]
		max := m.maxs[attr.GetName()]
		rng := max - min
		isFloat := attr.GetType() == base.Float64Type
		for r := 0; r < rows; r++ {
			v := data.Get(inSpec, r)
			if !(isFloat && isFloatCol) {
				out.Set(outSpecs[i], r, v)
				continue
			}
			f := base.UnpackBytesToFloat(v)
			if math.IsNaN(f) {
				out.Set(outSpecs[i], r, v)
				continue
			}
			var y float64
			if !inverse {
				if rng > 0 {
					y = (f - min) / rng
				} else {
					y = 0.0 // zero-range column: documented rule
				}
			} else {
				y = f*rng + min // rng == 0 naturally restores the minimum
			}
			out.Set(outSpecs[i], r, base.PackFloatToBytes(y))
		}
	}
	return out, nil
}

// minMaxScalerState is the serialised form of a fitted MinMaxScaler.
type minMaxScalerState struct {
	Schema []colDef           `json:"schema"`
	Mins   map[string]float64 `json:"mins"`
	Maxs   map[string]float64 `json:"maxs"`
}

func (m *MinMaxScaler) marshalState() ([]byte, error) {
	return json.Marshal(minMaxScalerState{Schema: m.schema, Mins: m.mins, Maxs: m.maxs})
}

func (m *MinMaxScaler) unmarshalState(raw []byte) error {
	var st minMaxScalerState
	if err := json.Unmarshal(raw, &st); err != nil {
		return err
	}
	m.schema = st.Schema
	m.mins = st.Mins
	m.maxs = st.Maxs
	m.fitted = true
	return nil
}

func init() {
	registerStage("standard-scaler", func() Transformer { return NewStandardScaler() })
	registerStage("minmax-scaler", func() Transformer { return NewMinMaxScaler() })
}
