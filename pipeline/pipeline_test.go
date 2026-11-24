package pipeline

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"os/exec"
	"strings"
	"testing"

	"github.com/sjwhitworth/golearn/base"
	"github.com/sjwhitworth/golearn/evaluation"
	"github.com/sjwhitworth/golearn/trees"
	. "github.com/smartystreets/goconvey/convey"
)

//
// Test data construction helpers
//

type testCol struct {
	name  string
	kind  string // "float" | "cat" | "class"
	fvals []float64
	cvals []string
}

func floatCol(name string, vals ...float64) testCol {
	return testCol{name: name, kind: "float", fvals: vals}
}

func catCol(name string, vals ...string) testCol {
	return testCol{name: name, kind: "cat", cvals: vals}
}

func classCol(name string, vals ...string) testCol {
	return testCol{name: name, kind: "class", cvals: vals}
}

// buildInstances constructs a DenseInstances with the given columns. NaN in a
// float column marks a missing value.
func buildInstances(cols ...testCol) *base.DenseInstances {
	rows := 0
	if len(cols) > 0 {
		if cols[0].kind == "float" {
			rows = len(cols[0].fvals)
		} else {
			rows = len(cols[0].cvals)
		}
	}
	inst := base.NewDenseInstances()
	specs := make([]base.AttributeSpec, len(cols))
	catAttrs := make([]*base.CategoricalAttribute, len(cols))
	for i, c := range cols {
		var a base.Attribute
		if c.kind == "float" {
			a = base.NewFloatAttribute(c.name)
		} else {
			ca := base.NewCategoricalAttribute()
			ca.SetName(c.name)
			a = ca
			catAttrs[i] = ca
		}
		specs[i] = inst.AddAttribute(a)
		if c.kind == "class" {
			if err := inst.AddClassAttribute(a); err != nil {
				panic(err)
			}
		}
	}
	if err := inst.Extend(rows); err != nil {
		panic(err)
	}
	for i, c := range cols {
		for r := 0; r < rows; r++ {
			if c.kind == "float" {
				inst.Set(specs[i], r, base.PackFloatToBytes(c.fvals[r]))
			} else {
				inst.Set(specs[i], r, catAttrs[i].GetSysValFromString(c.cvals[r]))
			}
		}
	}
	return inst
}

func colFloats(data base.FixedDataGrid, name string) []float64 {
	a := base.GetAttributeByName(data, name)
	if a == nil {
		panic(fmt.Sprintf("no such attribute: %s", name))
	}
	spec, err := data.GetAttribute(a)
	if err != nil {
		panic(err)
	}
	_, rows := data.Size()
	ret := make([]float64, rows)
	for r := 0; r < rows; r++ {
		ret[r] = base.UnpackBytesToFloat(data.Get(spec, r))
	}
	return ret
}

func colStrings(data base.FixedDataGrid, name string) []string {
	a := base.GetAttributeByName(data, name)
	if a == nil {
		panic(fmt.Sprintf("no such attribute: %s", name))
	}
	ca, ok := a.(*base.CategoricalAttribute)
	if !ok {
		panic(fmt.Sprintf("not a categorical attribute: %s", name))
	}
	spec, err := data.GetAttribute(a)
	if err != nil {
		panic(err)
	}
	_, rows := data.Size()
	ret := make([]string, rows)
	for r := 0; r < rows; r++ {
		ret[r] = ca.GetStringFromSysVal(data.Get(spec, r))
	}
	return ret
}

// instancesIdentical compares two data sets value-by-value (and by attribute
// names/types and class attribute count).
func instancesIdentical(a, b base.FixedDataGrid) bool {
	ac, ar := a.Size()
	bc, br := b.Size()
	if ac != bc || ar != br {
		return false
	}
	aAttrs, bAttrs := a.AllAttributes(), b.AllAttributes()
	for i := range aAttrs {
		if aAttrs[i].GetName() != bAttrs[i].GetName() || aAttrs[i].GetType() != bAttrs[i].GetType() {
			return false
		}
	}
	if len(a.AllClassAttributes()) != len(b.AllClassAttributes()) {
		return false
	}
	aSpecs := base.ResolveAllAttributes(a)
	bSpecs := base.ResolveAllAttributes(b)
	for r := 0; r < ar; r++ {
		for i := range aSpecs {
			if !bytes.Equal(a.Get(aSpecs[i], r), b.Get(bSpecs[i], r)) {
				return false
			}
		}
	}
	return true
}

func anyNaNOrInf(vals []float64) bool {
	for _, v := range vals {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return true
		}
	}
	return false
}

func almostEqual(a, b, tol float64) bool {
	return math.Abs(a-b) <= tol
}

//
// 1. Numeric columns: imputation with training statistics + both scalings
//

func TestMeanImputer(t *testing.T) {
	Convey("Given a data set with missing values in a numeric column", t, func() {
		inst := buildInstances(
			floatCol("a", 1.0, math.NaN(), 3.0, 5.0),
			floatCol("b", 10.0, 20.0, 30.0, 40.0),
			classCol("cls", "x", "y", "x", "y"),
		)

		imp := NewMeanImputer()
		So(imp.Fit(inst), ShouldBeNil)
		out, err := imp.Transform(inst)
		So(err, ShouldBeNil)

		Convey("Missing values are filled with the training mean", func() {
			So(colFloats(out, "a"), ShouldResemble, []float64{1.0, 3.0, 3.0, 5.0})
		})
		Convey("Complete columns and the class column pass through unchanged", func() {
			So(colFloats(out, "b"), ShouldResemble, []float64{10.0, 20.0, 30.0, 40.0})
			So(colStrings(out, "cls"), ShouldResemble, []string{"x", "y", "x", "y"})
		})
		Convey("Row count is preserved", func() {
			_, rows := out.Size()
			So(rows, ShouldEqual, 4)
		})
	})

	Convey("Given a numeric column that is entirely missing", t, func() {
		inst := buildInstances(
			floatCol("a", math.NaN(), math.NaN(), math.NaN()),
			floatCol("b", 1.0, 2.0, 3.0),
		)
		imp := NewMeanImputer()
		So(imp.Fit(inst), ShouldBeNil)
		out, err := imp.Transform(inst)
		So(err, ShouldBeNil)

		Convey("It degrades to a documented 0.0 fill, recorded in AllMissingCols, without producing NaN", func() {
			So(imp.AllMissingCols, ShouldResemble, []string{"a"})
			So(colFloats(out, "a"), ShouldResemble, []float64{0.0, 0.0, 0.0})
			So(anyNaNOrInf(colFloats(out, "a")), ShouldBeFalse)
		})
	})
}

func TestStandardScaler(t *testing.T) {
	Convey("Given a numeric data set", t, func() {
		inst := buildInstances(
			floatCol("a", 1.0, 2.0, 3.0),
			floatCol("b", 0.0, 10.0, 20.0),
		)
		sc := NewStandardScaler()
		So(sc.Fit(inst), ShouldBeNil)
		out, err := sc.Transform(inst)
		So(err, ShouldBeNil)

		Convey("Columns are standardised to zero mean and unit variance", func() {
			// mean 2, population std sqrt(2/3)
			std := math.Sqrt(2.0 / 3.0)
			want := []float64{(1.0 - 2.0) / std, 0.0, (3.0 - 2.0) / std}
			for i, v := range colFloats(out, "a") {
				So(almostEqual(v, want[i], 1e-12), ShouldBeTrue)
			}
		})
		Convey("InverseTransform restores the original scale within tolerance", func() {
			back, err := sc.InverseTransform(out)
			So(err, ShouldBeNil)
			for i, v := range colFloats(back, "a") {
				So(almostEqual(v, []float64{1.0, 2.0, 3.0}[i], 1e-9), ShouldBeTrue)
			}
			for i, v := range colFloats(back, "b") {
				So(almostEqual(v, []float64{0.0, 10.0, 20.0}[i], 1e-9), ShouldBeTrue)
			}
		})
	})
}

func TestMinMaxScaler(t *testing.T) {
	Convey("Given a numeric data set", t, func() {
		inst := buildInstances(
			floatCol("a", 1.0, 2.0, 3.0),
		)
		mm := NewMinMaxScaler()
		So(mm.Fit(inst), ShouldBeNil)
		out, err := mm.Transform(inst)
		So(err, ShouldBeNil)

		Convey("Columns are scaled into [0, 1]", func() {
			So(colFloats(out, "a"), ShouldResemble, []float64{0.0, 0.5, 1.0})
		})
		Convey("InverseTransform restores the original scale within tolerance", func() {
			back, err := mm.InverseTransform(out)
			So(err, ShouldBeNil)
			for i, v := range colFloats(back, "a") {
				So(almostEqual(v, []float64{1.0, 2.0, 3.0}[i], 1e-9), ShouldBeTrue)
			}
		})
	})
}

//
// 2. Zero-variance columns must not produce Inf or NaN
//

func TestZeroVarianceColumns(t *testing.T) {
	Convey("Given a constant numeric column", t, func() {
		inst := buildInstances(
			floatCol("const", 5.0, 5.0, 5.0),
			floatCol("var", 1.0, 2.0, 3.0),
		)

		Convey("StandardScaler maps it to 0.0 without NaN/Inf and inverts back to the mean", func() {
			sc := NewStandardScaler()
			So(sc.Fit(inst), ShouldBeNil)
			out, err := sc.Transform(inst)
			So(err, ShouldBeNil)
			So(colFloats(out, "const"), ShouldResemble, []float64{0.0, 0.0, 0.0})
			So(anyNaNOrInf(colFloats(out, "const")), ShouldBeFalse)
			So(anyNaNOrInf(colFloats(out, "var")), ShouldBeFalse)

			back, err := sc.InverseTransform(out)
			So(err, ShouldBeNil)
			So(colFloats(back, "const"), ShouldResemble, []float64{5.0, 5.0, 5.0})
		})

		Convey("MinMaxScaler maps it to 0.0 without NaN/Inf and inverts back to the minimum", func() {
			mm := NewMinMaxScaler()
			So(mm.Fit(inst), ShouldBeNil)
			out, err := mm.Transform(inst)
			So(err, ShouldBeNil)
			So(colFloats(out, "const"), ShouldResemble, []float64{0.0, 0.0, 0.0})
			So(anyNaNOrInf(colFloats(out, "const")), ShouldBeFalse)

			back, err := mm.InverseTransform(out)
			So(err, ShouldBeNil)
			So(colFloats(back, "const"), ShouldResemble, []float64{5.0, 5.0, 5.0})
		})
	})
}

//
// 3 + 4 + 5. Categorical expansion, unseen categories, naming
//

func TestOneHotEncoder(t *testing.T) {
	Convey("Given a mixed data set with a categorical column", t, func() {
		inst := buildInstances(
			floatCol("n", 1.0, 2.0, 3.0, 4.0),
			catCol("color", "red", "green", "red", "blue"),
			classCol("cls", "x", "y", "x", "y"),
		)

		enc := NewOneHotEncoder()
		So(enc.Fit(inst), ShouldBeNil)
		out, err := enc.Transform(inst)
		So(err, ShouldBeNil)

		Convey("The vocabulary is fixed at fit time (sorted) and expanded into indicator columns", func() {
			So(enc.Vocabulary("color"), ShouldResemble, []string{"blue", "green", "red"})
			So(enc.OutputNames("color"), ShouldResemble, []string{"color_blue", "color_green", "color_red"})
			So(colFloats(out, "color_red"), ShouldResemble, []float64{1.0, 0.0, 1.0, 0.0})
			So(colFloats(out, "color_green"), ShouldResemble, []float64{0.0, 1.0, 0.0, 0.0})
			So(colFloats(out, "color_blue"), ShouldResemble, []float64{0.0, 0.0, 0.0, 1.0})
		})
		Convey("Numeric columns and the class column pass through; rows are preserved", func() {
			So(colFloats(out, "n"), ShouldResemble, []float64{1.0, 2.0, 3.0, 4.0})
			So(colStrings(out, "cls"), ShouldResemble, []string{"x", "y", "x", "y"})
			_, rows := out.Size()
			So(rows, ShouldEqual, 4)
			So(len(out.AllClassAttributes()), ShouldEqual, 1)
		})

		Convey("Unseen categories at prediction time produce all-zero rows and are counted, not errors", func() {
			newData := buildInstances(
				floatCol("n", 5.0, 6.0, 7.0),
				catCol("color", "yellow", "red", "purple"),
			)
			outNew, err := enc.Transform(newData)
			So(err, ShouldBeNil)
			// "yellow" and "purple" are unseen: all-zero indicator rows
			So(colFloats(outNew, "color_red"), ShouldResemble, []float64{0.0, 1.0, 0.0})
			So(colFloats(outNew, "color_green"), ShouldResemble, []float64{0.0, 0.0, 0.0})
			So(colFloats(outNew, "color_blue"), ShouldResemble, []float64{0.0, 0.0, 0.0})
			So(enc.UnseenCounts(), ShouldResemble, map[string]int{"color": 2})
		})

		Convey("InverseTransform is explicitly not supported", func() {
			_, err := enc.InverseTransform(out)
			So(errors.Is(err, ErrNoInverse), ShouldBeTrue)
		})
	})

	Convey("Given a categorical column whose generated names collide with existing columns", t, func() {
		inst := buildInstances(
			floatCol("color_red", 1.0, 2.0),
			catCol("color", "red", "blue"),
		)
		enc := NewOneHotEncoder()
		So(enc.Fit(inst), ShouldBeNil)

		Convey("The naming rule is stable and predictable: a numeric suffix is appended", func() {
			So(enc.OutputNames("color"), ShouldResemble, []string{"color_blue", "color_red_2"})
			// Deterministic: fitting a fresh encoder on the same schema gives the same names
			enc2 := NewOneHotEncoder()
			So(enc2.Fit(inst), ShouldBeNil)
			So(enc2.OutputNames("color"), ShouldResemble, enc.OutputNames("color"))
		})
	})

	Convey("Given a categorical column with an empty vocabulary", t, func() {
		// No cell is ever set and the attribute has no values: the column
		// is entirely missing.
		inst := base.NewDenseInstances()
		ca := base.NewCategoricalAttribute()
		ca.SetName("empty")
		inst.AddAttribute(ca)
		fa := base.NewFloatAttribute("n")
		fspec := inst.AddAttribute(fa)
		So(inst.Extend(3), ShouldBeNil)
		for r := 0; r < 3; r++ {
			inst.Set(fspec, r, base.PackFloatToBytes(float64(r)))
		}

		Convey("Fit fails explicitly instead of producing a degenerate table", func() {
			enc := NewOneHotEncoder()
			err := enc.Fit(inst)
			So(err, ShouldNotBeNil)
			So(err.Error(), ShouldContainSubstring, "empty vocabulary")
		})
	})
}

//
// 6 + 7. Chained pipelines: declared order, class/row preservation
//

func TestPipelineChaining(t *testing.T) {
	Convey("Given a mixed data set with missing values", t, func() {
		inst := buildInstances(
			floatCol("a", 1.0, math.NaN(), 3.0, 5.0),
			catCol("color", "red", "green", "red", "blue"),
			classCol("cls", "x", "y", "x", "y"),
		)

		Convey("A pipeline applies imputation, scaling and encoding in declaration order", func() {
			p := NewPipeline(NewMeanImputer(), NewStandardScaler(), NewOneHotEncoder())
			out, err := p.FitTransform(inst)
			So(err, ShouldBeNil)

			// Manual composition: impute (mean 3 -> [1,3,3,5]), then
			// standardise the imputed column (mean 3, std sqrt(2))
			mean, std := 3.0, math.Sqrt(2.0)
			want := []float64{(1.0 - mean) / std, 0.0, 0.0, (5.0 - mean) / std}
			for i, v := range colFloats(out, "a") {
				So(almostEqual(v, want[i], 1e-12), ShouldBeTrue)
			}
			// Encoding happened after scaling and saw the scaled table
			So(colFloats(out, "color_red"), ShouldResemble, []float64{1.0, 0.0, 1.0, 0.0})
			// Class column untouched, row order and count preserved
			So(colStrings(out, "cls"), ShouldResemble, []string{"x", "y", "x", "y"})
			_, rows := out.Size()
			So(rows, ShouldEqual, 4)
		})

		Convey("The declared order is honoured, never reordered internally", func() {
			// Two scalers in different orders give different results, and
			// each pipeline must match the manual sequential composition.
			only := buildInstances(floatCol("a", 1.0, 2.0, 3.0))

			pAB := NewPipeline(NewStandardScaler(), NewMinMaxScaler())
			outAB, err := pAB.FitTransform(only)
			So(err, ShouldBeNil)
			pBA := NewPipeline(NewMinMaxScaler(), NewStandardScaler())
			outBA, err := pBA.FitTransform(only)
			So(err, ShouldBeNil)

			// Manual: standardise, then min-max over the standardised values
			mean, std := 2.0, math.Sqrt(2.0/3.0)
			z := []float64{(1.0 - mean) / std, 0.0, (3.0 - mean) / std}
			zmin, zmax := z[0], z[2]
			for i, v := range colFloats(outAB, "a") {
				So(almostEqual(v, (z[i]-zmin)/(zmax-zmin), 1e-12), ShouldBeTrue)
			}
			// Manual: min-max, then standardise over [0, 0.5, 1]
			mm := []float64{0.0, 0.5, 1.0}
			mmean := 0.5
			mstd := math.Sqrt((0.25 + 0.0 + 0.25) / 3.0)
			for i, v := range colFloats(outBA, "a") {
				So(almostEqual(v, (mm[i]-mmean)/mstd, 1e-12), ShouldBeTrue)
			}
			// And the two orders genuinely differ
			So(almostEqual(colFloats(outAB, "a")[0], colFloats(outBA, "a")[0], 1e-9), ShouldBeFalse)
		})
	})
}

//
// Failure modes: unfitted transformers and schema mismatches
//

func TestFailureModes(t *testing.T) {
	Convey("Given unfitted transformers", t, func() {
		inst := buildInstances(floatCol("a", 1.0, 2.0))

		Convey("Transform returns ErrNotFitted", func() {
			_, err := NewMeanImputer().Transform(inst)
			So(errors.Is(err, ErrNotFitted), ShouldBeTrue)
			_, err = NewStandardScaler().Transform(inst)
			So(errors.Is(err, ErrNotFitted), ShouldBeTrue)
			_, err = NewMinMaxScaler().Transform(inst)
			So(errors.Is(err, ErrNotFitted), ShouldBeTrue)
			_, err = NewOneHotEncoder().Transform(inst)
			So(errors.Is(err, ErrNotFitted), ShouldBeTrue)
		})
		Convey("Pipeline.Transform and InverseTransform return ErrNotFitted", func() {
			p := NewPipeline(NewMeanImputer())
			_, err := p.Transform(inst)
			So(errors.Is(err, ErrNotFitted), ShouldBeTrue)
			_, err = p.InverseTransform(inst)
			So(errors.Is(err, ErrNotFitted), ShouldBeTrue)
		})
	})

	Convey("Given a fitted transformer and a table with mismatched dimensions", t, func() {
		train := buildInstances(
			floatCol("a", 1.0, 2.0),
			floatCol("b", 3.0, 4.0),
		)
		sc := NewStandardScaler()
		So(sc.Fit(train), ShouldBeNil)

		Convey("Transform fails with an identifiable, descriptive error", func() {
			tooFew := buildInstances(floatCol("a", 1.0, 2.0))
			_, err := sc.Transform(tooFew)
			So(err, ShouldNotBeNil)
			So(err.Error(), ShouldContainSubstring, "dimension mismatch")

			tooMany := buildInstances(
				floatCol("a", 1.0, 2.0),
				floatCol("b", 3.0, 4.0),
				floatCol("c", 5.0, 6.0),
			)
			_, err = sc.Transform(tooMany)
			So(err, ShouldNotBeNil)
			So(err.Error(), ShouldContainSubstring, "dimension mismatch")

			wrongType := buildInstances(
				floatCol("a", 1.0, 2.0),
				catCol("b", "u", "v"),
			)
			_, err = sc.Transform(wrongType)
			So(err, ShouldNotBeNil)
			So(err.Error(), ShouldContainSubstring, "schema mismatch")
		})
	})

	Convey("Inverse transformation of non-invertible stages fails explicitly", t, func() {
		inst := buildInstances(floatCol("a", 1.0, 2.0))
		imp := NewMeanImputer()
		So(imp.Fit(inst), ShouldBeNil)
		_, err := imp.InverseTransform(inst)
		So(errors.Is(err, ErrNoInverse), ShouldBeTrue)

		p := NewPipeline(NewMeanImputer(), NewStandardScaler())
		So(p.Fit(inst), ShouldBeNil)
		_, err = p.InverseTransform(inst)
		So(errors.Is(err, ErrNoInverse), ShouldBeTrue)
	})
}

//
// 8. Serialisation: versioned format, identical output after restore
//

func serialiseToString(p *Pipeline) string {
	var buf bytes.Buffer
	if err := p.Save(&buf); err != nil {
		panic(err)
	}
	return buf.String()
}

func TestSerialization(t *testing.T) {
	Convey("Given a fitted pipeline on mixed data", t, func() {
		inst := buildInstances(
			floatCol("a", 1.0, math.NaN(), 3.0, 5.0),
			floatCol("b", 0.0, 10.0, 20.0, 30.0),
			catCol("color", "red", "green", "red", "blue"),
			classCol("cls", "x", "y", "x", "y"),
		)
		p := NewPipeline(NewMeanImputer(), NewStandardScaler(), NewOneHotEncoder())
		So(p.Fit(inst), ShouldBeNil)
		want, err := p.Transform(inst)
		So(err, ShouldBeNil)

		Convey("A save/load round trip produces value-identical output", func() {
			var buf bytes.Buffer
			So(p.Save(&buf), ShouldBeNil)
			p2, err := LoadPipeline(&buf)
			So(err, ShouldBeNil)
			got, err := p2.Transform(inst)
			So(err, ShouldBeNil)
			So(instancesIdentical(want, got), ShouldBeTrue)
		})

		Convey("Saving an unfitted pipeline fails", func() {
			var buf bytes.Buffer
			err := NewPipeline(NewMeanImputer()).Save(&buf)
			So(errors.Is(err, ErrNotFitted), ShouldBeTrue)
		})

		Convey("A version mismatch is rejected with a descriptive error", func() {
			saved := serialiseToString(p)
			var env map[string]interface{}
			So(json.Unmarshal([]byte(saved), &env), ShouldBeNil)
			env["version"] = 999.0
			raw, err := json.Marshal(env)
			So(err, ShouldBeNil)
			_, err = LoadPipeline(bytes.NewReader(raw))
			So(err, ShouldNotBeNil)
			So(err.Error(), ShouldContainSubstring, "unsupported format version 999")
		})

		Convey("An unknown stage type is rejected", func() {
			raw := `{"version": 1, "stages": [{"type": "no-such-stage", "state": {}}]}`
			_, err := LoadPipeline(strings.NewReader(raw))
			So(err, ShouldNotBeNil)
			So(err.Error(), ShouldContainSubstring, "unknown stage type")
		})
	})
}

// TestCrossProcessRestore proves that a pipeline saved in one process can be
// restored in another process and produces exactly the same output. The child
// process is this same test binary re-executed with an environment marker.
func TestCrossProcessRestore(t *testing.T) {
	if os.Getenv("GOLEARN_PIPELINE_CHILD") == "1" {
		// Child process: load the pipeline, transform the fixed input and
		// print every float cell, row by row, with full precision.
		p, err := LoadPipeline(mustOpen(os.Getenv("GOLEARN_PIPELINE_FILE")))
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		out, err := p.Transform(crossProcessInput())
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		fmt.Print(dumpFloats(out))
		os.Exit(0)
	}

	Convey("Given a pipeline fitted in this process", t, func() {
		p := NewPipeline(NewMeanImputer(), NewMinMaxScaler(), NewOneHotEncoder())
		So(p.Fit(crossProcessInput()), ShouldBeNil)
		want, err := p.Transform(crossProcessInput())
		So(err, ShouldBeNil)

		f, err := os.CreateTemp("", "golearn-pipeline-*.json")
		So(err, ShouldBeNil)
		defer os.Remove(f.Name())
		So(p.Save(f), ShouldBeNil)
		So(f.Close(), ShouldBeNil)

		Convey("Restoring it in another process yields byte-identical values", func() {
			cmd := exec.Command(os.Args[0], "-test.run=TestCrossProcessRestore")
			cmd.Env = append(os.Environ(),
				"GOLEARN_PIPELINE_CHILD=1",
				"GOLEARN_PIPELINE_FILE="+f.Name(),
			)
			childOut, err := cmd.Output()
			So(err, ShouldBeNil)
			So(string(childOut), ShouldEqual, dumpFloats(want))
		})
	})
}

func mustOpen(path string) *os.File {
	f, err := os.Open(path)
	if err != nil {
		panic(err)
	}
	return f
}

func crossProcessInput() *base.DenseInstances {
	return buildInstances(
		floatCol("a", 1.5, math.NaN(), 3.25, 5.75),
		catCol("color", "red", "green", "red", "blue"),
		classCol("cls", "x", "y", "x", "y"),
	)
}

func dumpFloats(data base.FixedDataGrid) string {
	var sb strings.Builder
	specs := base.ResolveAllAttributes(data)
	_, rows := data.Size()
	for r := 0; r < rows; r++ {
		for _, s := range specs {
			raw := data.Get(s, r)
			if s.GetAttribute().GetType() == base.Float64Type {
				sb.WriteString(fmt.Sprintf("%.17g", base.UnpackBytesToFloat(raw)))
			} else {
				sb.WriteString(s.GetAttribute().GetStringFromSysVal(raw))
			}
			sb.WriteString("\n")
		}
	}
	return sb.String()
}

//
// Leakage: fitting must only see training data
//

func TestNoLeakage(t *testing.T) {
	Convey("Given a data set whose validation rows have wildly different statistics", t, func() {
		trainVals := []float64{1.0, 1.1, 0.9, 1.0, 1.1, 0.9, 1.0, 1.1}
		full := buildInstances(
			floatCol("a", append(append([]float64{}, trainVals...), 1000.0, 2000.0)...),
			catCol("c", "p", "p", "q", "q", "p", "q", "p", "q", "unseen1", "unseen2"),
			classCol("cls", "x", "x", "x", "x", "y", "y", "y", "y", "y", "y"),
		)
		attrs := full.AllAttributes()
		trainRows := []int{0, 1, 2, 3, 4, 5, 6, 7}
		validRows := []int{8, 9}
		trainView := base.NewInstancesViewFromVisible(full, trainRows, attrs)
		validView := base.NewInstancesViewFromVisible(full, validRows, attrs)

		Convey("Statistics fitted on the training fold differ from whole-data statistics", func() {
			scTrain := NewStandardScaler()
			So(scTrain.Fit(trainView), ShouldBeNil)
			scFull := NewStandardScaler()
			So(scFull.Fit(full), ShouldBeNil)
			So(almostEqual(scTrain.Means()["a"], 1.0125, 1e-9), ShouldBeTrue)
			So(almostEqual(scFull.Means()["a"], 1.0125, 1e-9), ShouldBeFalse)
		})

		Convey("Changing validation-fold data does not change the fitted state", func() {
			scBefore := NewStandardScaler()
			So(scBefore.Fit(trainView), ShouldBeNil)

			// Overwrite the validation rows with different values
			spec, err := full.GetAttribute(base.GetAttributeByName(full, "a"))
			So(err, ShouldBeNil)
			for _, r := range validRows {
				full.Set(spec, r, base.PackFloatToBytes(5555.0))
			}

			scAfter := NewStandardScaler()
			So(scAfter.Fit(trainView), ShouldBeNil)
			So(scAfter.Means(), ShouldResemble, scBefore.Means())
			So(scAfter.Stds(), ShouldResemble, scBefore.Stds())
		})

		Convey("The vocabulary is fixed by the training fold; validation-only values are reported as unseen", func() {
			enc := NewOneHotEncoder()
			So(enc.Fit(trainView), ShouldBeNil)
			So(enc.Vocabulary("c"), ShouldResemble, []string{"p", "q"})
			_, err := enc.Transform(validView)
			So(err, ShouldBeNil)
			So(enc.UnseenCounts(), ShouldResemble, map[string]int{"c": 2})
		})
	})
}

//
// Integration: transformed data is consumable by existing models and metrics,
// and cross-validation re-fits the pipeline inside each fold.
//

func TestIntegrationWithModelsAndCV(t *testing.T) {
	Convey("Given a labelled mixed data set", t, func() {
		inst := buildInstances(
			floatCol("f1", 0.1, 0.2, 0.1, 0.3, 5.0, 5.5, 4.8, 5.2, 0.15, 5.1, 0.25, 4.9),
			floatCol("f2", 1.0, math.NaN(), 1.2, 0.8, 9.0, 8.5, 9.5, 8.8, 1.1, 9.2, 0.9, 8.7),
			catCol("kind", "a", "a", "b", "a", "b", "b", "a", "b", "a", "b", "a", "b"),
			classCol("cls", "low", "low", "low", "low", "high", "high", "high", "high", "low", "high", "low", "high"),
		)
		newPipe := func() *Pipeline {
			return NewPipeline(NewMeanImputer(), NewMinMaxScaler(), NewOneHotEncoder())
		}

		Convey("A transformed data set can be consumed by an existing classifier and evaluation metric", func() {
			out, err := newPipe().FitTransform(inst)
			So(err, ShouldBeNil)

			cls := trees.NewID3DecisionTree(0.0)
			So(cls.Fit(out), ShouldBeNil)
			pred, err := cls.Predict(out)
			So(err, ShouldBeNil)
			cf, err := evaluation.GetConfusionMatrix(out, pred)
			So(err, ShouldBeNil)
			// The classes are linearly separable; the tree should fit them well
			So(evaluation.GetAccuracy(cf), ShouldBeGreaterThan, 0.9)
		})

		Convey("CrossValidate re-fits the pipeline per fold and returns one confusion matrix per fold", func() {
			cfs, err := CrossValidate(inst, newPipe, trees.NewID3DecisionTree(0.0), 3)
			So(err, ShouldBeNil)
			So(len(cfs), ShouldEqual, 3)
			mean, _ := evaluation.GetCrossValidatedMetric(cfs, evaluation.GetAccuracy)
			So(mean, ShouldBeGreaterThan, 0.0)
		})
	})
}
