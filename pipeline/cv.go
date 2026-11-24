package pipeline

import (
	"math/rand"

	"github.com/sjwhitworth/golearn/base"
	"github.com/sjwhitworth/golearn/evaluation"
)

// CrossValidate runs k-fold cross-validation with the feature pipeline fitted
// inside each fold, so no information from a validation fold can leak into
// the statistics or vocabularies used to transform it.
//
// For every fold a fresh Pipeline is obtained from newPipeline, fitted on the
// fold's training rows only (FitTransform), and then applied to the held-out
// validation rows (Transform). cls is trained and evaluated on the
// transformed data, exactly like
// evaluation.GenerateCrossFoldValidationConfusionMatrices does on raw data.
//
// newPipeline must return a new, unfitted Pipeline on every call (the stages
// carry fitted state and must not be shared across folds).
func CrossValidate(data base.FixedDataGrid, newPipeline func() *Pipeline, cls base.Classifier, folds int) ([]evaluation.ConfusionMatrix, error) {
	_, rows := data.Size()

	// Assign each row to a fold (same scheme as the evaluation package).
	foldMap := make([]int, rows)
	inverseFoldMap := make(map[int][]int)
	for i := 0; i < rows; i++ {
		fold := rand.Intn(folds)
		foldMap[i] = fold
		inverseFoldMap[fold] = append(inverseFoldMap[fold], i)
	}

	ret := make([]evaluation.ConfusionMatrix, folds)
	for i := 0; i < folds; i++ {
		// Fold i is for validation.
		validationData := base.NewInstancesViewFromVisible(data, inverseFoldMap[i], data.AllAttributes())
		trainRows := make([]int, 0)
		for j := 0; j < folds; j++ {
			if i == j {
				continue
			}
			trainRows = append(trainRows, inverseFoldMap[j]...)
		}
		trainData := base.NewInstancesViewFromVisible(data, trainRows, data.AllAttributes())

		// Fresh pipeline per fold, fitted on the training rows only.
		pipe := newPipeline()
		trainTransformed, err := pipe.FitTransform(trainData)
		if err != nil {
			return nil, err
		}
		validationTransformed, err := pipe.Transform(validationData)
		if err != nil {
			return nil, err
		}

		// Train and evaluate the classifier on the transformed data.
		if err := cls.Fit(trainTransformed); err != nil {
			return nil, err
		}
		pred, err := cls.Predict(validationTransformed)
		if err != nil {
			return nil, err
		}
		cf, err := evaluation.GetConfusionMatrix(validationTransformed, pred)
		if err != nil {
			return nil, err
		}
		ret[i] = cf
	}
	return ret, nil
}
