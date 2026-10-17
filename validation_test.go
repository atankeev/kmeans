package kmeans_test

import (
	"math"
	"testing"

	"github.com/atankeev/kmeans"
	"github.com/stretchr/testify/require"
)

func TestCluster_InvalidData(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name string
		data [][]float64
		k    int
		want error
	}{
		{"nil", nil, 1, kmeans.ErrEmptyData},
		{"empty", [][]float64{}, 1, kmeans.ErrEmptyData},
		{"nil features", [][]float64{nil, nil}, 1, kmeans.ErrNoFeatures},
		{"empty features", [][]float64{{}, {}}, 1, kmeans.ErrNoFeatures},
		{"ragged", [][]float64{{1, 2}, {3}}, 1, kmeans.ErrInvalidData},
		{"ragged empty first", [][]float64{{}, {1}}, 1, kmeans.ErrInvalidData},
		{"zero k", [][]float64{{1}, {2}}, 0, kmeans.ErrInvalidK},
		{"negative k", [][]float64{{1}, {2}}, -1, kmeans.ErrInvalidK},
		{"too many clusters", [][]float64{{1}, {2}}, 3, kmeans.ErrInvalidK},
		{"nan", [][]float64{{1, 2}, {math.NaN(), 3}}, 1, kmeans.ErrNonFiniteData},
		{"positive infinity", [][]float64{{1, 2}, {3, math.Inf(1)}}, 1, kmeans.ErrNonFiniteData},
		{"negative infinity", [][]float64{{1, 2}, {3, math.Inf(-1)}}, 1, kmeans.ErrNonFiniteData},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			result, err := kmeans.Cluster(tt.data, tt.k)
			require.Nil(t, result)
			require.ErrorIs(t, err, tt.want)
		})
	}
}

func TestCluster_RaggedDimensionsDiagnostic(t *testing.T) {
	t.Parallel()
	result, err := kmeans.Cluster([][]float64{{1, 2, 3}, {4, 5}}, 1)
	require.Nil(t, result)
	require.ErrorIs(t, err, kmeans.ErrInvalidData)
	require.ErrorContains(t, err, "index 1")
	require.ErrorContains(t, err, "dimension 2")
	require.ErrorContains(t, err, "expected 3")
	require.ErrorContains(t, kmeans.ErrInvalidData, "consistent dimensions")
}

func TestCluster_ValidationPrecedence(t *testing.T) {
	t.Parallel()
	result, err := kmeans.Cluster(nil, 1)
	require.Nil(t, result)
	require.ErrorIs(t, err, kmeans.ErrEmptyData)
	require.NotErrorIs(t, err, kmeans.ErrInvalidK)
	result, err = kmeans.Cluster([][]float64{{}, {1}}, 1)
	require.Nil(t, result)
	require.ErrorIs(t, err, kmeans.ErrInvalidData)
	require.NotErrorIs(t, err, kmeans.ErrNoFeatures)
}

func TestCluster_InvalidFinalSettings(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name           string
		valid, invalid kmeans.Option
		want           error
	}{
		{"runs zero", kmeans.WithRuns(1), kmeans.WithRuns(0), kmeans.ErrInvalidRuns},
		{"runs negative", kmeans.WithRuns(1), kmeans.WithRuns(-1), kmeans.ErrInvalidRuns},
		{"iterations zero", kmeans.WithMaxIterations(2), kmeans.WithMaxIterations(0), kmeans.ErrInvalidMaxIterations},
		{"iterations negative", kmeans.WithMaxIterations(2), kmeans.WithMaxIterations(-1), kmeans.ErrInvalidMaxIterations},
		{"trials zero", kmeans.WithNCentroidsInitTrials(1), kmeans.WithNCentroidsInitTrials(0), kmeans.ErrInvalidNCentroidsInitTrials},
		{"trials negative", kmeans.WithNCentroidsInitTrials(1), kmeans.WithNCentroidsInitTrials(-1), kmeans.ErrInvalidNCentroidsInitTrials},
		{"tolerance zero", kmeans.WithTolerance(1e-4), kmeans.WithTolerance(0), kmeans.ErrInvalidTolerance},
		{"tolerance negative", kmeans.WithTolerance(1e-4), kmeans.WithTolerance(-1), kmeans.ErrInvalidTolerance},
		{"tolerance nan", kmeans.WithTolerance(1e-4), kmeans.WithTolerance(math.NaN()), kmeans.ErrInvalidTolerance},
		{"tolerance positive infinity", kmeans.WithTolerance(1e-4), kmeans.WithTolerance(math.Inf(1)), kmeans.ErrInvalidTolerance},
		{"tolerance negative infinity", kmeans.WithTolerance(1e-4), kmeans.WithTolerance(math.Inf(-1)), kmeans.ErrInvalidTolerance},
		{"init empty", kmeans.WithInitMethod(kmeans.InitRandom), kmeans.WithInitMethod(""), kmeans.ErrInvalidInitMethod},
		{"init unknown", kmeans.WithInitMethod(kmeans.InitKMeansPlusPlus), kmeans.WithInitMethod("kmeans++"), kmeans.ErrInvalidInitMethod},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			data := [][]float64{{0}, {2}}
			result, err := kmeans.Cluster(data, 1, tt.invalid)
			require.Nil(t, result)
			require.ErrorIs(t, err, tt.want)
			result, err = kmeans.Cluster(data, 1, tt.invalid, tt.valid)
			require.NoError(t, err)
			require.NotNil(t, result)
			result, err = kmeans.Cluster(data, 1, tt.valid, tt.invalid)
			require.Nil(t, result)
			require.ErrorIs(t, err, tt.want)
		})
	}
}

type wrappedOption struct{ kmeans.Option }

func TestCluster_RejectsUnsupportedOptions(t *testing.T) {
	t.Parallel()
	var typedNil *wrappedOption
	valid := kmeans.WithSeed(0)
	tests := []struct {
		name   string
		option kmeans.Option
	}{
		{"nil", nil}, {"wrapper value", wrappedOption{valid}}, {"wrapper pointer", &wrappedOption{valid}},
		{"nil embedded value", wrappedOption{}}, {"nil embedded pointer", &wrappedOption{}}, {"typed nil", typedNil},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			require.NotPanics(t, func() {
				result, err := kmeans.Cluster([][]float64{{0}, {2}}, 1, tt.option, valid)
				require.Nil(t, result)
				require.ErrorIs(t, err, kmeans.ErrInvalidOption)
			})
		})
	}
}
