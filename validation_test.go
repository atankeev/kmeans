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
		{name: "nil", data: nil, k: 1, want: kmeans.ErrEmptyData},
		{name: "empty", data: [][]float64{}, k: 1, want: kmeans.ErrEmptyData},
		{name: "nil features", data: [][]float64{nil, nil}, k: 1, want: kmeans.ErrNoFeatures},
		{name: "empty features", data: [][]float64{{}, {}}, k: 1, want: kmeans.ErrNoFeatures},
		{name: "ragged", data: [][]float64{{1, 2}, {3}}, k: 1, want: kmeans.ErrInvalidData},
		{name: "ragged empty first", data: [][]float64{{}, {1}}, k: 1, want: kmeans.ErrInvalidData},
		{name: "zero k", data: [][]float64{{1}, {2}}, k: 0, want: kmeans.ErrInvalidK},
		{name: "negative k", data: [][]float64{{1}, {2}}, k: -1, want: kmeans.ErrInvalidK},
		{name: "too many clusters", data: [][]float64{{1}, {2}}, k: 3, want: kmeans.ErrInvalidK},
		{name: "nan", data: [][]float64{{1, 2}, {math.NaN(), 3}}, k: 1, want: kmeans.ErrNonFiniteData},
		{
			name: "positive infinity",
			data: [][]float64{{1, 2}, {3, math.Inf(1)}},
			k:    1,
			want: kmeans.ErrNonFiniteData,
		},
		{
			name: "negative infinity",
			data: [][]float64{{1, 2}, {3, math.Inf(-1)}},
			k:    1,
			want: kmeans.ErrNonFiniteData,
		},
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
		{
			name:    "runs zero",
			valid:   kmeans.WithRuns(1),
			invalid: kmeans.WithRuns(0),
			want:    kmeans.ErrInvalidRuns,
		},
		{
			name:    "runs negative",
			valid:   kmeans.WithRuns(1),
			invalid: kmeans.WithRuns(-1),
			want:    kmeans.ErrInvalidRuns,
		},
		{
			name:    "iterations zero",
			valid:   kmeans.WithMaxIterations(2),
			invalid: kmeans.WithMaxIterations(0),
			want:    kmeans.ErrInvalidMaxIterations,
		},
		{
			name:    "iterations negative",
			valid:   kmeans.WithMaxIterations(2),
			invalid: kmeans.WithMaxIterations(-1),
			want:    kmeans.ErrInvalidMaxIterations,
		},
		{
			name:    "trials zero",
			valid:   kmeans.WithNCentroidsInitTrials(1),
			invalid: kmeans.WithNCentroidsInitTrials(0),
			want:    kmeans.ErrInvalidNCentroidsInitTrials,
		},
		{
			name:    "trials negative",
			valid:   kmeans.WithNCentroidsInitTrials(1),
			invalid: kmeans.WithNCentroidsInitTrials(-1),
			want:    kmeans.ErrInvalidNCentroidsInitTrials,
		},
		{
			name:    "tolerance zero",
			valid:   kmeans.WithTolerance(1e-4),
			invalid: kmeans.WithTolerance(0),
			want:    kmeans.ErrInvalidTolerance,
		},
		{
			name:    "tolerance negative",
			valid:   kmeans.WithTolerance(1e-4),
			invalid: kmeans.WithTolerance(-1),
			want:    kmeans.ErrInvalidTolerance,
		},
		{
			name:    "tolerance nan",
			valid:   kmeans.WithTolerance(1e-4),
			invalid: kmeans.WithTolerance(math.NaN()),
			want:    kmeans.ErrInvalidTolerance,
		},
		{
			name:    "tolerance positive infinity",
			valid:   kmeans.WithTolerance(1e-4),
			invalid: kmeans.WithTolerance(math.Inf(1)),
			want:    kmeans.ErrInvalidTolerance,
		},
		{
			name:    "tolerance negative infinity",
			valid:   kmeans.WithTolerance(1e-4),
			invalid: kmeans.WithTolerance(math.Inf(-1)),
			want:    kmeans.ErrInvalidTolerance,
		},
		{
			name:    "init empty",
			valid:   kmeans.WithInitMethod(kmeans.InitRandom),
			invalid: kmeans.WithInitMethod(""),
			want:    kmeans.ErrInvalidInitMethod,
		},
		{
			name:    "init unknown",
			valid:   kmeans.WithInitMethod(kmeans.InitKMeansPlusPlus),
			invalid: kmeans.WithInitMethod("kmeans++"),
			want:    kmeans.ErrInvalidInitMethod,
		},
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
		{"nil", nil}, {"wrapper value", wrappedOption{Option: valid}}, {"wrapper pointer", &wrappedOption{Option: valid}},
		{"nil embedded value", wrappedOption{}}, {"nil embedded pointer", &wrappedOption{}}, {"typed nil", typedNil},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			require.NotPanics(t, func() {
				result, err := kmeans.Cluster([][]float64{{0}, {2}}, 1, tt.option, valid)
				require.Nil(t, result)
				require.ErrorIs(t, err, kmeans.ErrInvalidOption)
				require.Contains(t, err.Error(), "kmeans.Cluster")
			})
		})
	}
}
