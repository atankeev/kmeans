package kmeans

import (
	"math"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestKmeans_ValidateData(t *testing.T) {
	t.Parallel()
	t.Run("valid data", func(t *testing.T) {
		t.Parallel()
		data := [][]float64{
			{1.0, 2.0, 3.0},
			{4.0, 5.0, 6.0},
			{7.0, 8.0, 9.0},
		}
		kmeans := New(2)

		err := kmeans.validateData(data)
		require.NoError(t, err)
	})

	t.Run("empty data", func(t *testing.T) {
		t.Parallel()
		data := [][]float64{}
		kmeans := New(2)

		err := kmeans.validateData(data)
		require.ErrorIs(t, err, ErrEmptyData)
	})

	t.Run("nil data", func(t *testing.T) {
		t.Parallel()
		var data [][]float64
		kmeans := New(2)

		err := kmeans.validateData(data)
		require.ErrorIs(t, err, ErrEmptyData)
	})

	t.Run("inconsistent dimensions", func(t *testing.T) {
		t.Parallel()
		data := [][]float64{
			{1.0, 2.0, 3.0},
			{4.0, 5.0}, // Different dimension
			{7.0, 8.0, 9.0},
		}
		kmeans := New(2)

		err := kmeans.validateData(data)
		require.Error(t, err)
		require.ErrorContains(t, err, "dimension")
	})

	t.Run("more clusters than data points", func(t *testing.T) {
		t.Parallel()
		data := [][]float64{
			{1.0, 2.0},
			{3.0, 4.0},
		}

		kmeans := New(3)

		err := kmeans.validateData(data)
		require.ErrorIs(t, err, ErrInvalidK)
		require.ErrorContains(t, err, "cannot be greater than number of samples")
	})

	t.Run("equal clusters and data points", func(t *testing.T) {
		t.Parallel()
		data := [][]float64{
			{1.0, 2.0},
			{3.0, 4.0},
		}
		kmeans := New(2)

		err := kmeans.validateData(data)
		require.NoError(t, err)
	})

	t.Run("single data point", func(t *testing.T) {
		t.Parallel()
		data := [][]float64{{1.0, 2.0, 3.0}}
		kmeans := New(1)

		err := kmeans.validateData(data)
		require.NoError(t, err)
	})
}

func TestCluster_Lloyd_NonFiniteValues(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name       string
		coordinate float64
	}{
		{name: "NaN", coordinate: math.NaN()},
		{name: "positive infinity", coordinate: math.Inf(1)},
		{name: "negative infinity", coordinate: math.Inf(-1)},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			kmeans := NewWithOptions(2,
				WithInitMethod(InitKMeansPlusPlus),
				WithRandomSeed(42),
			)
			data := [][]float64{
				{1.0, 2.0},
				{3.0, 4.0},
				{tt.coordinate, 5.0},
				{6.0, 7.0},
			}

			result, err := kmeans.Cluster(data)

			require.Nil(t, result)
			require.ErrorIs(t, err, ErrNonFiniteData)
			require.EqualError(t, err, "invalid data: data contains a non-finite coordinate at point 2, dimension 0")
		})
	}
}

func TestCluster_NoFeatures(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		data [][]float64
	}{
		{name: "nil samples", data: [][]float64{nil, nil}},
		{name: "empty samples", data: [][]float64{{}, {}}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			result, err := New(1).Cluster(tt.data)

			require.Nil(t, result)
			require.ErrorIs(t, err, ErrNoFeatures)
			require.EqualError(t, err, "invalid data: data must contain at least one feature")
		})
	}
}

func TestCluster_EmptyAndNonEmptySamplesHaveInconsistentDimensions(t *testing.T) {
	t.Parallel()

	result, err := New(1).Cluster([][]float64{{}, {1}})

	require.Nil(t, result)
	require.Error(t, err)
	require.NotErrorIs(t, err, ErrNoFeatures)
	require.EqualError(t, err, "invalid data: data point at index 1 has dimension 1, expected 0")
}

func TestWithRandomSeed(t *testing.T) {
	t.Parallel()

	kmeans := NewWithOptions(2, WithRandomSeed(42))
	require.Equal(t, int64(42), kmeans.RandomSeed)

	kmeans = NewWithOptions(2, WithRandomSeed(0))
	require.Zero(t, kmeans.RandomSeed)
}

func TestInvalidIntegerFunctionalOptions(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name        string
		option      Option
		wantErr     error
		assertValue func(*testing.T, *Kmeans)
	}{
		{
			name:    "n_init zero",
			option:  WithNInit(0),
			wantErr: ErrInvalidNInit,
			assertValue: func(t *testing.T, kmeans *Kmeans) {
				require.Zero(t, kmeans.NInit)
			},
		},
		{
			name:    "n_init negative",
			option:  WithNInit(-1),
			wantErr: ErrInvalidNInit,
			assertValue: func(t *testing.T, kmeans *Kmeans) {
				require.Equal(t, -1, kmeans.NInit)
			},
		},
		{
			name:    "max_iter zero",
			option:  WithMaxIter(0),
			wantErr: ErrInvalidMaxIter,
			assertValue: func(t *testing.T, kmeans *Kmeans) {
				require.Zero(t, kmeans.MaxIter)
			},
		},
		{
			name:    "max_iter negative",
			option:  WithMaxIter(-1),
			wantErr: ErrInvalidMaxIter,
			assertValue: func(t *testing.T, kmeans *Kmeans) {
				require.Equal(t, -1, kmeans.MaxIter)
			},
		},
		{
			name:    "centroid initialization trials zero",
			option:  WithNCentroidsInitTrials(0),
			wantErr: ErrInvalidNCentroidsInitTrials,
			assertValue: func(t *testing.T, kmeans *Kmeans) {
				require.Zero(t, kmeans.NCentroidsInitTrials)
			},
		},
		{
			name:    "centroid initialization trials negative",
			option:  WithNCentroidsInitTrials(-1),
			wantErr: ErrInvalidNCentroidsInitTrials,
			assertValue: func(t *testing.T, kmeans *Kmeans) {
				require.Equal(t, -1, kmeans.NCentroidsInitTrials)
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			assertInvalidOption(t, tt.option, tt.wantErr, tt.assertValue)
		})
	}
}

func TestInvalidTolOption(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name        string
		value       float64
		assertValue func(*testing.T, float64)
	}{
		{name: "zero", value: 0, assertValue: func(t *testing.T, value float64) { require.Zero(t, value) }},
		{name: "negative", value: -1, assertValue: func(t *testing.T, value float64) { require.Equal(t, -1.0, value) }},
		{name: "NaN", value: math.NaN(), assertValue: func(t *testing.T, value float64) { require.True(t, math.IsNaN(value)) }},
		{name: "positive infinity", value: math.Inf(1), assertValue: func(t *testing.T, value float64) { require.True(t, math.IsInf(value, 1)) }},
		{name: "negative infinity", value: math.Inf(-1), assertValue: func(t *testing.T, value float64) { require.True(t, math.IsInf(value, -1)) }},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			assertInvalidOption(t, WithTol(tt.value), ErrInvalidTol, func(t *testing.T, kmeans *Kmeans) {
				tt.assertValue(t, kmeans.Tol)
			})
		})
	}
}

func assertInvalidOption(
	t *testing.T,
	option Option,
	wantErr error,
	assertValue func(*testing.T, *Kmeans),
) {
	t.Helper()

	kmeans := NewWithOptions(2, option)
	assertValue(t, kmeans)
	require.ErrorIs(t, kmeans.Validate(), wantErr)

	_, err := kmeans.Cluster([][]float64{{0}, {1}})
	require.ErrorIs(t, err, wantErr)
}

func TestFunctionalOptionLastValueWins(t *testing.T) {
	t.Parallel()

	t.Run("valid value replaces invalid value", func(t *testing.T) {
		t.Parallel()

		kmeans := NewWithOptions(2, WithMaxIter(0), WithMaxIter(1))

		require.Equal(t, 1, kmeans.MaxIter)
		require.NoError(t, kmeans.Validate())
	})

	t.Run("invalid value replaces valid value", func(t *testing.T) {
		t.Parallel()

		kmeans := NewWithOptions(2, WithMaxIter(1), WithMaxIter(0))

		require.Zero(t, kmeans.MaxIter)
		require.ErrorIs(t, kmeans.Validate(), ErrInvalidMaxIter)
	})
}

func TestCluster_ValidationErrors(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name    string
		kmeans  *Kmeans
		data    [][]float64
		wantErr bool
	}{
		{
			name:    "uninitialized config",
			kmeans:  &Kmeans{initialized: false},
			data:    [][]float64{{1.0, 2.0}, {3.0, 4.0}},
			wantErr: true,
		},
		{
			name: "invalid k (zero)",
			kmeans: NewWithOptions(0,
				WithInitMethod(InitKMeansPlusPlus),
				WithMaxIter(100),
			),
			data:    [][]float64{{1.0, 2.0}, {3.0, 4.0}},
			wantErr: true,
		},
		{
			name: "invalid k (negative)",
			kmeans: NewWithOptions(-1,
				WithInitMethod(InitKMeansPlusPlus),
				WithMaxIter(100),
			),
			data:    [][]float64{{1.0, 2.0}, {3.0, 4.0}},
			wantErr: true,
		},
		{
			name: "empty data",
			kmeans: NewWithOptions(2,
				WithInitMethod(InitKMeansPlusPlus),
				WithMaxIter(100),
			),
			data:    [][]float64{},
			wantErr: true,
		},
		{
			name: "k greater than data points",
			kmeans: NewWithOptions(5,
				WithInitMethod(InitKMeansPlusPlus),
				WithMaxIter(100),
			),
			data:    [][]float64{{1.0, 2.0}, {3.0, 4.0}},
			wantErr: true,
		},
		{
			name: "inconsistent dimensions",
			kmeans: NewWithOptions(2,
				WithInitMethod(InitKMeansPlusPlus),
				WithMaxIter(100),
			),
			data:    [][]float64{{1.0, 2.0}, {3.0}},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			_, err := tt.kmeans.Cluster(tt.data)

			if tt.wantErr {
				require.Error(t, err)
			} else {
				require.NoError(t, err)
			}
		})
	}
}

func TestCluster_InvalidKBoundaries(t *testing.T) {
	t.Parallel()

	data := [][]float64{{0}, {1}}
	tests := []struct {
		name             string
		nClusters        int
		wantErr          bool
		wantErrorMessage string
	}{
		{name: "zero", nClusters: 0, wantErr: true},
		{name: "negative", nClusters: -1, wantErr: true},
		{
			name:             "greater than sample count",
			nClusters:        3,
			wantErr:          true,
			wantErrorMessage: "invalid data: k must be between 1 and number of samples, inclusive: number of clusters (3) cannot be greater than number of samples (2)",
		},
		{name: "equal to sample count", nClusters: len(data)},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			result, err := NewWithOptions(tt.nClusters, WithRandomSeed(42)).Cluster(data)

			if !tt.wantErr {
				require.NoError(t, err)
				require.NotNil(t, result)
				return
			}

			require.Nil(t, result)
			require.ErrorIs(t, err, ErrInvalidK)
			if tt.wantErrorMessage != "" {
				require.EqualError(t, err, tt.wantErrorMessage)
			}
		})
	}
}

func TestCluster_EmptyDataPrecedesSampleCountValidation(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		data [][]float64
	}{
		{name: "nil", data: nil},
		{name: "empty", data: [][]float64{}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			result, err := New(1).Cluster(tt.data)

			require.Nil(t, result)
			require.ErrorIs(t, err, ErrEmptyData)
			require.NotErrorIs(t, err, ErrInvalidK)
		})
	}
}

func TestErrInvalidKMessage(t *testing.T) {
	t.Parallel()

	require.EqualError(t, ErrInvalidK, "k must be between 1 and number of samples, inclusive")
}

func TestValidate_NInit(t *testing.T) {
	t.Parallel()
	k := &Kmeans{
		NClusters:            2,
		NInit:                0,
		MaxIter:              100,
		Tol:                  1e-4,
		NCentroidsInitTrials: 5,
		RandomSeed:           42,
		initialized:          true,
	}

	err := k.Validate()
	require.Error(t, err)
}

func TestValidate_MaxIter(t *testing.T) {
	t.Parallel()
	k := &Kmeans{
		NClusters:            2,
		NInit:                1,
		MaxIter:              0,
		Tol:                  1e-4,
		NCentroidsInitTrials: 5,
		RandomSeed:           42,
		initialized:          true,
	}

	err := k.Validate()
	require.Error(t, err)
}

func TestValidate_Tol(t *testing.T) {
	t.Parallel()
	k := &Kmeans{
		NClusters:            2,
		NInit:                1,
		MaxIter:              100,
		Tol:                  0,
		NCentroidsInitTrials: 5,
		RandomSeed:           42,
		initialized:          true,
	}

	err := k.Validate()
	require.Error(t, err)
}

func TestValidate_NCentroidsInitTrials(t *testing.T) {
	t.Parallel()
	k := &Kmeans{
		NClusters:            2,
		NInit:                1,
		MaxIter:              100,
		Tol:                  1e-4,
		NCentroidsInitTrials: 0,
		RandomSeed:           42,
		initialized:          true,
	}

	err := k.Validate()
	require.Error(t, err)
}

func TestValidate_InitMethod(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		method  InitMethod
		wantErr error
	}{
		{
			name:   "random",
			method: InitRandom,
		},
		{
			name:   "k-means++",
			method: InitKMeansPlusPlus,
		},
		{
			name:    "empty",
			method:  "",
			wantErr: ErrInvalidInitMethod,
		},
		{
			name:    "unknown",
			method:  "kmeans++",
			wantErr: ErrInvalidInitMethod,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			kmeans := NewWithOptions(2, WithInitMethod(tt.method))
			err := kmeans.Validate()

			if tt.wantErr == nil {
				require.NoError(t, err)
				return
			}

			require.ErrorIs(t, err, tt.wantErr)
		})
	}
}

func TestCluster_InvalidInitMethod(t *testing.T) {
	t.Parallel()

	kmeans := NewWithOptions(2, WithInitMethod("kmeans++"))
	_, err := kmeans.Cluster([][]float64{{1, 2}, {3, 4}})

	require.ErrorIs(t, err, ErrInvalidInitMethod)
}
