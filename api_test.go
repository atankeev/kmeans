package kmeans_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/atankeev/kmeans"
	"github.com/stretchr/testify/require"
)

func TestCluster_DefaultCall(t *testing.T) {
	t.Parallel()
	result, err := kmeans.Cluster([][]float64{{0}, {2}}, 1)
	require.NoError(t, err)
	require.Equal(t, []int{0, 0}, result.Labels)
	require.Equal(t, [][]float64{{1}}, result.Centroids)
	require.Equal(t, float64(2), result.Inertia) //nolint:testifylint // Integer coordinates give an exactly representable inertia.
	require.True(t, result.Converged)
	require.Positive(t, result.Iterations)
}

func TestCluster_DefaultsMatchExplicitSettings(t *testing.T) {
	t.Parallel()
	require.Equal(t, 10, kmeans.DefaultRuns)
	require.Equal(t, 300, kmeans.DefaultMaxIterations)
	require.Equal(t, 1e-4, kmeans.DefaultTolerance) //nolint:testifylint // The exported default must match this exact configuration value.
	data := [][]float64{{0}, {1}, {2}, {4}, {8}, {16}}
	for _, k := range []int{1, 2, 3, 6} {
		t.Run(fmt.Sprintf("k=%d", k), func(t *testing.T) {
			t.Parallel()
			want, err := kmeans.Cluster(data, k, kmeans.WithSeed(0))
			require.NoError(t, err)
			got, err := kmeans.Cluster(data, k, kmeans.WithSeed(0), kmeans.WithRuns(kmeans.DefaultRuns), kmeans.WithMaxIterations(kmeans.DefaultMaxIterations), kmeans.WithTolerance(kmeans.DefaultTolerance), kmeans.WithInitMethod(kmeans.InitKMeansPlusPlus), kmeans.WithNCentroidsInitTrials(2+int(math.Log(float64(k)))))
			require.NoError(t, err)
			require.Equal(t, want, got)
		})
	}
}

func TestCluster_ValidOptionOverrides(t *testing.T) {
	t.Parallel()
	data := [][]float64{{0}, {1}, {2}, {4}, {8}, {16}}
	final := []kmeans.Option{kmeans.WithInitMethod(kmeans.InitRandom), kmeans.WithRuns(1), kmeans.WithMaxIterations(1), kmeans.WithTolerance(1e-12), kmeans.WithSeed(0), kmeans.WithNCentroidsInitTrials(1)}
	baseline, err := kmeans.Cluster(data, 2, final...)
	require.NoError(t, err)
	overridden := []kmeans.Option{kmeans.WithInitMethod(kmeans.InitKMeansPlusPlus), kmeans.WithRuns(10), kmeans.WithMaxIterations(300), kmeans.WithTolerance(1e-4), kmeans.WithSeed(42), kmeans.WithNCentroidsInitTrials(5)}
	overridden = append(overridden, final...)
	got, err := kmeans.Cluster(data, 2, overridden...)
	require.NoError(t, err)
	require.Equal(t, baseline, got)
	require.False(t, got.Converged)
	require.Equal(t, 1, got.Iterations)
	repeated, err := kmeans.Cluster(data, 2, final...)
	require.NoError(t, err)
	require.Equal(t, baseline, repeated)
}

func TestCluster_OneIterationCanConverge(t *testing.T) {
	t.Parallel()
	result, err := kmeans.Cluster([][]float64{{4}, {4}}, 1, kmeans.WithRuns(3), kmeans.WithMaxIterations(1), kmeans.WithSeed(0))
	require.NoError(t, err)
	require.True(t, result.Converged)
	require.Equal(t, 1, result.Iterations)
}
