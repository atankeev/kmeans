package kmeans_test

import (
	"fmt"
	"testing"

	"github.com/atankeev/kmeans"
	"github.com/stretchr/testify/require"
)

func TestCluster_SelectsLowestInertiaBeforePublicUnderflow(t *testing.T) {
	t.Parallel()

	original := []float64{0, 1, 2, 4, 8, 16}
	for _, method := range []kmeans.InitMethod{kmeans.InitKMeansPlusPlus, kmeans.InitRandom} {
		for _, scale := range []float64{1, 1e-200} {
			for _, nInit := range []int{1, 10} {
				t.Run(fmt.Sprintf("method=%v/scale=%g/nInit=%d", method, scale, nInit), func(t *testing.T) {
					t.Parallel()
					data := make([][]float64, len(original))
					for i, value := range original {
						data[i] = []float64{value * scale}
					}
					result, err := kmeans.NewWithOptions(2,
						kmeans.WithInitMethod(method), kmeans.WithRandomSeed(0), kmeans.WithNInit(nInit),
					).Cluster(data)
					require.NoError(t, err)
					if nInit == 1 {
						assertPartitionInOriginalUnits(t, result, original, scale, 4, []float64{1.75, 12}, 40.75)
					} else {
						assertPartitionInOriginalUnits(t, result, original, scale, 5, []float64{3, 16}, 40)
					}
					if scale == 1e-200 {
						require.Zero(t, result.Inertia)
					}
				})
			}
		}
	}
}

func assertPartitionInOriginalUnits(t *testing.T, result *kmeans.Result, original []float64, scale float64, split int, centers []float64, inertia float64) {
	t.Helper()
	require.NotNil(t, result)
	require.Len(t, result.Centroids, 2)
	require.Len(t, result.Labels, len(original))
	left, right := result.Labels[0], result.Labels[split]
	require.NotEqual(t, left, right)
	require.InDelta(t, centers[0], result.Centroids[left][0]/scale, 1e-12)
	require.InDelta(t, centers[1], result.Centroids[right][0]/scale, 1e-12)
	var recomputed float64
	for i, value := range original {
		wantLabel := left
		if i >= split {
			wantLabel = right
		}
		require.Equal(t, wantLabel, result.Labels[i], "sample %g", value)
		delta := value - result.Centroids[result.Labels[i]][0]/scale
		recomputed += delta * delta
	}
	require.InDelta(t, inertia, recomputed, 1e-12)
	if scale == 1 {
		require.InDelta(t, inertia, result.Inertia, 1e-12)
	}
}

func TestCluster_FourPointPartitionAtTinyScale(t *testing.T) {
	t.Parallel()
	original := []float64{0, 1, 9, 10}
	for _, method := range []kmeans.InitMethod{kmeans.InitKMeansPlusPlus, kmeans.InitRandom} {
		for _, scale := range []float64{1, 1e-200} {
			for _, nInit := range []int{1, 10} {
				t.Run(fmt.Sprintf("method=%v/scale=%g/nInit=%d", method, scale, nInit), func(t *testing.T) {
					t.Parallel()
					data := make([][]float64, len(original))
					for i, value := range original {
						data[i] = []float64{value * scale}
					}
					result, err := kmeans.NewWithOptions(2, kmeans.WithInitMethod(method),
						kmeans.WithRandomSeed(42), kmeans.WithNInit(nInit)).Cluster(data)
					require.NoError(t, err)
					assertPartitionInOriginalUnits(t, result, original, scale, 2, []float64{0.5, 9.5}, 1)
				})
			}
		}
	}
}

func TestCluster_EqualInertiaRetainsFirstRun(t *testing.T) {
	t.Parallel()
	data := [][]float64{{0}, {1}, {9}, {10}}
	run := func(seed int64, nInit int) *kmeans.Result {
		t.Helper()
		result, err := kmeans.NewWithOptions(2, kmeans.WithInitMethod(kmeans.InitRandom),
			kmeans.WithRandomSeed(seed), kmeans.WithNInit(nInit)).Cluster(data)
		require.NoError(t, err)
		require.Equal(t, 1.0, result.Inertia)
		return result
	}
	first := run(1, 1)
	require.Equal(t, [][]float64{{0.5}, {9.5}}, first.Centroids)
	require.Equal(t, []int{0, 0, 1, 1}, first.Labels)
	// Seed 43 starts with samples (3, 1), the second initialization for seed 1.
	// Its reversed cluster ordering makes a last-run tie preference observable.
	laterCandidate := run(43, 1)
	require.Equal(t, [][]float64{{9.5}, {0.5}}, laterCandidate.Centroids)
	require.Equal(t, []int{1, 1, 0, 0}, laterCandidate.Labels)
	require.Equal(t, first, run(1, 2))
}

func TestCluster_LowerInertiaNonConvergedRunWins(t *testing.T) {
	t.Parallel()
	data := [][]float64{{18}, {13}, {0}, {8}, {25}, {24}, {24}}
	run := func(nInit int) (*kmeans.Result, error) {
		return kmeans.NewWithOptions(2, kmeans.WithInitMethod(kmeans.InitRandom),
			kmeans.WithRandomSeed(0), kmeans.WithMaxIter(2), kmeans.WithNInit(nInit)).Cluster(data)
	}
	first, err := run(1)
	require.NoError(t, err)
	require.InDelta(t, 138.8, first.Inertia, 1e-12)
	require.Equal(t, []int{0, 0, 1, 1, 0, 0, 0}, first.Labels)
	require.Equal(t, [][]float64{{20.8}, {4}}, first.Centroids)
	best, err := run(2)
	require.ErrorIs(t, err, kmeans.ErrConvergenceFailed)
	require.NotNil(t, best)
	require.Equal(t, 116.75, best.Inertia)
	require.Equal(t, []int{1, 0, 0, 0, 1, 1, 1}, best.Labels)
	require.Equal(t, [][]float64{{7}, {22.75}}, best.Centroids)
	require.Less(t, best.Inertia, first.Inertia)
}

func TestCluster_TinyMovementDoesNotPrematurelyConverge(t *testing.T) {
	t.Parallel()
	original := []float64{0, 1, 2, 4, 8, 16}
	for _, scale := range []float64{1, 1e-200} {
		t.Run(fmt.Sprintf("scale=%g", scale), func(t *testing.T) {
			t.Parallel()
			data := make([][]float64, len(original))
			for i, value := range original {
				data[i] = []float64{value * scale}
			}
			run := func(maxIter int) (*kmeans.Result, error) {
				return kmeans.NewWithOptions(2, kmeans.WithInitMethod(kmeans.InitRandom),
					kmeans.WithRandomSeed(0), kmeans.WithNInit(1), kmeans.WithMaxIter(maxIter)).Cluster(data)
			}
			first, err := run(1)
			require.ErrorIs(t, err, kmeans.ErrConvergenceFailed)
			assertPartitionInOriginalUnits(t, first, original, scale, 3, []float64{0, 6.2}, 109.12)
			refined, err := run(300)
			require.NoError(t, err)
			assertPartitionInOriginalUnits(t, refined, original, scale, 4, []float64{1.75, 12}, 40.75)
			if scale == 1e-200 {
				require.Zero(t, first.Inertia)
				require.Zero(t, refined.Inertia)
			}
		})
	}
}
