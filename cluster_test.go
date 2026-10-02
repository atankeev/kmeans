package kmeans

import (
	"math"
	"slices"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestCluster_Lloyd_Basic(t *testing.T) {
	t.Parallel()
	kmeans := NewWithOptions(2,
		WithInitMethod(InitKMeansPlusPlus),
		WithRandomSeed(42),
		WithMaxIter(100),
		WithTol(1e-4),
	)

	data := [][]float64{
		{1.0, 2.0},
		{1.5, 1.8},
		{5.0, 8.0},
		{8.0, 8.0},
	}

	result, err := kmeans.Cluster(data)

	require.NoError(t, err)
	requireTwoPairPartition(t, result)
	require.Positive(t, result.Inertia)
}

func requireTwoPairPartition(t *testing.T, result *Result) {
	t.Helper()
	require.NotNil(t, result)
	require.Len(t, result.Centroids, 2)
	require.Len(t, result.Labels, 4)
	require.Equal(t, result.Labels[0], result.Labels[1])
	require.Equal(t, result.Labels[2], result.Labels[3])
	require.NotEqual(t, result.Labels[0], result.Labels[2])
	for i, label := range result.Labels {
		require.GreaterOrEqual(t, label, 0, "sample %d", i)
		require.Less(t, label, len(result.Centroids), "sample %d", i)
	}
}

func TestCluster_ResultDoesNotAliasSourceData(t *testing.T) {
	t.Parallel()
	data := [][]float64{{0, 0}, {2, 2}, {10, 10}, {12, 12}}
	result, err := NewWithOptions(2, WithRandomSeed(42)).Cluster(data)
	require.NoError(t, err)
	requireTwoPairPartition(t, result)
	for i, centroid := range result.Centroids {
		require.Len(t, centroid, 2, "centroid %d", i)
	}
	originalCenters := slices.Clone(result.Centroids[0])
	otherCenter := slices.Clone(result.Centroids[1])
	originalData := make([][]float64, len(data))
	for i, point := range data {
		originalData[i] = slices.Clone(point)
	}

	result.Centroids[0][0] = 100
	require.Equal(t, originalData, data)
	require.Equal(t, otherCenter, result.Centroids[1])

	for _, point := range data {
		point[1] = 200
	}
	require.Equal(t, []float64{100, originalCenters[1]}, result.Centroids[0])
	require.Equal(t, otherCenter, result.Centroids[1])
}

func TestCluster_CentroidRowsHaveIsolatedCapacity(t *testing.T) {
	t.Parallel()

	kmeans := NewWithOptions(2,
		WithRandomSeed(42),
		WithNInit(1),
		WithTol(1),
	)

	result, err := kmeans.Cluster([][]float64{{0}, {1}, {9}, {10}})
	require.NoError(t, err)
	require.Equal(t, [][]float64{{0.5}, {9.5}}, result.Centroids)

	for _, centroid := range result.Centroids {
		require.Equal(t, len(centroid), cap(centroid))
	}

	appended := append(result.Centroids[0], 999)
	require.Equal(t, []float64{0.5, 999}, appended)
	require.Equal(t, []float64{9.5}, result.Centroids[1])

	result.Centroids[0][0] = 1.5
	require.Equal(t, []float64{1.5}, result.Centroids[0])
	require.Equal(t, []float64{9.5}, result.Centroids[1])
}

func TestCluster_Lloyd_KMeansPlusPlusInit(t *testing.T) {
	t.Parallel()
	kmeans := NewWithOptions(2,
		WithInitMethod(InitKMeansPlusPlus),
		WithRandomSeed(123),
	)

	data := [][]float64{
		{0.0, 0.0},
		{0.0, 1.0},
		{10.0, 10.0},
		{10.0, 11.0},
	}

	result, err := kmeans.Cluster(data)

	require.NoError(t, err)
	require.Len(t, result.Centroids, 2)
	require.Len(t, result.Labels, 4)

	cluster0Count := 0
	cluster1Count := 0
	for _, label := range result.Labels {
		if label == 0 {
			cluster0Count++
		} else {
			cluster1Count++
		}
	}
	require.Equal(t, 2, cluster0Count)
	require.Equal(t, 2, cluster1Count)
}

func TestCluster_Lloyd_RandomInit(t *testing.T) {
	t.Parallel()
	kmeans := NewWithOptions(2,
		WithInitMethod(InitRandom),
		WithRandomSeed(456),
	)

	data := [][]float64{
		{1.0, 1.0},
		{2.0, 2.0},
		{9.0, 9.0},
		{10.0, 10.0},
	}

	result, err := kmeans.Cluster(data)

	require.NoError(t, err)
	require.Len(t, result.Centroids, 2)
	require.Len(t, result.Labels, 4)
	require.Positive(t, result.Inertia)
}

func TestCluster_Lloyd_Convergence(t *testing.T) {
	t.Parallel()
	kmeans := NewWithOptions(2,
		WithInitMethod(InitKMeansPlusPlus),
		WithRandomSeed(42),
		WithMaxIter(300),
		WithTol(1e-6),
	)

	data := [][]float64{
		{1.0, 2.0},
		{1.1, 2.1},
		{1.2, 2.2},
		{10.0, 10.0},
		{10.1, 10.1},
		{10.2, 10.2},
	}

	result, err := kmeans.Cluster(data)

	require.NoError(t, err)
	require.Positive(t, result.Inertia)

	require.Len(t, result.Centroids, 2)
	centroid0X := result.Centroids[0][0]
	centroid1X := result.Centroids[1][0]

	require.True(t, math.Abs(centroid0X-centroid1X) > 4, "centroids should be in different regions")
}

func TestCluster_Lloyd_IteratesUntilConvergence(t *testing.T) {
	t.Parallel()

	data := make([][]float64, 500)
	for i := range data {
		if i < 250 {
			data[i] = []float64{float64(i % 10), float64(i % 10)}
		} else {
			data[i] = []float64{float64(i%10) + 50, float64(i%10) + 50}
		}
	}

	run := func(maxIter int) (*Result, error) {
		kmeans := NewWithOptions(5,
			WithInitMethod(InitRandom),
			WithRandomSeed(7),
			WithNInit(1),
			WithMaxIter(maxIter),
			WithTol(1e-4),
		)

		return kmeans.Cluster(data)
	}

	oneIter, err := run(1)
	require.ErrorIs(t, err, ErrConvergenceFailed)
	require.NotNil(t, oneIter)

	manyIter, err := run(300)
	require.NoError(t, err)

	require.Less(t, manyIter.Inertia, oneIter.Inertia,
		"Lloyd should keep refining centroids across iterations until convergence")
}

func TestCluster_Lloyd_MaxIterLimit(t *testing.T) {
	t.Parallel()

	kmeans := NewWithOptions(1,
		WithInitMethod(InitKMeansPlusPlus),
		WithRandomSeed(42),
		WithNInit(3),
		WithMaxIter(1),
		WithTol(1e-12),
	)

	data := [][]float64{
		{0},
		{10},
	}

	result, err := kmeans.Cluster(data)

	require.ErrorIs(t, err, ErrConvergenceFailed)
	require.NotNil(t, result)
	require.Equal(t, [][]float64{{5}}, result.Centroids)
	require.Equal(t, []int{0, 0}, result.Labels)
	require.Equal(t, 50.0, result.Inertia)
}

func TestCluster_Lloyd_ConvergesOnLastIteration(t *testing.T) {
	t.Parallel()

	kmeans := NewWithOptions(1,
		WithInitMethod(InitKMeansPlusPlus),
		WithRandomSeed(42),
		WithNInit(1),
		WithMaxIter(2),
		WithTol(1e-12),
	)

	result, err := kmeans.Cluster([][]float64{{0}, {10}})

	require.NoError(t, err)
	require.Equal(t, [][]float64{{5}}, result.Centroids)
	require.Equal(t, []int{0, 0}, result.Labels)
	require.Equal(t, 50.0, result.Inertia)
}

func TestCluster_Lloyd_MultipleNInit(t *testing.T) {
	t.Parallel()
	kmeans := NewWithOptions(2,
		WithInitMethod(InitRandom),
		WithRandomSeed(42),
		WithNInit(5),
		WithMaxIter(50),
	)

	data := [][]float64{
		{1.0, 2.0},
		{1.5, 1.8},
		{5.0, 8.0},
		{8.0, 8.0},
	}

	result, err := kmeans.Cluster(data)

	require.NoError(t, err)
	require.Len(t, result.Centroids, 2)
	require.Positive(t, result.Inertia)
}

func TestCluster_Lloyd_DifferentTol(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name string
		tol  float64
	}{
		{"Very precise", 1e-10},
		{"Default precision", 1e-4},
		{"Low precision", 1e-2},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			kmeans := NewWithOptions(2,
				WithInitMethod(InitKMeansPlusPlus),
				WithRandomSeed(42),
				WithTol(tt.tol),
				WithMaxIter(100),
			)

			data := [][]float64{
				{1.0, 2.0},
				{1.5, 1.8},
				{5.0, 8.0},
				{8.0, 8.0},
			}

			result, err := kmeans.Cluster(data)

			require.NoError(t, err)
			requireTwoPairPartition(t, result)
			require.Positive(t, result.Inertia)
		})
	}
}

func TestCluster_Lloyd_SingleDimension(t *testing.T) {
	t.Parallel()
	kmeans := NewWithOptions(2,
		WithInitMethod(InitKMeansPlusPlus),
		WithRandomSeed(42),
	)

	data := [][]float64{
		{1.0},
		{2.0},
		{3.0},
		{10.0},
		{11.0},
		{12.0},
	}

	result, err := kmeans.Cluster(data)

	require.NoError(t, err)
	require.Len(t, result.Centroids, 2)
	require.Len(t, result.Labels, 6)
	require.Positive(t, result.Inertia)
	require.Len(t, result.Centroids[0], 1)
	require.Len(t, result.Centroids[1], 1)
}

func TestCluster_Lloyd_AllSamePoints(t *testing.T) {
	t.Parallel()
	kmeans := NewWithOptions(2,
		WithInitMethod(InitKMeansPlusPlus),
		WithRandomSeed(42),
	)

	data := [][]float64{
		{1.0, 1.0},
		{1.0, 1.0},
		{1.0, 1.0},
		{1.0, 1.0},
	}

	result, err := kmeans.Cluster(data)

	require.NoError(t, err)
	require.Len(t, result.Centroids, 2)
	require.Zero(t, result.Inertia)
}

func TestCluster_Lloyd_LinearData(t *testing.T) {
	t.Parallel()
	kmeans := NewWithOptions(3,
		WithInitMethod(InitKMeansPlusPlus),
		WithRandomSeed(42),
	)

	data := make([][]float64, 30)
	for i := 0; i < 30; i++ {
		data[i] = []float64{float64(i), float64(i)}
	}

	result, err := kmeans.Cluster(data)

	require.NoError(t, err)
	require.NotNil(t, result)
	require.Len(t, result.Centroids, 3)
	require.Len(t, result.Labels, 30)
	counts := make([]int, 3)
	sums := make([]float64, 3)
	seen := make(map[int]bool, 3)
	previousLabel := -1
	var inertia float64
	for i, label := range result.Labels {
		require.GreaterOrEqual(t, label, 0, "sample %d", i)
		require.Less(t, label, 3, "sample %d", i)
		require.Len(t, result.Centroids[label], 2, "centroid %d", label)
		if label != previousLabel {
			require.False(t, seen[label], "cluster %d is not contiguous", label)
			seen[label] = true
			previousLabel = label
		}
		counts[label]++
		sums[label] += data[i][0]
		for dimension, value := range data[i] {
			delta := value - result.Centroids[label][dimension]
			inertia += delta * delta
		}
	}
	for label, count := range counts {
		require.Positive(t, count, "cluster %d", label)
		mean := sums[label] / float64(count)
		require.InDelta(t, mean, result.Centroids[label][0], 1e-12, "centroid %d x", label)
		require.InDelta(t, mean, result.Centroids[label][1], 1e-12, "centroid %d y", label)
	}
	require.InDelta(t, inertia, result.Inertia, 1e-10)
}
