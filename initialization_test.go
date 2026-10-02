package kmeans

import (
	"math/rand"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestKmeans_InitRandomCentroids(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name       string
		nClusters  int
		data       [][]float64
		randomSeed int64
	}{
		{
			name:      "basic test with 2 clusters",
			nClusters: 2,
			data: [][]float64{
				{1.0, 2.0},
				{3.0, 4.0},
				{5.0, 6.0},
				{7.0, 8.0},
			},
			randomSeed: 42,
		},
		{
			name:      "test with 3 clusters and more data points",
			nClusters: 3,
			data: [][]float64{
				{1.0, 1.0},
				{2.0, 2.0},
				{3.0, 3.0},
				{4.0, 4.0},
				{5.0, 5.0},
				{6.0, 6.0},
			},
			randomSeed: 123,
		},
		{
			name:      "test with single cluster",
			nClusters: 1,
			data: [][]float64{
				{1.0, 2.0, 3.0},
				{4.0, 5.0, 6.0},
				{7.0, 8.0, 9.0},
			},
			randomSeed: 456,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			kmeans := NewWithOptions(tt.nClusters, WithRandomSeed(tt.randomSeed))

			centroids := kmeans.initRandomCentroids(tt.data, rand.New(rand.NewSource(tt.randomSeed)))

			require.Len(t, centroids, tt.nClusters)

			assertDistinctSourceRows(t, tt.data, centroids)
		})
	}
}

func assertDistinctSourceRows(t *testing.T, data, centroids [][]float64) {
	t.Helper()
	seen := make([][]float64, 0, len(centroids))
	for i, centroid := range centroids {
		require.Len(t, centroid, len(data[0]), "centroid %d", i)
		require.Contains(t, data, centroid, "centroid %d", i)
		require.NotContains(t, seen, centroid, "duplicate centroid at index %d", i)
		seen = append(seen, centroid)
	}
}

func TestKmeans_InitRandomCentroids_UsesSuppliedGenerator(t *testing.T) {
	t.Parallel()
	data := [][]float64{
		{1.0, 2.0},
		{3.0, 4.0},
		{5.0, 6.0},
		{7.0, 8.0},
	}

	// Seed 42 selects indexes 1 and 3. The configured seed deliberately differs
	// so ignoring the supplied generator changes this observation.
	kmeans := NewWithOptions(2, WithRandomSeed(999))
	centroids := kmeans.initRandomCentroids(data, rand.New(rand.NewSource(42)))
	require.Equal(t, [][]float64{{3, 4}, {7, 8}}, centroids)
}

func TestKmeans_InitRandomCentroids_CopiesSelectedPoint(t *testing.T) {
	t.Parallel()
	data := [][]float64{{1, 2}, {3, 4}, {5, 6}, {7, 8}}
	centroids := New(1).initRandomCentroids(data, rand.New(rand.NewSource(42)))
	require.Equal(t, [][]float64{{3, 4}}, centroids)

	centroids[0][0] = 30
	require.Equal(t, []float64{3, 4}, data[1])
	data[1][1] = 40
	require.Equal(t, []float64{30, 4}, centroids[0])
}

func TestKmeans_InitRandomCentroids_EdgeCases(t *testing.T) {
	t.Parallel()
	t.Run("single data point", func(t *testing.T) {
		t.Parallel()
		data := [][]float64{{1.0, 2.0, 3.0}}
		kmeans := NewWithOptions(1, WithRandomSeed(42))

		centroids := kmeans.initRandomCentroids(data, rand.New(rand.NewSource(42)))

		require.Len(t, centroids, 1)

		require.Equal(t, data[0], centroids[0])
	})
}

func checkedWeightedChoice(t *testing.T, weights []scaledValue, rng *rand.Rand) int {
	t.Helper()
	index := scaledWeightedRandomChoice(weights, rng)
	require.GreaterOrEqual(t, index, 0, "weighted choice returned a negative index")
	require.Less(t, index, len(weights), "weighted choice returned an out-of-range index")
	return index
}

func TestScaledWeightedRandomChoice(t *testing.T) {
	t.Parallel()
	t.Run("single weight", func(t *testing.T) {
		t.Parallel()
		weights := []scaledValue{scaledFromFloat(1.0)}
		rng := rand.New(rand.NewSource(42))

		result := checkedWeightedChoice(t, weights, rng)
		require.Zero(t, result)
	})

	t.Run("equal weights", func(t *testing.T) {
		t.Parallel()
		weights := []scaledValue{scaledFromFloat(1.0), scaledFromFloat(1.0), scaledFromFloat(1.0)}
		rng := rand.New(rand.NewSource(42))

		counts := make([]int, len(weights))
		nTrials := 1000

		for range nTrials {
			result := checkedWeightedChoice(t, weights, rng)
			counts[result]++
		}

		for i, count := range counts {
			require.Positive(t, count, "index %d was never selected in %d trials", i, nTrials)
		}
	})

	t.Run("zero weights", func(t *testing.T) {
		t.Parallel()
		weights := []scaledValue{scaledFromFloat(0.0), scaledFromFloat(0.0), scaledFromFloat(0.0)}
		rng := rand.New(rand.NewSource(42))

		checkedWeightedChoice(t, weights, rng)
	})

	t.Run("mixed zero and non-zero weights", func(t *testing.T) {
		t.Parallel()
		weights := []scaledValue{scaledFromFloat(0.0), scaledFromFloat(1.0), scaledFromFloat(0.0), scaledFromFloat(2.0)}
		rng := rand.New(rand.NewSource(42))

		counts := make([]int, len(weights))
		nTrials := 1000

		for range nTrials {
			result := checkedWeightedChoice(t, weights, rng)
			counts[result]++
		}

		require.Zero(t, counts[0])
		require.Zero(t, counts[2])
		require.Positive(t, counts[1])
		require.Positive(t, counts[3])

		ratio := float64(counts[3]) / float64(counts[1])
		require.GreaterOrEqual(t, ratio, 1.5)
		require.LessOrEqual(t, ratio, 2.5)
	})

	t.Run("very large weights", func(t *testing.T) {
		t.Parallel()
		weights := []scaledValue{scaledFromFloat(1e10), scaledFromFloat(2e10), scaledFromFloat(3e10)}
		rng := rand.New(rand.NewSource(42))

		counts := make([]int, len(weights))
		nTrials := 1000

		for range nTrials {
			result := checkedWeightedChoice(t, weights, rng)
			counts[result]++
		}

		for i, count := range counts {
			require.Positive(t, count, "index %d was never selected in %d trials", i, nTrials)
		}

		ratio1 := float64(counts[1]) / float64(counts[0])
		ratio2 := float64(counts[2]) / float64(counts[1])
		require.GreaterOrEqual(t, ratio1, 1.5)
		require.LessOrEqual(t, ratio1, 2.5)
		require.GreaterOrEqual(t, ratio2, 1.2)
		require.LessOrEqual(t, ratio2, 1.8)
	})

	t.Run("very small weights", func(t *testing.T) {
		t.Parallel()
		weights := []scaledValue{scaledFromFloat(1e-10), scaledFromFloat(2e-10), scaledFromFloat(3e-10)}
		rng := rand.New(rand.NewSource(42))

		counts := make([]int, len(weights))
		nTrials := 1000

		for range nTrials {
			result := checkedWeightedChoice(t, weights, rng)
			counts[result]++
		}

		for i, count := range counts {
			require.Positive(t, count, "index %d was never selected in %d trials", i, nTrials)
		}
	})
}

func TestScaledWeightedRandomChoice_EdgeCases(t *testing.T) {
	t.Parallel()

	t.Run("single non-zero weight", func(t *testing.T) {
		t.Parallel()
		weights := []scaledValue{scaledFromFloat(0.0), scaledFromFloat(5.0), scaledFromFloat(0.0)}
		rng := rand.New(rand.NewSource(42))

		for range 100 {
			result := checkedWeightedChoice(t, weights, rng)
			require.Equal(t, 1, result)
		}
	})

}

func TestScaledWeightedRandomChoice_Properties(t *testing.T) {
	t.Parallel()
	t.Run("deterministic with same seed", func(t *testing.T) {
		t.Parallel()
		weights := []scaledValue{scaledFromFloat(1.0), scaledFromFloat(2.0), scaledFromFloat(3.0)}
		rng1 := rand.New(rand.NewSource(42))
		rng2 := rand.New(rand.NewSource(42))

		for range 10 {
			result1 := checkedWeightedChoice(t, weights, rng1)
			result2 := checkedWeightedChoice(t, weights, rng2)
			require.Equal(t, result1, result2)
		}
	})

	t.Run("different seeds produce different results", func(t *testing.T) {
		t.Parallel()
		weights := []scaledValue{scaledFromFloat(1.0), scaledFromFloat(2.0), scaledFromFloat(3.0)}
		rng1 := rand.New(rand.NewSource(42))
		rng2 := rand.New(rand.NewSource(123))

		different := false
		for range 10 {
			result1 := checkedWeightedChoice(t, weights, rng1)
			result2 := checkedWeightedChoice(t, weights, rng2)
			if result1 != result2 {
				different = true
				break
			}
		}

		require.True(t, different)
	})

	t.Run("scaling weights preserves distribution", func(t *testing.T) {
		t.Parallel()
		weights1 := []scaledValue{scaledFromFloat(1.0), scaledFromFloat(2.0), scaledFromFloat(3.0)}
		weights2 := []scaledValue{scaledFromFloat(10.0), scaledFromFloat(20.0), scaledFromFloat(30.0)} // Scaled by 10
		rng1 := rand.New(rand.NewSource(42))
		rng2 := rand.New(rand.NewSource(42))

		for range 10 {
			result1 := checkedWeightedChoice(t, weights1, rng1)
			result2 := checkedWeightedChoice(t, weights2, rng2)
			require.Equal(t, result1, result2)
		}
	})

	t.Run("valid index range", func(t *testing.T) {
		t.Parallel()
		weights := []scaledValue{scaledFromFloat(1.0), scaledFromFloat(2.0), scaledFromFloat(3.0), scaledFromFloat(4.0), scaledFromFloat(5.0)}
		rng := rand.New(rand.NewSource(42))

		for range 100 {
			checkedWeightedChoice(t, weights, rng)
		}
	})
}

func TestKmeans_InitKMeansPlusPlusCentroidsWithDistances(t *testing.T) {
	t.Parallel()
	t.Run("no duplicate centroids", func(t *testing.T) {
		t.Parallel()
		data := [][]float64{
			{1.0, 1.0},
			{2.0, 2.0},
			{3.0, 3.0},
			{4.0, 4.0},
			{5.0, 5.0},
			{6.0, 6.0},
		}

		kmeans := NewWithOptions(3, WithRandomSeed(42))

		centroids := kmeans.initKMeansPlusPlusCentroidsWithDistances(data, make([]scaledValue, len(data)), kmeans.newRandomState())

		require.Len(t, centroids, 3)

		assertDistinctSourceRows(t, data, centroids)

	})

	t.Run("single cluster", func(t *testing.T) {
		t.Parallel()
		data := [][]float64{
			{1.0, 2.0, 3.0},
			{4.0, 5.0, 6.0},
			{7.0, 8.0, 9.0},
		}

		kmeans := NewWithOptions(1, WithRandomSeed(42))

		centroids := kmeans.initKMeansPlusPlusCentroidsWithDistances(data, make([]scaledValue, len(data)), kmeans.newRandomState())

		require.Len(t, centroids, 1)

		require.Contains(t, data, centroids[0])
	})

	t.Run("clusters equal to data points", func(t *testing.T) {
		t.Parallel()
		data := [][]float64{
			{1.0, 1.0},
			{2.0, 2.0},
			{3.0, 3.0},
		}

		kmeans := NewWithOptions(3, WithRandomSeed(42))

		centroids := kmeans.initKMeansPlusPlusCentroidsWithDistances(data, make([]scaledValue, len(data)), kmeans.newRandomState())

		require.Len(t, centroids, 3)

		assertDistinctSourceRows(t, data, centroids)

		require.ElementsMatch(t, data, centroids)
	})

	t.Run("deterministic with same seed", func(t *testing.T) {
		t.Parallel()
		data := [][]float64{
			{1.0, 1.0},
			{2.0, 2.0},
			{3.0, 3.0},
			{4.0, 4.0},
			{5.0, 5.0},
		}

		kmeans1 := NewWithOptions(3, WithRandomSeed(42))
		kmeans2 := NewWithOptions(3, WithRandomSeed(42))

		centroids1 := kmeans1.initKMeansPlusPlusCentroidsWithDistances(data, make([]scaledValue, len(data)), kmeans1.newRandomState())
		centroids2 := kmeans2.initKMeansPlusPlusCentroidsWithDistances(data, make([]scaledValue, len(data)), kmeans2.newRandomState())

		require.Equal(t, centroids1, centroids2)
	})

	t.Run("different seeds produce different results", func(t *testing.T) {
		t.Parallel()
		data := [][]float64{
			{1.0, 1.0},
			{2.0, 2.0},
			{3.0, 3.0},
			{4.0, 4.0},
			{5.0, 5.0},
		}

		kmeans1 := NewWithOptions(3, WithRandomSeed(42))
		kmeans2 := NewWithOptions(3, WithRandomSeed(123456))

		centroids1 := kmeans1.initKMeansPlusPlusCentroidsWithDistances(data, make([]scaledValue, len(data)), kmeans1.newRandomState())
		centroids2 := kmeans2.initKMeansPlusPlusCentroidsWithDistances(data, make([]scaledValue, len(data)), kmeans2.newRandomState())

		require.NotEqual(t, centroids1, centroids2)
	})

	t.Run("duplicate data points", func(t *testing.T) {
		t.Parallel()
		data := [][]float64{
			{1.0, 1.0},
			{1.0, 1.0},
			{1.0, 1.0},
			{10.0, 10.0},
			{10.0, 10.0},
		}

		kmeans := NewWithOptions(len(data), WithRandomSeed(42))

		centroids := kmeans.initKMeansPlusPlusCentroidsWithDistances(data, make([]scaledValue, len(data)), kmeans.newRandomState())

		require.Len(t, centroids, len(data))
		require.ElementsMatch(t, data, centroids)
	})

	t.Run("greedy k-means++ mode (nLocalTrials >= 2)", func(t *testing.T) {
		t.Parallel()
		data := [][]float64{
			{1.0, 1.0},
			{2.0, 2.0},
			{3.0, 3.0},
			{4.0, 4.0},
			{5.0, 5.0},
		}

		kmeans := NewWithOptions(2, WithRandomSeed(42))

		centroids := kmeans.initKMeansPlusPlusCentroidsWithDistances(data, make([]scaledValue, len(data)), kmeans.newRandomState())

		require.Len(t, centroids, 2)

		assertDistinctSourceRows(t, data, centroids)
	})

	t.Run("greedy k-means++ mode (nLocalTrials > 1)", func(t *testing.T) {
		t.Parallel()
		data := [][]float64{
			{1.0, 1.0},
			{2.0, 2.0},
			{3.0, 3.0},
			{4.0, 4.0},
			{5.0, 5.0},
			{6.0, 6.0},
			{7.0, 7.0},
			{8.0, 8.0},
		}

		kmeans := NewWithOptions(5, WithRandomSeed(42))

		centroids := kmeans.initKMeansPlusPlusCentroidsWithDistances(data, make([]scaledValue, len(data)), kmeans.newRandomState())

		require.Len(t, centroids, 5)

		assertDistinctSourceRows(t, data, centroids)
	})

	t.Run("high dimensional data", func(t *testing.T) {
		t.Parallel()
		data := [][]float64{
			{1.0, 2.0, 3.0, 4.0, 5.0},
			{2.0, 3.0, 4.0, 5.0, 6.0},
			{3.0, 4.0, 5.0, 6.0, 7.0},
			{4.0, 5.0, 6.0, 7.0, 8.0},
			{5.0, 6.0, 7.0, 8.0, 9.0},
			{6.0, 7.0, 8.0, 9.0, 10.0},
		}

		kmeans := NewWithOptions(3, WithRandomSeed(42))

		centroids := kmeans.initKMeansPlusPlusCentroidsWithDistances(data, make([]scaledValue, len(data)), kmeans.newRandomState())

		require.Len(t, centroids, 3)

		for i, centroid := range centroids {
			require.Len(t, centroid, 5, "centroid %d", i)
		}

		assertDistinctSourceRows(t, data, centroids)
	})

	t.Run("data with negative coordinates", func(t *testing.T) {
		t.Parallel()
		data := [][]float64{
			{-1.0, -2.0},
			{-3.0, -4.0},
			{1.0, 2.0},
			{3.0, 4.0},
			{0.0, 0.0},
		}

		kmeans := NewWithOptions(3, WithRandomSeed(42))

		centroids := kmeans.initKMeansPlusPlusCentroidsWithDistances(data, make([]scaledValue, len(data)), kmeans.newRandomState())

		require.Len(t, centroids, 3)

		assertDistinctSourceRows(t, data, centroids)
	})

	t.Run("data with decimal coordinates", func(t *testing.T) {
		t.Parallel()
		data := [][]float64{
			{1.5, 2.7},
			{3.2, 4.8},
			{5.1, 6.3},
			{7.9, 8.4},
			{9.6, 10.2},
		}

		kmeans := NewWithOptions(3, WithRandomSeed(42))

		centroids := kmeans.initKMeansPlusPlusCentroidsWithDistances(data, make([]scaledValue, len(data)), kmeans.newRandomState())

		require.Len(t, centroids, 3)

		assertDistinctSourceRows(t, data, centroids)
	})

	t.Run("large number of clusters", func(t *testing.T) {
		t.Parallel()
		data := [][]float64{
			{1.0, 1.0}, {2.0, 2.0}, {3.0, 3.0}, {4.0, 4.0}, {5.0, 5.0},
			{6.0, 6.0}, {7.0, 7.0}, {8.0, 8.0}, {9.0, 9.0}, {10.0, 10.0},
			{11.0, 11.0}, {12.0, 12.0}, {13.0, 13.0}, {14.0, 14.0}, {15.0, 15.0},
		}

		kmeans := NewWithOptions(8, WithRandomSeed(42))

		centroids := kmeans.initKMeansPlusPlusCentroidsWithDistances(data, make([]scaledValue, len(data)), kmeans.newRandomState())

		require.Len(t, centroids, 8)

		assertDistinctSourceRows(t, data, centroids)
	})

	t.Run("standard k-means++ mode (nLocalTrials = 1)", func(t *testing.T) {
		t.Parallel()
		data := [][]float64{
			{1.0, 1.0},
			{2.0, 2.0},
			{3.0, 3.0},
			{4.0, 4.0},
			{5.0, 5.0},
		}
		kmeans := NewWithOptions(2, WithRandomSeed(42), WithNCentroidsInitTrials(1))

		centroids := kmeans.initKMeansPlusPlusCentroidsWithDistances(data, make([]scaledValue, len(data)), kmeans.newRandomState())
		require.Len(t, centroids, 2)
		assertDistinctSourceRows(t, data, centroids)
	})
}

func TestInitCentroidsWithDistances_Random(t *testing.T) {
	t.Parallel()
	kmeans := NewWithOptions(2,
		WithInitMethod(InitRandom),
		WithRandomSeed(42),
	)

	data := [][]float64{
		{1.0, 2.0},
		{3.0, 4.0},
		{5.0, 6.0},
		{7.0, 8.0},
	}

	centroids := kmeans.initCentroidsWithDistances(data, make([]scaledValue, len(data)), kmeans.newRandomState())

	require.Len(t, centroids, 2)
	assertDistinctSourceRows(t, data, centroids)
}

func TestInitCentroidsWithDistances_KMeansPlusPlus(t *testing.T) {
	t.Parallel()
	kmeans := NewWithOptions(2,
		WithInitMethod(InitKMeansPlusPlus),
		WithRandomSeed(42),
	)

	data := [][]float64{
		{1.0, 2.0},
		{3.0, 4.0},
		{10.0, 11.0},
		{12.0, 13.0},
	}

	centroids := kmeans.initCentroidsWithDistances(data, make([]scaledValue, len(data)), kmeans.newRandomState())

	require.Len(t, centroids, 2)
	for _, centroid := range centroids {
		require.Len(t, centroid, 2)
	}
}

func TestInitCentroidsWithDistances_Default(t *testing.T) {
	t.Parallel()

	kmeans := NewWithOptions(2, WithRandomSeed(42))
	require.Equal(t, InitKMeansPlusPlus, kmeans.Init)

	data := [][]float64{
		{1.0, 2.0},
		{3.0, 4.0},
		{5.0, 6.0},
	}

	centroids := kmeans.initCentroidsWithDistances(data, make([]scaledValue, len(data)), kmeans.newRandomState())

	require.Len(t, centroids, 2)
}
