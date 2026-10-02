package kmeans

import (
	"fmt"
	"math"
	"math/rand"
	"slices"
	"sync"
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

func TestScaledSquaredDistance(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name     string
		p1       []float64
		p2       []float64
		expected float64
	}{
		{
			name:     "zero distance",
			p1:       []float64{1.0, 2.0, 3.0},
			p2:       []float64{1.0, 2.0, 3.0},
			expected: 0.0,
		},
		{
			name:     "simple 2D distance",
			p1:       []float64{0.0, 0.0},
			p2:       []float64{3.0, 4.0},
			expected: 25.0, // 3² + 4² = 9 + 16 = 25
		},
		{
			name:     "simple 3D distance",
			p1:       []float64{1.0, 2.0, 3.0},
			p2:       []float64{4.0, 6.0, 9.0},
			expected: 61.0, // 3² + 4² + 6² = 9 + 16 + 36 = 61
		},
		{
			name:     "negative coordinates",
			p1:       []float64{-1.0, -2.0},
			p2:       []float64{2.0, 3.0},
			expected: 34.0, // 3² + 5² = 9 + 25 = 34
		},
		{
			name:     "decimal coordinates",
			p1:       []float64{0.5, 1.5},
			p2:       []float64{2.5, 4.5},
			expected: 13.0, // 2² + 3² = 4 + 9 = 13
		},
		{
			name:     "single dimension",
			p1:       []float64{5.0},
			p2:       []float64{8.0},
			expected: 9.0, // 3² = 9
		},
		{
			name:     "high dimensional",
			p1:       []float64{1.0, 2.0, 3.0, 4.0, 5.0},
			p2:       []float64{2.0, 4.0, 6.0, 8.0, 10.0},
			expected: 55.0, // 1² + 2² + 3² + 4² + 5² = 1 + 4 + 9 + 16 + 25 = 55
		},
		{
			name:     "zero coordinates",
			p1:       []float64{0.0, 0.0, 0.0},
			p2:       []float64{0.0, 0.0, 0.0},
			expected: 0.0,
		},
		{
			name:     "mixed positive and negative",
			p1:       []float64{-3.0, 2.0, -1.0},
			p2:       []float64{1.0, -4.0, 3.0},
			expected: 68.0, // 4² + 6² + 4² = 16 + 36 + 16 = 68
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			result := requireFiniteScaled(t, scaledSquaredDistance(tt.p1, tt.p2))

			require.InDelta(t, tt.expected, result, 1e-10)
		})
	}
}

func TestScaledSquaredDistance_EdgeCases(t *testing.T) {
	t.Parallel()
	t.Run("empty slices", func(t *testing.T) {
		t.Parallel()
		result := requireFiniteScaled(t, scaledSquaredDistance([]float64{}, []float64{}))
		require.Zero(t, result)
	})

	t.Run("nil slices", func(t *testing.T) {
		t.Parallel()
		result := requireFiniteScaled(t, scaledSquaredDistance(nil, nil))
		require.Zero(t, result)
	})

	t.Run("very large numbers", func(t *testing.T) {
		t.Parallel()
		p1 := []float64{1e10, 2e10}
		p2 := []float64{3e10, 4e10}
		result := requireFiniteScaled(t, scaledSquaredDistance(p1, p2))
		expected := 8e20 // (2e10)² + (2e10)² = 4e20 + 4e20 = 8e20
		require.Equal(t, expected, result)
	})

	t.Run("very small numbers", func(t *testing.T) {
		t.Parallel()
		p1 := []float64{1e-10, 2e-10}
		p2 := []float64{3e-10, 4e-10}
		result := requireFiniteScaled(t, scaledSquaredDistance(p1, p2))
		expected := 8e-20 // (2e-10)² + (2e-10)² = 4e-20 + 4e-20 = 8e-20
		// The result is near zero, so compare with an absolute tolerance.
		require.InDelta(t, expected, result, 1e-30)
	})
}

func TestScaledSquaredDistance_Properties(t *testing.T) {
	t.Parallel()
	t.Run("commutativity", func(t *testing.T) {
		t.Parallel()
		p1 := []float64{1.0, 2.0, 3.0}
		p2 := []float64{4.0, 5.0, 6.0}

		d1 := requireFiniteScaled(t, scaledSquaredDistance(p1, p2))
		d2 := requireFiniteScaled(t, scaledSquaredDistance(p2, p1))

		require.InDelta(t, d1, d2, 1e-10)
	})

	t.Run("scaling property", func(t *testing.T) {
		t.Parallel()
		p1 := []float64{1.0, 2.0}
		p2 := []float64{3.0, 4.0}
		scale := 2.0

		d1 := requireFiniteScaled(t, scaledSquaredDistance(p1, p2))

		scaledP1 := []float64{p1[0] * scale, p1[1] * scale}
		scaledP2 := []float64{p2[0] * scale, p2[1] * scale}
		d2 := requireFiniteScaled(t, scaledSquaredDistance(scaledP1, scaledP2))

		expected := d1 * scale * scale
		require.InDelta(t, expected, d2, 1e-10)
	})
}

func TestUpdateScaledMinDistances(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name     string
		data     [][]float64
		centers  [][]float64
		expected []float64
	}{
		{
			name: "single center, single point",
			data: [][]float64{
				{1.0, 2.0},
			},
			centers: [][]float64{
				{1.0, 2.0},
			},
			expected: []float64{0.0}, // Point is exactly at center
		},
		{
			name: "single center, multiple points",
			data: [][]float64{
				{0.0, 0.0},
				{3.0, 4.0},
				{6.0, 8.0},
			},
			centers: [][]float64{
				{0.0, 0.0},
			},
			expected: []float64{0.0, 25.0, 100.0}, // 0, 3²+4²=25, 6²+8²=100
		},
		{
			name: "two centers, points assigned to nearest",
			data: [][]float64{
				{1.0, 1.0}, // Closer to center1 (distance = 2)
				{9.0, 9.0}, // Closer to center2 (distance = 2)
				{2.0, 2.0}, // Closer to center1 (distance = 8)
				{8.0, 8.0}, // Closer to center2 (distance = 8)
			},
			centers: [][]float64{
				{0.0, 0.0},   // center1
				{10.0, 10.0}, // center2
			},
			expected: []float64{2.0, 2.0, 8.0, 8.0},
		},
		{
			name: "three centers, optimal assignment",
			data: [][]float64{
				{1.0, 1.0}, // Closest to center1 (distance = 2)
				{5.0, 5.0}, // Closest to center2 (distance = 0)
				{9.0, 9.0}, // Closest to center3 (distance = 2)
			},
			centers: [][]float64{
				{0.0, 0.0},   // center1
				{5.0, 5.0},   // center2
				{10.0, 10.0}, // center3
			},
			expected: []float64{2.0, 0.0, 2.0},
		},
		{
			name:     "empty data",
			data:     [][]float64{},
			centers:  [][]float64{{1.0, 2.0}, {3.0, 4.0}},
			expected: []float64{},
		},
		{
			name: "3D points",
			data: [][]float64{
				{1.0, 2.0, 3.0}, // Distance to center1 = 14, to center2 = 14
				{4.0, 5.0, 6.0}, // Distance to center1 = 77, to center2 = 2
			},
			centers: [][]float64{
				{0.0, 0.0, 0.0}, // center1
				{5.0, 5.0, 5.0}, // center2
			},
			expected: []float64{14.0, 2.0}, // min(14,14)=14, min(77,2)=2
		},
		{
			name: "points equidistant from centers",
			data: [][]float64{
				{5.0, 0.0}, // Equidistant from both centers (distance = 25)
			},
			centers: [][]float64{
				{0.0, 0.0},  // distance = 25
				{10.0, 0.0}, // distance = 25
			},
			expected: []float64{25.0}, // Should pick the minimum (both are equal)
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			distances := make([]scaledValue, len(tt.data))
			for i, center := range tt.centers {
				updateScaledMinDistances(distances, tt.data, center, i == 0)
			}
			var expectedInertia float64
			for i, expected := range tt.expected {
				require.InDelta(t, expected, requireFiniteScaled(t, distances[i]), 1e-10)
				expectedInertia += expected
			}
			labels := assignPointsToClusters(tt.data, tt.centers, make([]int, len(tt.data)))
			inertia := calculateScaledInertiaByLabels(tt.data, tt.centers, labels)
			require.InDelta(t, expectedInertia, requireFiniteScaled(t, inertia), 1e-10)
		})
	}
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

func TestCalculateScaledTolerance(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name              string
		data              [][]float64
		relativeTolerance float64
		want              float64
	}{
		{
			name:              "single feature",
			data:              [][]float64{{0}, {2}},
			relativeTolerance: 0.1,
			want:              0.1,
		},
		{
			name:              "mean variance across features",
			data:              [][]float64{{0, 0}, {2, 4}},
			relativeTolerance: 0.2,
			want:              0.5,
		},
		{
			name:              "zero variance",
			data:              [][]float64{{3, -2}, {3, -2}},
			relativeTolerance: 0.1,
			want:              0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			require.InDelta(t, tt.want, requireFiniteScaled(t, calculateScaledTolerance(tt.data, tt.relativeTolerance)), 1e-12)
		})
	}
}

func TestCalculateScaledTolerance_Invariance(t *testing.T) {
	t.Parallel()

	data := [][]float64{{-2, 1}, {0, 4}, {5, 9}}
	translated := [][]float64{{98, -49}, {100, -46}, {105, -41}}
	scaled := [][]float64{{-20, 10}, {0, 40}, {50, 90}}

	tolerance := requireFiniteScaled(t, calculateScaledTolerance(data, 1e-4))
	require.InDelta(t, tolerance, requireFiniteScaled(t, calculateScaledTolerance(translated, 1e-4)), 1e-15)
	require.InDelta(t, tolerance*100, requireFiniteScaled(t, calculateScaledTolerance(scaled, 1e-4)), 1e-15)
}

func TestConvergenceScaleInvariance(t *testing.T) {
	t.Parallel()

	data := [][]float64{{0, 0}, {2, 4}}
	oldCenters := [][]float64{{0, 0}}
	newCenters := [][]float64{{0.1, 0.2}}
	scaledData := [][]float64{{0, 0}, {20, 40}}
	scaledOldCenters := [][]float64{{0, 0}}
	scaledNewCenters := [][]float64{{1, 2}}

	converged := checkScaledConvergence(oldCenters, newCenters, calculateScaledTolerance(data, 0.1))
	scaledConverged := checkScaledConvergence(
		scaledOldCenters,
		scaledNewCenters,
		calculateScaledTolerance(scaledData, 0.1),
	)

	require.True(t, converged)
	require.Equal(t, converged, scaledConverged)
}

func TestCheckScaledConvergence(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name       string
		oldCenters [][]float64
		newCenters [][]float64
		tolerance  float64
		want       bool
	}{
		{
			name:       "identical zero-valued centers with zero tolerance",
			oldCenters: [][]float64{{0, 0}, {0, 0}},
			newCenters: [][]float64{{0, 0}, {0, 0}},
			tolerance:  0,
			want:       true,
		},
		{
			name:       "multidimensional shift below tolerance",
			oldCenters: [][]float64{{0, 0}},
			newCenters: [][]float64{{0.3, 0.39}},
			tolerance:  0.25,
			want:       true,
		},
		{
			name:       "multidimensional shift exactly at tolerance",
			oldCenters: [][]float64{{0, 0}},
			newCenters: [][]float64{{0.3, 0.4}},
			tolerance:  0.25,
			want:       true,
		},
		{
			name:       "multidimensional shift above tolerance",
			oldCenters: [][]float64{{0, 0}},
			newCenters: [][]float64{{0.3, 0.41}},
			tolerance:  0.25,
			want:       false,
		},
		{
			name:       "Different number of centers",
			oldCenters: [][]float64{{1, 2}, {3, 4}},
			newCenters: [][]float64{{1, 2}},
			tolerance:  0.1,
			want:       false,
		},
		{
			name:       "different center dimensions",
			oldCenters: [][]float64{{1, 2}},
			newCenters: [][]float64{{1}},
			tolerance:  0.1,
			want:       false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			require.Equal(t, tt.want, checkScaledConvergence(tt.oldCenters, tt.newCenters, scaledFromFloat(tt.tolerance)))
		})
	}
}

func TestAssignPointsToClusters(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name      string
		data      [][]float64
		centers   [][]float64
		want      []int
		wantPanic bool
	}{
		{
			name:    "Simple case: 3 points, 2 centers",
			data:    [][]float64{{0, 0}, {1, 1}, {10, 10}},
			centers: [][]float64{{0, 0}, {10, 10}},
			want:    []int{0, 0, 1},
		},
		{
			name:    "Identical points, different centers",
			data:    [][]float64{{1, 1}, {1, 1}},
			centers: [][]float64{{0, 0}, {2, 2}},
			want:    []int{0, 0},
		},
		{
			name:    "Identical centers",
			data:    [][]float64{{1, 1}, {2, 2}},
			centers: [][]float64{{0, 0}, {0, 0}},
			want:    []int{0, 0},
		},
		{
			name:    "Empty data",
			data:    [][]float64{},
			centers: [][]float64{{0, 0}},
			want:    []int{},
		},
		{
			name:    "Points coincide with centers",
			data:    [][]float64{{1, 1}, {2, 2}},
			centers: [][]float64{{1, 1}, {2, 2}},
			want:    []int{0, 1},
		},
		{
			name:    "Equidistant to two centers, should pick min index",
			data:    [][]float64{{1, 0}},
			centers: [][]float64{{0, 0}, {2, 0}},
			want:    []int{0},
		},
		{
			name:    "One point, one center",
			data:    [][]float64{{1, 1}},
			centers: [][]float64{{0, 0}},
			want:    []int{0},
		},
		{
			name:    "One point, several centers",
			data:    [][]float64{{5, 5}},
			centers: [][]float64{{0, 0}, {10, 10}, {4, 4}},
			want:    []int{2},
		},
		{
			name:    "Several points, one center",
			data:    [][]float64{{1, 1}, {2, 2}, {3, 3}},
			centers: [][]float64{{0, 0}},
			want:    []int{0, 0, 0},
		},
		{
			name:      "Different dimensions (should panic)",
			data:      [][]float64{{1, 2}},
			centers:   [][]float64{{1}},
			wantPanic: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			l := make([]int, len(tt.data))
			if tt.wantPanic {
				require.Panics(t, func() { assignPointsToClusters(tt.data, tt.centers, l) })
				return
			}
			require.Equal(t, tt.want, assignPointsToClusters(tt.data, tt.centers, l))
		})
	}
}

func TestCalculateScaledInertiaByLabels(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name      string
		data      [][]float64
		centers   [][]float64
		labels    []int
		want      float64
		wantPanic bool
	}{
		{
			name:    "Simple case: 3 points, 2 centers, correct labels",
			data:    [][]float64{{0, 0}, {1, 1}, {10, 10}},
			centers: [][]float64{{0, 0}, {10, 10}},
			labels:  []int{0, 0, 1},
			want:    0 + 2 + 0, // (0,0)-(0,0):0; (1,1)-(0,0):2; (10,10)-(10,10):0
		},
		{
			name:    "All points to one center",
			data:    [][]float64{{1, 1}, {2, 2}},
			centers: [][]float64{{0, 0}},
			labels:  []int{0, 0},
			want:    2 + 8, // (1,1)-(0,0):2; (2,2)-(0,0):8
		},
		{
			name:    "Each point to its own center",
			data:    [][]float64{{1, 1}, {2, 2}},
			centers: [][]float64{{1, 1}, {2, 2}},
			labels:  []int{0, 1},
			want:    0 + 0,
		},
		{
			name:    "Empty data, centers, labels",
			data:    [][]float64{},
			centers: [][]float64{},
			labels:  []int{},
			want:    0,
		},
		{
			name:    "One point, one center, label 0",
			data:    [][]float64{{1, 2}},
			centers: [][]float64{{0, 0}},
			labels:  []int{0},
			want:    5, // (1,2)-(0,0): 1^2+2^2=5
		},
		{
			name:    "One point, several centers, label points to correct center",
			data:    [][]float64{{5, 5}},
			centers: [][]float64{{0, 0}, {10, 10}, {4, 4}},
			labels:  []int{2},
			want:    2, // (5,5)-(4,4): 1^2+1^2=2
		},
		{
			name:      "Label out of range (should panic)",
			data:      [][]float64{{1, 1}},
			centers:   [][]float64{{0, 0}},
			labels:    []int{1},
			wantPanic: true,
		},
		{
			name:      "Different dimensions (should panic)",
			data:      [][]float64{{1, 2}},
			centers:   [][]float64{{1}},
			labels:    []int{0},
			wantPanic: true,
		},
		{
			name:      "Labels shorter than data (should panic)",
			data:      [][]float64{{1, 1}, {2, 2}},
			centers:   [][]float64{{0, 0}},
			labels:    []int{0},
			wantPanic: true,
		},
		{
			name:    "All points coincide with centers",
			data:    [][]float64{{1, 1}, {2, 2}},
			centers: [][]float64{{1, 1}, {2, 2}},
			labels:  []int{0, 1},
			want:    0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			if tt.wantPanic {
				require.Panics(t, func() { calculateScaledInertiaByLabels(tt.data, tt.centers, tt.labels) })
				return
			}
			got := requireFiniteScaled(t, calculateScaledInertiaByLabels(tt.data, tt.centers, tt.labels))
			require.InDelta(t, tt.want, got, 1e-9)
		})
	}
}

func TestUpdateCentersLloydInto(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name       string
		nClusters  int
		data       [][]float64
		labels     []int
		centers    [][]float64
		want       [][]float64
		wantLabels []int
		wantPanic  bool
	}{
		{
			name:       "Simple case: 2 clusters, 2D",
			nClusters:  2,
			data:       [][]float64{{1, 2}, {3, 4}, {5, 6}},
			labels:     []int{0, 1, 0},
			centers:    [][]float64{{1, 2}, {3, 4}},
			want:       [][]float64{{(1 + 5) / 2.0, (2 + 6) / 2.0}, {3, 4}},
			wantLabels: []int{0, 1, 0},
		},
		{
			name:       "Empty cluster takes farthest assigned point",
			nClusters:  2,
			data:       [][]float64{{10, 20}, {30, 40}},
			labels:     []int{1, 1},
			centers:    [][]float64{{10, 20}, {30, 40}},
			want:       [][]float64{{10, 20}, {30, 40}},
			wantLabels: []int{0, 1},
		},
		{
			name:       "Each point its own cluster",
			nClusters:  2,
			data:       [][]float64{{1, 2}, {3, 4}},
			labels:     []int{0, 1},
			centers:    [][]float64{{1, 2}, {3, 4}},
			want:       [][]float64{{1, 2}, {3, 4}},
			wantLabels: []int{0, 1},
		},
		{
			name:       "Multiple empty clusters use distinct candidates",
			nClusters:  3,
			data:       [][]float64{{10, 20}, {30, 20}, {20, 20}},
			labels:     []int{0, 0, 0},
			centers:    [][]float64{{20, 20}, {100, 100}, {200, 200}},
			want:       [][]float64{{20, 20}, {10, 20}, {30, 20}},
			wantLabels: []int{1, 2, 0},
		},
		{
			name:      "Empty data (should panic)",
			nClusters: 2,
			data:      [][]float64{},
			labels:    []int{},
			centers:   [][]float64{{0, 0}, {1, 1}},
			wantPanic: true,
		},
		{
			name:      "Labels length mismatch (should panic)",
			nClusters: 2,
			data:      [][]float64{{1, 2}},
			labels:    []int{},
			centers:   [][]float64{{1, 2}, {3, 4}},
			wantPanic: true,
		},
		{
			name:      "Different dimensions (should panic)",
			nClusters: 2,
			data:      [][]float64{{1, 2}, {3}},
			labels:    []int{0, 1},
			centers:   [][]float64{{1, 2}, {3, 4}},
			wantPanic: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			k := &Kmeans{NClusters: tt.nClusters}
			got := newCenterBuffer(k.NClusters, len(tt.centers[0]))
			run := func() {
				k.updateCentersLloydInto(got, make([]int, k.NClusters), make([]scaledValue, len(tt.data)), tt.data, tt.labels, tt.centers)
			}
			if tt.wantPanic {
				require.Panics(t, run)
				return
			}
			require.NotPanics(t, run)
			require.Equal(t, tt.want, got)
			require.Equal(t, tt.wantLabels, tt.labels)
		})
	}
}

func TestCluster_Lloyd_RecoversEmptyClusters(t *testing.T) {
	t.Parallel()

	data := [][]float64{
		{100, 200},
		{100, 200},
		{100, 200},
		{100, 200},
	}
	cluster := func() *Result {
		kmeans := NewWithOptions(3,
			WithInitMethod(InitKMeansPlusPlus),
			WithRandomSeed(42),
			WithNInit(1),
		)
		result, err := kmeans.Cluster(data)
		require.NoError(t, err)

		return result
	}

	first := cluster()
	second := cluster()
	require.NotNil(t, first)
	require.NotNil(t, second)
	require.Len(t, first.Centroids, 3)
	require.Len(t, first.Labels, len(data))
	for i, centroid := range first.Centroids {
		require.Len(t, centroid, 2, "centroid %d", i)
	}
	require.Equal(t, first, second)

	counts := make([]int, 3)
	for pointIndex, label := range first.Labels {
		require.GreaterOrEqual(t, label, 0, "sample %d", pointIndex)
		require.Less(t, label, len(first.Centroids), "sample %d", pointIndex)
		counts[label]++
		assignedDistance := requireFiniteScaled(t, scaledSquaredDistance(data[pointIndex], first.Centroids[label]))
		for _, centroid := range first.Centroids {
			require.LessOrEqual(t, assignedDistance, requireFiniteScaled(t, scaledSquaredDistance(data[pointIndex], centroid)))
		}
	}
	for _, count := range counts {
		require.Positive(t, count)
	}
	require.Equal(t, requireFiniteScaled(t, calculateScaledInertiaByLabels(data, first.Centroids, first.Labels)), first.Inertia)
	for _, centroid := range first.Centroids {
		require.Equal(t, []float64{100, 200}, centroid)
		require.NotEqual(t, []float64{0, 0}, centroid)
	}
}

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

func TestNewCenterBuffer_LimitsEveryRowCapacity(t *testing.T) {
	t.Parallel()

	const (
		nClusters = 3
		dim       = 2
	)
	centers := newCenterBuffer(nClusters, dim)

	require.Len(t, centers, nClusters)
	for _, center := range centers {
		require.Len(t, center, dim)
		require.Equal(t, len(center), cap(center))
	}
}

func TestCluster_ConcurrentCallsAreReproducible(t *testing.T) {
	t.Parallel()

	data := [][]float64{
		{0, 0}, {0, 1}, {1, 0},
		{8, 8}, {8, 9}, {9, 8},
		{16, 0}, {16, 1}, {17, 0},
	}
	kmeans := NewWithOptions(3, WithRandomSeed(42), WithNInit(1))
	requireConcurrentCallsAreReproducible(t, kmeans, data)
}

func TestCluster_ScaledArithmeticConcurrentCallsAreReproducible(t *testing.T) {
	t.Parallel()

	data := [][]float64{{-1e308}, {-1e308}, {1e308}, {1e308}}
	kmeans := NewWithOptions(2,
		WithRandomSeed(3),
		WithNInit(1),
		WithMaxIter(1),
	)
	requireConcurrentCallsAreReproducible(t, kmeans, data)
}

func requireConcurrentCallsAreReproducible(t *testing.T, kmeans *Kmeans, data [][]float64) {
	t.Helper()

	want, err := kmeans.Cluster(data)
	require.NoError(t, err)

	const callCount = 16
	type outcome struct {
		result *Result
		err    error
	}

	start := make(chan struct{})
	outcomes := make(chan outcome, callCount)
	var wg sync.WaitGroup
	wg.Add(callCount)
	for range callCount {
		go func() {
			defer wg.Done()
			<-start

			result, clusterErr := kmeans.Cluster(data)
			outcomes <- outcome{result: result, err: clusterErr}
		}()
	}

	close(start)
	wg.Wait()
	close(outcomes)

	for outcome := range outcomes {
		require.NoError(t, outcome.err)
		require.Equal(t, want, outcome.result)
	}
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

func TestCluster_Lloyd_LargeNumbers(t *testing.T) {
	t.Parallel()

	kmeans := NewWithOptions(2,
		WithInitMethod(InitKMeansPlusPlus),
		WithRandomSeed(42),
	)

	data := [][]float64{
		{1e9, 1e9},
		{1e9 + 1, 1e9 + 1},
		{1e9 + 2, 1e9 + 2},
		{1e10, 1e10},
		{1e10 + 1, 1e10 + 1},
		{1e10 + 2, 1e10 + 2},
	}

	result, err := kmeans.Cluster(data)

	require.NoError(t, err)
	require.Len(t, result.Centroids, 2)
	require.Positive(t, result.Inertia)
}

func TestCluster_NegativeCoordinates(t *testing.T) {
	t.Parallel()

	kmeans := NewWithOptions(1,
		WithRandomSeed(42),
		WithNInit(1),
	)
	data := [][]float64{{-4, -2}, {-2, -4}}

	result, err := kmeans.Cluster(data)

	require.NoError(t, err)
	require.Equal(t, [][]float64{{-3, -3}}, result.Centroids)
	require.Equal(t, []int{0, 0}, result.Labels)
	require.Equal(t, 4.0, result.Inertia)
}

func TestCluster_IdenticalLargeCoordinatesRemainFinite(t *testing.T) {
	t.Parallel()

	kmeans := NewWithOptions(1,
		WithRandomSeed(42),
		WithNInit(1),
	)

	result, err := kmeans.Cluster([][]float64{{1e308}, {1e308}})

	require.NoError(t, err)
	require.Equal(t, [][]float64{{1e308}}, result.Centroids)
	require.Equal(t, []int{0, 0}, result.Labels)
	require.Zero(t, result.Inertia)
}

func TestCluster_IdenticalMaxFloatCoordinatesRemainFinite(t *testing.T) {
	t.Parallel()

	kmeans := NewWithOptions(1,
		WithRandomSeed(42),
		WithNInit(1),
	)
	data := [][]float64{{math.MaxFloat64}, {math.MaxFloat64}, {math.MaxFloat64}}

	result, err := kmeans.Cluster(data)

	require.NoError(t, err)
	require.Equal(t, [][]float64{{math.MaxFloat64}}, result.Centroids)
	require.Zero(t, result.Inertia)
}

func TestCluster_UnrepresentableInertiaReturnsNumericalOverflow(t *testing.T) {
	t.Parallel()

	kmeans := NewWithOptions(1,
		WithRandomSeed(42),
		WithNInit(1),
	)

	result, err := kmeans.Cluster([][]float64{{-1e308}, {1e308}})

	require.Nil(t, result)
	require.ErrorIs(t, err, ErrNumericalOverflow)
}

func TestCluster_UnrepresentableIntermediateDistancesRemainComparable(t *testing.T) {
	t.Parallel()

	kmeans := NewWithOptions(2,
		WithRandomSeed(3),
		WithNInit(1),
		WithMaxIter(1),
	)

	result, err := kmeans.Cluster([][]float64{{-1e308}, {-1e308}, {1e308}, {1e308}})

	require.NoError(t, err)
	require.ElementsMatch(t, [][]float64{{-1e308}, {1e308}}, result.Centroids)
	require.Zero(t, result.Inertia)
}

func TestCluster_NonFiniteIntermediatesDoNotEstablishConvergence(t *testing.T) {
	t.Parallel()

	kmeans := NewWithOptions(2,
		WithInitMethod(InitRandom),
		WithRandomSeed(3),
		WithNInit(1),
		WithMaxIter(3),
	)

	result, err := kmeans.Cluster([][]float64{{1e308}, {1e308}, {1e307}, {1e307}})

	require.NoError(t, err)
	require.ElementsMatch(t, [][]float64{{1e307}, {1e308}}, result.Centroids)
	require.Zero(t, result.Inertia)
}

func TestCluster_NumericalOverflowAbortsMultipleInitializationRuns(t *testing.T) {
	t.Parallel()

	data := [][]float64{{-1e308}, {-1e308}, {1e308}, {1e308}}
	first, err := NewWithOptions(2,
		WithInitMethod(InitRandom),
		WithRandomSeed(0),
		WithNInit(1),
		WithMaxIter(1),
	).Cluster(data)
	require.NoError(t, err)
	require.NotNil(t, first)
	require.Zero(t, first.Inertia)
	require.ElementsMatch(t, [][]float64{{-1e308}, {1e308}}, first.Centroids)

	kmeans := NewWithOptions(2,
		WithInitMethod(InitRandom),
		WithRandomSeed(0),
		WithNInit(2),
		WithMaxIter(1),
	)

	result, err := kmeans.Cluster(data)

	require.Nil(t, result)
	require.ErrorIs(t, err, ErrNumericalOverflow)
}

func TestCluster_LargeCommonOffsetHasRepresentableResult(t *testing.T) {
	t.Parallel()

	const (
		offset = 1e150
		delta  = 1e140
	)
	kmeans := NewWithOptions(1,
		WithRandomSeed(42),
		WithNInit(1),
	)
	data := [][]float64{{offset - delta}, {offset + delta}}
	wantCentroid := data[0][0]/2 + data[1][0]/2

	result, err := kmeans.Cluster(data)

	require.NoError(t, err)
	require.Equal(t, wantCentroid, result.Centroids[0][0])
	require.Positive(t, result.Inertia)
	require.False(t, math.IsInf(result.Inertia, 0))
	require.False(t, math.IsNaN(result.Inertia))
}

func TestCluster_Lloyd_SmallNumbers(t *testing.T) {
	t.Parallel()
	kmeans := NewWithOptions(2,
		WithInitMethod(InitKMeansPlusPlus),
		WithRandomSeed(42),
	)

	data := [][]float64{
		{1e-9, 1e-9},
		{2e-9, 2e-9},
		{3e-9, 3e-9},
		{1e-8, 1e-8},
		{2e-8, 2e-8},
		{3e-8, 3e-8},
	}

	result, err := kmeans.Cluster(data)

	require.NoError(t, err)
	require.Len(t, result.Centroids, 2)
	require.Positive(t, result.Inertia)
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

func TestLloydKMeans_SingleRun(t *testing.T) {
	t.Parallel()

	kmeans := NewWithOptions(2, WithRandomSeed(42))

	data := [][]float64{
		{1.0, 2.0},
		{3.0, 4.0},
		{10.0, 11.0},
		{12.0, 13.0},
	}

	result, _, converged, err := kmeans.lloydKMeans(
		data, make([]int, len(data)), make([]int, len(data)),
		make([]scaledValue, len(data)), kmeans.newRandomState(), calculateScaledTolerance(data, kmeans.Tol),
	)
	require.True(t, converged)

	require.NoError(t, err)
	require.Len(t, result.Centroids, 2)
}

// Each Lloyd benchmark iteration measures Cluster with seed 42 and the default NInit of 10.
func BenchmarkLloyd_KMeansPlusPlus(b *testing.B) {
	// 1,000 two-dimensional points in two groups; 5 clusters, up to 100 iterations.
	data := make([][]float64, 1000)
	for i := 0; i < 1000; i++ {
		if i < 500 {
			data[i] = []float64{float64(i % 10), float64(i % 10)}
		} else {
			data[i] = []float64{float64(i%10) + 50, float64(i%10) + 50}
		}
	}

	kmeans := NewWithOptions(5,
		WithInitMethod(InitKMeansPlusPlus),
		WithMaxIter(100),
		WithRandomSeed(42),
	)
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		if _, err := kmeans.Cluster(data); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkLloyd_RandomInit(b *testing.B) {
	// 1,000 two-dimensional points in two groups; 5 clusters, up to 100 iterations.
	data := make([][]float64, 1000)
	for i := 0; i < 1000; i++ {
		if i < 500 {
			data[i] = []float64{float64(i % 10), float64(i % 10)}
		} else {
			data[i] = []float64{float64(i%10) + 50, float64(i%10) + 50}
		}
	}

	kmeans := NewWithOptions(5,
		WithInitMethod(InitRandom),
		WithMaxIter(100),
		WithRandomSeed(42),
	)
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		if _, err := kmeans.Cluster(data); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkLloyd_VaryingDataSize(b *testing.B) {
	// Two-dimensional points in two groups; 5 clusters, up to 50 iterations.
	sizes := []int{100, 500, 1000, 5000}

	for _, size := range sizes {
		data := make([][]float64, size)
		for i := 0; i < size; i++ {
			if i < size/2 {
				data[i] = []float64{float64(i % 10), float64(i % 10)}
			} else {
				data[i] = []float64{float64(i%10) + 50, float64(i%10) + 50}
			}
		}

		b.Run(fmt.Sprintf("size_%d", size), func(b *testing.B) {
			kmeans := NewWithOptions(5,
				WithInitMethod(InitKMeansPlusPlus),
				WithMaxIter(50),
				WithRandomSeed(42),
			)
			b.ReportAllocs()
			b.ResetTimer()
			for b.Loop() {
				if _, err := kmeans.Cluster(data); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
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

func requireFiniteScaled(t *testing.T, value scaledValue) float64 {
	t.Helper()
	result, representable := value.float64()
	require.True(t, representable, "expected a finite float64 value")
	return result
}

func TestUpdateScaledMinDistances_IncrementalProperties(t *testing.T) {
	t.Parallel()

	data := [][]float64{{1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}}
	distances := make([]scaledValue, len(data))
	updateScaledMinDistances(distances, data, []float64{0, 0, 0, 0, 0}, true)
	require.Equal(t, 55.0, requireFiniteScaled(t, distances[0]))
	require.Equal(t, 330.0, requireFiniteScaled(t, distances[1]))

	updateScaledMinDistances(distances, data, []float64{5, 5, 5, 5, 5}, false)
	require.Equal(t, 30.0, requireFiniteScaled(t, distances[0]))
	require.Equal(t, 55.0, requireFiniteScaled(t, distances[1]))

	for _, point := range data {
		updateScaledMinDistances(distances, data, point, false)
	}
	require.Equal(t, []scaledValue{{}, {}}, distances)
}

func TestCalculateScaledInertiaByLabels_Additivity(t *testing.T) {
	t.Parallel()

	data := [][]float64{{1, 1}, {9, 9}}
	centers := [][]float64{{0, 0}, {10, 10}}
	labels := []int{0, 1}
	first := calculateScaledInertiaByLabels(data[:1], centers, labels[:1])
	second := calculateScaledInertiaByLabels(data[1:], centers, labels[1:])
	combined := calculateScaledInertiaByLabels(data, centers, labels)
	require.Equal(t, 4.0, requireFiniteScaled(t, combined))
	require.Zero(t, combined.compare(first.add(second)))
}
