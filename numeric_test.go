package kmeans

import (
	"math"
	"testing"

	"github.com/stretchr/testify/require"
)

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
