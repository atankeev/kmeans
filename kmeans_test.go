package kmeans

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
	"reflect"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestKmeans_InitRandomCentroids(t *testing.T) {
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
			// Create Kmeans instance with fixed random seed
			kmeans := NewWithOptions(tt.nClusters, WithRandomSeed(tt.randomSeed))

			// Call the method
			centroids := kmeans.initRandomCentroids(tt.data, rand.New(rand.NewSource(tt.randomSeed)))

			// Verify the number of centroids
			if len(centroids) != tt.nClusters {
				t.Errorf("expected %d centroids, got %d", tt.nClusters, len(centroids))
			}

			// Verify each centroid
			for i, centroid := range centroids {
				// Check that centroid has the same dimensions as data points
				if len(centroid) != len(tt.data[0]) {
					t.Errorf("centroid %d has wrong dimensions: expected %d, got %d",
						i, len(tt.data[0]), len(centroid))
				}

				// Check that centroid is a copy of one of the data points
				if !slices.ContainsFunc(tt.data, func(dataPoint []float64) bool {
					return reflect.DeepEqual(centroid, dataPoint)
				}) {
					t.Errorf("centroid %d is not a copy of any data point", i)
				}
			}

			// Verify that centroids are different from each other (with high probability)
			// This is not guaranteed due to randomness, but we can check for duplicates
			centroidSet := make(map[string]bool)
			for i, centroid := range centroids {
				centroidStr := sliceToString(centroid)
				if centroidSet[centroidStr] {
					t.Errorf("duplicate centroid found at index %d", i)
				}
				centroidSet[centroidStr] = true
			}
		})
	}
}

// Helper function to convert slice to string for map key
func sliceToString(slice []float64) string {
	result := make([]string, 0, len(slice))
	for _, val := range slice {
		result = append(result, fmt.Sprintf("%.6f", val))
	}
	return strings.Join(result, ",")
}

// Test that initRandomCentroids produces different results with different seeds
func TestKmeans_InitRandomCentroids_DifferentSeeds(t *testing.T) {
	data := [][]float64{
		{1.0, 2.0},
		{3.0, 4.0},
		{5.0, 6.0},
		{7.0, 8.0},
	}

	kmeans1 := NewWithOptions(2, WithRandomSeed(42))
	kmeans2 := NewWithOptions(2, WithRandomSeed(123))

	centroids1 := kmeans1.initRandomCentroids(data, rand.New(rand.NewSource(42)))
	centroids2 := kmeans2.initRandomCentroids(data, rand.New(rand.NewSource(123)))

	// Check that results are different (this is probabilistic)
	centroids1Str := sliceToString(centroids1[0]) + "|" + sliceToString(centroids1[1])
	centroids2Str := sliceToString(centroids2[0]) + "|" + sliceToString(centroids2[1])

	if centroids1Str == centroids2Str {
		t.Log("Warning: Same centroids generated with different seeds (this can happen by chance)")
	}
}

// Test edge case with minimum data
func TestKmeans_InitRandomCentroids_EdgeCases(t *testing.T) {
	t.Run("single data point", func(t *testing.T) {
		data := [][]float64{{1.0, 2.0, 3.0}}
		kmeans := NewWithOptions(1, WithRandomSeed(42))

		centroids := kmeans.initRandomCentroids(data, rand.New(rand.NewSource(42)))

		if len(centroids) != 1 {
			t.Errorf("expected 1 centroid, got %d", len(centroids))
		}

		if !reflect.DeepEqual(centroids[0], data[0]) {
			t.Errorf("expected centroid to match data point")
		}
	})

	t.Run("data with different dimensions", func(t *testing.T) {
		data := [][]float64{
			{1.0, 2.0},
			{3.0, 4.0, 5.0}, // Different dimension
		}
		kmeans := NewWithOptions(1, WithRandomSeed(42))

		// This should handle the case gracefully
		centroids := kmeans.initRandomCentroids(data, rand.New(rand.NewSource(42)))

		if len(centroids) != 1 {
			t.Errorf("expected 1 centroid, got %d", len(centroids))
		}
	})
}

// Test validateData method
func TestKmeans_ValidateData(t *testing.T) {
	t.Run("valid data", func(t *testing.T) {
		data := [][]float64{
			{1.0, 2.0, 3.0},
			{4.0, 5.0, 6.0},
			{7.0, 8.0, 9.0},
		}
		kmeans := New(2)

		err := kmeans.validateData(data)
		if err != nil {
			t.Errorf("expected no error for valid data, got %v", err)
		}
	})

	t.Run("empty data", func(t *testing.T) {
		var data [][]float64
		kmeans := New(2)

		err := kmeans.validateData(data)
		if !errors.Is(err, ErrEmptyData) {
			t.Errorf("expected ErrEmptyData, got %v", err)
		}
	})

	t.Run("nil data", func(t *testing.T) {
		var data [][]float64
		kmeans := New(2)

		err := kmeans.validateData(data)
		if !errors.Is(err, ErrEmptyData) {
			t.Errorf("expected ErrEmptyData, got %v", err)
		}
	})

	t.Run("inconsistent dimensions", func(t *testing.T) {
		data := [][]float64{
			{1.0, 2.0, 3.0},
			{4.0, 5.0}, // Different dimension
			{7.0, 8.0, 9.0},
		}
		kmeans := New(2)

		err := kmeans.validateData(data)
		if err == nil {
			t.Error("expected error for inconsistent dimensions")
		}
		if !strings.Contains(err.Error(), "dimension") {
			t.Errorf("expected dimension error, got %v", err)
		}
	})

	t.Run("more clusters than data points", func(t *testing.T) {
		data := [][]float64{
			{1.0, 2.0},
			{3.0, 4.0},
		}

		// More clusters than data points
		kmeans := New(3)

		err := kmeans.validateData(data)
		require.ErrorIs(t, err, ErrInvalidK)
		if !strings.Contains(err.Error(), "cannot be greater than number of samples") {
			t.Errorf("expected cluster count error, got %v", err)
		}
	})

	t.Run("equal clusters and data points", func(t *testing.T) {
		data := [][]float64{
			{1.0, 2.0},
			{3.0, 4.0},
		}
		// Equal to number of data points
		kmeans := New(2)

		err := kmeans.validateData(data)
		if err != nil {
			t.Errorf("expected no error for equal clusters and data points, got %v", err)
		}
	})

	t.Run("single data point", func(t *testing.T) {
		data := [][]float64{{1.0, 2.0, 3.0}}
		kmeans := New(1)

		err := kmeans.validateData(data)
		if err != nil {
			t.Errorf("expected no error for single data point, got %v", err)
		}
	})
}

// TestSquaredEuclideanDistance tests the squaredEuclideanDistance function
func TestSquaredEuclideanDistance(t *testing.T) {
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
			result := squaredEuclideanDistance(tt.p1, tt.p2)

			// Use approximate comparison for floating point values
			if !isApproximatelyEqual(result, tt.expected, 1e-10) {
				t.Errorf("squaredEuclideanDistance(%v, %v) = %f, want %f",
					tt.p1, tt.p2, result, tt.expected)
			}
		})
	}
}

// TestSquaredEuclideanDistance_EdgeCases tests edge cases for the squaredEuclideanDistance function
func TestSquaredEuclideanDistance_EdgeCases(t *testing.T) {
	t.Run("empty slices", func(t *testing.T) {
		result := squaredEuclideanDistance([]float64{}, []float64{})
		expected := 0.0
		if result != expected {
			t.Errorf("squaredEuclideanDistance([], []) = %f, want %f", result, expected)
		}
	})

	t.Run("nil slices", func(t *testing.T) {
		result := squaredEuclideanDistance(nil, nil)
		expected := 0.0
		if result != expected {
			t.Errorf("squaredEuclideanDistance(nil, nil) = %f, want %f", result, expected)
		}
	})

	t.Run("very large numbers", func(t *testing.T) {
		p1 := []float64{1e10, 2e10}
		p2 := []float64{3e10, 4e10}
		result := squaredEuclideanDistance(p1, p2)
		expected := 8e20 // (2e10)² + (2e10)² = 4e20 + 4e20 = 8e20
		if !isApproximatelyEqual(result, expected, 1e-10) {
			t.Errorf("squaredEuclideanDistance(%v, %v) = %e, want %e", p1, p2, result, expected)
		}
	})

	t.Run("very small numbers", func(t *testing.T) {
		p1 := []float64{1e-10, 2e-10}
		p2 := []float64{3e-10, 4e-10}
		result := squaredEuclideanDistance(p1, p2)
		expected := 8e-20 // (2e-10)² + (2e-10)² = 4e-20 + 4e-20 = 8e-20
		if !isApproximatelyEqual(result, expected, 1e-30) {
			t.Errorf("squaredEuclideanDistance(%v, %v) = %e, want %e", p1, p2, result, expected)
		}
	})
}

// TestSquaredEuclideanDistance_Properties tests mathematical properties of the function
func TestSquaredEuclideanDistance_Properties(t *testing.T) {
	t.Run("commutativity", func(t *testing.T) {
		p1 := []float64{1.0, 2.0, 3.0}
		p2 := []float64{4.0, 5.0, 6.0}

		d1 := squaredEuclideanDistance(p1, p2)
		d2 := squaredEuclideanDistance(p2, p1)

		if !isApproximatelyEqual(d1, d2, 1e-10) {
			t.Errorf("distance should be commutative: %f != %f", d1, d2)
		}
	})

	t.Run("triangle inequality for squared distances", func(t *testing.T) {
		// Note: squared Euclidean distance does NOT satisfy triangle inequality
		// This test demonstrates that property
		p1 := []float64{0.0, 0.0}
		p2 := []float64{3.0, 0.0}
		p3 := []float64{3.0, 4.0}

		d12 := squaredEuclideanDistance(p1, p2) // 9
		d23 := squaredEuclideanDistance(p2, p3) // 16
		d13 := squaredEuclideanDistance(p1, p3) // 25

		// d13 > d12 + d23 (25 > 9 + 16 = 25 is false, but 25 > 9 + 16 = 25 is false)
		// This shows that squared distance doesn't satisfy triangle inequality
		if d13 <= d12+d23 {
			t.Logf("Squared distance doesn't satisfy triangle inequality: %f <= %f + %f", d13, d12, d23)
		}
	})

	t.Run("scaling property", func(t *testing.T) {
		p1 := []float64{1.0, 2.0}
		p2 := []float64{3.0, 4.0}
		scale := 2.0

		d1 := squaredEuclideanDistance(p1, p2)

		// Scale both points
		scaledP1 := []float64{p1[0] * scale, p1[1] * scale}
		scaledP2 := []float64{p2[0] * scale, p2[1] * scale}
		d2 := squaredEuclideanDistance(scaledP1, scaledP2)

		expected := d1 * scale * scale
		if !isApproximatelyEqual(d2, expected, 1e-10) {
			t.Errorf("scaled distance should be %f, got %f", expected, d2)
		}
	})
}

// Helper function to compare floating point values with tolerance
func isApproximatelyEqual(a, b, tolerance float64) bool {
	return (a-b) <= tolerance && (b-a) <= tolerance
}

// TestCalculateInertia tests the calculateInertia function
func TestCalculateInertia(t *testing.T) {
	tests := []struct {
		name     string
		data     [][]float64
		centers  [][]float64
		expected float64
	}{
		{
			name: "single center, single point",
			data: [][]float64{
				{1.0, 2.0},
			},
			centers: [][]float64{
				{1.0, 2.0},
			},
			expected: 0.0, // Point is exactly at center
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
			expected: 125.0, // 0 + 25 + 100 = 125
		},
		{
			name: "two centers, points assigned to nearest",
			data: [][]float64{
				{1.0, 1.0}, // Closer to center1
				{9.0, 9.0}, // Closer to center2
				{2.0, 2.0}, // Closer to center1
				{8.0, 8.0}, // Closer to center2
			},
			centers: [][]float64{
				{0.0, 0.0},   // center1
				{10.0, 10.0}, // center2
			},
			expected: 20.0, // 2 + 2 + 8 + 8 = 20
		},
		{
			name: "three centers, optimal assignment",
			data: [][]float64{
				{1.0, 1.0}, // Closest to center1
				{5.0, 5.0}, // Closest to center2
				{9.0, 9.0}, // Closest to center3
			},
			centers: [][]float64{
				{0.0, 0.0},   // center1
				{5.0, 5.0},   // center2
				{10.0, 10.0}, // center3
			},
			expected: 4.0, // 2 + 0 + 2 = 4
		},
		{
			name: "empty data",
			data: [][]float64{},
			centers: [][]float64{
				{1.0, 2.0},
				{3.0, 4.0},
			},
			expected: 0.0,
		},
		{
			name: "empty centers",
			data: [][]float64{
				{1.0, 2.0},
				{3.0, 4.0},
			},
			centers:  [][]float64{},
			expected: math.Inf(1) * 2, // Each point gets Inf distance
		},
		{
			name: "3D points",
			data: [][]float64{
				{1.0, 2.0, 3.0},
				{4.0, 5.0, 6.0},
			},
			centers: [][]float64{
				{0.0, 0.0, 0.0},
				{5.0, 5.0, 5.0},
			},
			expected: 16.0, // 14 + 2 = 16
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := calculateInertia(tt.data, tt.centers)

			if math.IsInf(tt.expected, 1) {
				// Handle infinite expected values
				if !math.IsInf(result, 1) {
					t.Errorf("calculateInertia() = %f, want Inf", result)
				}
			} else {
				// Use approximate comparison for finite values
				if !isApproximatelyEqual(result, tt.expected, 1e-10) {
					t.Errorf("calculateInertia() = %f, want %f", result, tt.expected)
				}
			}
		})
	}
}

// TestCalculateInertia_EdgeCases tests edge cases for the calculateInertia function
func TestCalculateInertia_EdgeCases(t *testing.T) {
	t.Run("nil data", func(t *testing.T) {
		var data [][]float64
		centers := [][]float64{{1.0, 2.0}}
		result := calculateInertia(data, centers)
		expected := 0.0
		if result != expected {
			t.Errorf("calculateInertia(nil, %v) = %f, want %f", centers, result, expected)
		}
	})

	t.Run("nil centers", func(t *testing.T) {
		data := [][]float64{{1.0, 2.0}, {3.0, 4.0}}
		var centers [][]float64
		result := calculateInertia(data, centers)
		if !math.IsInf(result, 1) {
			t.Errorf("calculateInertia(%v, nil) = %f, want Inf", data, result)
		}
	})

	t.Run("single point with multiple centers", func(t *testing.T) {
		data := [][]float64{{5.0, 5.0}}
		centers := [][]float64{
			{0.0, 0.0},   // distance = 50
			{10.0, 10.0}, // distance = 50
			{5.0, 5.0},   // distance = 0 (closest)
		}
		result := calculateInertia(data, centers)
		expected := 0.0
		if !isApproximatelyEqual(result, expected, 1e-10) {
			t.Errorf("calculateInertia() = %f, want %f", result, expected)
		}
	})

	t.Run("points equidistant from centers", func(t *testing.T) {
		data := [][]float64{
			{5.0, 0.0}, // Equidistant from both centers
		}
		centers := [][]float64{
			{0.0, 0.0},  // distance = 25
			{10.0, 0.0}, // distance = 25
		}
		result := calculateInertia(data, centers)
		expected := 25.0 // Should pick the minimum (both are equal)
		if !isApproximatelyEqual(result, expected, 1e-10) {
			t.Errorf("calculateInertia() = %f, want %f", result, expected)
		}
	})
}

// TestCalculateInertia_Properties tests mathematical properties of the function
func TestCalculateInertia_Properties(t *testing.T) {
	t.Run("monotonicity with center movement", func(t *testing.T) {
		data := [][]float64{{1.0, 1.0}, {9.0, 9.0}}
		centers1 := [][]float64{{0.0, 0.0}, {10.0, 10.0}}
		centers2 := [][]float64{{1.0, 1.0}, {9.0, 9.0}} // Centers exactly at data points

		inertia1 := calculateInertia(data, centers1)
		inertia2 := calculateInertia(data, centers2)

		// Moving centers to data points should reduce inertia to zero
		if inertia2 >= inertia1 {
			t.Errorf("Moving centers to data points should reduce inertia: %f >= %f", inertia2, inertia1)
		}
		if inertia2 != 0.0 {
			t.Errorf("Centers at data points should give zero inertia: %f", inertia2)
		}
	})

	t.Run("additivity", func(t *testing.T) {
		data1 := [][]float64{{1.0, 1.0}}
		data2 := [][]float64{{9.0, 9.0}}
		centers := [][]float64{{0.0, 0.0}, {10.0, 10.0}}

		inertia1 := calculateInertia(data1, centers)
		inertia2 := calculateInertia(data2, centers)
		inertiaCombined := calculateInertia(append(data1, data2...), centers)

		// Inertia should be additive
		expected := inertia1 + inertia2
		if !isApproximatelyEqual(inertiaCombined, expected, 1e-10) {
			t.Errorf("Inertia should be additive: %f != %f + %f", inertiaCombined, inertia1, inertia2)
		}
	})

	t.Run("optimal centers give zero inertia", func(t *testing.T) {
		data := [][]float64{{1.0, 1.0}, {5.0, 5.0}, {9.0, 9.0}}
		centers := [][]float64{{1.0, 1.0}, {5.0, 5.0}, {9.0, 9.0}} // Centers exactly at data points

		inertia := calculateInertia(data, centers)
		expected := 0.0
		if !isApproximatelyEqual(inertia, expected, 1e-10) {
			t.Errorf("Optimal centers should give zero inertia: %f != %f", inertia, expected)
		}
	})
}

// TestComputeDistancesToCenters tests the computeDistancesToCenters function
func TestComputeDistancesToCenters(t *testing.T) {
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
			name: "empty centers",
			data: [][]float64{
				{1.0, 2.0},
				{3.0, 4.0},
			},
			centers:  [][]float64{},
			expected: []float64{math.Inf(1), math.Inf(1)},
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
			result := computeDistancesToCenters(tt.data, tt.centers, make([]float64, len(tt.data)))

			if len(result) != len(tt.expected) {
				t.Errorf("computeDistancesToCenters() returned %d distances, want %d", len(result), len(tt.expected))
				return
			}

			for i, expected := range tt.expected {
				if math.IsInf(expected, 1) {
					// Handle infinite expected values
					if !math.IsInf(result[i], 1) {
						t.Errorf("computeDistancesToCenters()[%d] = %f, want Inf", i, result[i])
					}
				} else {
					// Use approximate comparison for finite values
					if !isApproximatelyEqual(result[i], expected, 1e-10) {
						t.Errorf("computeDistancesToCenters()[%d] = %f, want %f", i, result[i], expected)
					}
				}
			}
		})
	}
}

// TestComputeDistancesToCenters_EdgeCases tests edge cases for the computeDistancesToCenters function
func TestComputeDistancesToCenters_EdgeCases(t *testing.T) {
	t.Run("nil data", func(t *testing.T) {
		var data [][]float64
		centers := [][]float64{{1.0, 2.0}}
		result := computeDistancesToCenters(data, centers, make([]float64, len(data)))
		expected := make([]float64, 0)
		if !reflect.DeepEqual(result, expected) {
			t.Errorf("computeDistancesToCenters(nil, %v) = %v, want %v", centers, result, expected)
		}
	})

	t.Run("nil centers", func(t *testing.T) {
		data := [][]float64{{1.0, 2.0}, {3.0, 4.0}}
		var centers [][]float64
		result := computeDistancesToCenters(data, centers, make([]float64, len(data)))
		expected := []float64{math.Inf(1), math.Inf(1)}
		if !reflect.DeepEqual(result, expected) {
			t.Errorf("computeDistancesToCenters(%v, nil) = %v, want %v", data, result, expected)
		}
	})

	t.Run("single point with multiple centers", func(t *testing.T) {
		data := [][]float64{{5.0, 5.0}}
		centers := [][]float64{
			{0.0, 0.0},   // distance = 50
			{10.0, 10.0}, // distance = 50
			{5.0, 5.0},   // distance = 0 (closest)
		}
		result := computeDistancesToCenters(data, centers, make([]float64, len(data)))
		expected := []float64{0.0}
		if !reflect.DeepEqual(result, expected) {
			t.Errorf("computeDistancesToCenters() = %v, want %v", result, expected)
		}
	})

	t.Run("high dimensional data", func(t *testing.T) {
		data := [][]float64{
			{1.0, 2.0, 3.0, 4.0, 5.0},
			{6.0, 7.0, 8.0, 9.0, 10.0},
		}
		centers := [][]float64{
			{0.0, 0.0, 0.0, 0.0, 0.0},
			{5.0, 5.0, 5.0, 5.0, 5.0},
		}
		result := computeDistancesToCenters(data, centers, make([]float64, len(data)))
		// Point 1: distance to center1 = 1²+2²+3²+4²+5² = 55, distance to center2 = 4²+3²+2²+1²+0² = 30 (closer)
		// Point 2: distance to center1 = 6²+7²+8²+9²+10² = 330, distance to center2 = 1²+2²+3²+4²+5² = 55 (closer)
		expected := []float64{30.0, 55.0}
		for i, expected := range expected {
			if !isApproximatelyEqual(result[i], expected, 1e-10) {
				t.Errorf("computeDistancesToCenters()[%d] = %f, want %f", i, result[i], expected)
			}
		}
	})
}

// TestComputeDistancesToCenters_Properties tests mathematical properties of the function
func TestComputeDistancesToCenters_Properties(t *testing.T) {
	t.Run("monotonicity with center movement", func(t *testing.T) {
		data := [][]float64{{1.0, 1.0}, {9.0, 9.0}}
		centers1 := [][]float64{{0.0, 0.0}, {10.0, 10.0}}
		centers2 := [][]float64{{1.0, 1.0}, {9.0, 9.0}} // Centers exactly at data points

		distances1 := computeDistancesToCenters(data, centers1, make([]float64, len(data)))
		distances2 := computeDistancesToCenters(data, centers2, make([]float64, len(data)))

		// Moving centers to data points should reduce distances to zero
		for i := range distances2 {
			if distances2[i] >= distances1[i] {
				t.Errorf("Moving centers to data points should reduce distances: distances2[%d] = %f >= distances1[%d] = %f",
					i, distances2[i], i, distances1[i])
			}
			if distances2[i] != 0.0 {
				t.Errorf("Centers at data points should give zero distance: distances2[%d] = %f", i, distances2[i])
			}
		}
	})

	t.Run("consistency with calculateInertia", func(t *testing.T) {
		data := [][]float64{{1.0, 1.0}, {9.0, 9.0}}
		centers := [][]float64{{0.0, 0.0}, {10.0, 10.0}}

		distances := computeDistancesToCenters(data, centers, make([]float64, len(data)))
		inertia := calculateInertia(data, centers)

		// Sum of distances should equal inertia
		sum := 0.0
		for _, dist := range distances {
			sum += dist
		}

		if !isApproximatelyEqual(sum, inertia, 1e-10) {
			t.Errorf("Sum of distances (%f) should equal inertia (%f)", sum, inertia)
		}
	})

	t.Run("all distances non-negative", func(t *testing.T) {
		data := [][]float64{{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}}
		centers := [][]float64{{0.0, 0.0}, {10.0, 10.0}}

		distances := computeDistancesToCenters(data, centers, make([]float64, len(data)))

		for i, dist := range distances {
			if dist < 0 {
				t.Errorf("Distance[%d] = %f should be non-negative", i, dist)
			}
		}
	})
}

// TestWeightedRandomChoice tests the weightedRandomChoice function
func TestWeightedRandomChoice(t *testing.T) {
	t.Run("single weight", func(t *testing.T) {
		weights := []float64{1.0}
		rng := rand.New(rand.NewSource(42))

		result := weightedRandomChoice(weights, rng)
		expected := 0
		if result != expected {
			t.Errorf("weightedRandomChoice([1.0]) = %d, want %d", result, expected)
		}
	})

	t.Run("equal weights", func(t *testing.T) {
		weights := []float64{1.0, 1.0, 1.0}
		rng := rand.New(rand.NewSource(42))

		// Run multiple times to check distribution
		counts := make([]int, len(weights))
		nTrials := 1000

		for range nTrials {
			result := weightedRandomChoice(weights, rng)
			if result >= 0 && result < len(weights) {
				counts[result]++
			}
		}

		// Check that all indices were selected (with some tolerance for randomness)
		for i, count := range counts {
			if count == 0 {
				t.Errorf("Index %d was never selected in %d trials", i, nTrials)
			}
		}
	})

	t.Run("zero weights", func(t *testing.T) {
		weights := []float64{0.0, 0.0, 0.0}
		rng := rand.New(rand.NewSource(42))

		result := weightedRandomChoice(weights, rng)
		// Should return a random index in [0, len(weights))
		if result < 0 || result >= len(weights) {
			t.Errorf("weightedRandomChoice([0,0,0]) = %d, want in [0,%d]", result, len(weights)-1)
		}
	})

	t.Run("mixed zero and non-zero weights", func(t *testing.T) {
		weights := []float64{0.0, 1.0, 0.0, 2.0}
		rng := rand.New(rand.NewSource(42))

		// Run multiple times to check distribution
		counts := make([]int, len(weights))
		nTrials := 1000

		for range nTrials {
			result := weightedRandomChoice(weights, rng)
			if result >= 0 && result < len(weights) {
				counts[result]++
			}
		}

		// Index 1 should be selected about 1/3 of the time, index 3 about 2/3 of the time
		// Index 0 and 2 should never be selected (zero weights)
		if counts[0] != 0 {
			t.Errorf("Index 0 (zero weight) was selected %d times, should be 0", counts[0])
		}
		if counts[2] != 0 {
			t.Errorf("Index 2 (zero weight) was selected %d times, should be 0", counts[2])
		}
		if counts[1] == 0 {
			t.Errorf("Index 1 was never selected")
		}
		if counts[3] == 0 {
			t.Errorf("Index 3 was never selected")
		}

		// Check approximate ratio (allowing for randomness)
		ratio := float64(counts[3]) / float64(counts[1])
		expectedRatio := 2.0 // weight[3] / weight[1] = 2.0 / 1.0
		if ratio < 1.5 || ratio > 2.5 {
			t.Errorf("Ratio of selections (index3/index1) = %f, expected around %f", ratio, expectedRatio)
		}
	})

	t.Run("very large weights", func(t *testing.T) {
		weights := []float64{1e10, 2e10, 3e10}
		rng := rand.New(rand.NewSource(42))

		// Run multiple times to check distribution
		counts := make([]int, len(weights))
		nTrials := 1000

		for range nTrials {
			result := weightedRandomChoice(weights, rng)
			if result >= 0 && result < len(weights) {
				counts[result]++
			}
		}

		// All indices should be selected
		for i, count := range counts {
			if count == 0 {
				t.Errorf("Index %d was never selected in %d trials", i, nTrials)
			}
		}

		// Check approximate ratios
		ratio1 := float64(counts[1]) / float64(counts[0])
		ratio2 := float64(counts[2]) / float64(counts[1])
		expectedRatio1 := 2.0 // weight[1] / weight[0] = 2e10 / 1e10
		expectedRatio2 := 1.5 // weight[2] / weight[1] = 3e10 / 2e10

		if ratio1 < 1.5 || ratio1 > 2.5 {
			t.Errorf("Ratio of selections (index1/index0) = %f, expected around %f", ratio1, expectedRatio1)
		}
		if ratio2 < 1.2 || ratio2 > 1.8 {
			t.Errorf("Ratio of selections (index2/index1) = %f, expected around %f", ratio2, expectedRatio2)
		}
	})

	t.Run("very small weights", func(t *testing.T) {
		weights := []float64{1e-10, 2e-10, 3e-10}
		rng := rand.New(rand.NewSource(42))

		// Run multiple times to check distribution
		counts := make([]int, len(weights))
		nTrials := 1000

		for range nTrials {
			result := weightedRandomChoice(weights, rng)
			if result >= 0 && result < len(weights) {
				counts[result]++
			}
		}

		// All indices should be selected
		for i, count := range counts {
			if count == 0 {
				t.Errorf("Index %d was never selected in %d trials", i, nTrials)
			}
		}
	})
}

// TestWeightedRandomChoice_EdgeCases tests edge cases for the weightedRandomChoice function
func TestWeightedRandomChoice_EdgeCases(t *testing.T) {
	t.Run("negative weights", func(t *testing.T) {
		weights := []float64{-1.0, 2.0, -3.0}
		rng := rand.New(rand.NewSource(42))

		// Should handle negative weights gracefully
		result := weightedRandomChoice(weights, rng)
		if result < 0 || result >= len(weights) {
			t.Errorf("weightedRandomChoice returned invalid index %d", result)
		}
	})

	t.Run("single non-zero weight", func(t *testing.T) {
		weights := []float64{0.0, 5.0, 0.0}
		rng := rand.New(rand.NewSource(42))

		// Should always return index 1
		for range 100 {
			result := weightedRandomChoice(weights, rng)
			if result != 1 {
				t.Errorf("weightedRandomChoice returned %d, expected 1", result)
			}
		}
	})

	t.Run("all weights are NaN", func(t *testing.T) {
		weights := []float64{math.NaN(), math.NaN(), math.NaN()}
		rng := rand.New(rand.NewSource(42))
		result := weightedRandomChoice(weights, rng)
		if result < 0 || result >= len(weights) {
			t.Errorf("weightedRandomChoice([NaN,NaN,NaN]) = %d, want in [0,%d]", result, len(weights)-1)
		}
	})

	t.Run("all weights are +Inf", func(t *testing.T) {
		weights := []float64{math.Inf(1), math.Inf(1), math.Inf(1)}
		rng := rand.New(rand.NewSource(42))
		result := weightedRandomChoice(weights, rng)
		if result < 0 || result >= len(weights) {
			t.Errorf("weightedRandomChoice([Inf,Inf,Inf]) = %d, want in [0,%d]", result, len(weights)-1)
		}
	})
}

// TestWeightedRandomChoice_Properties tests mathematical properties of the function
func TestWeightedRandomChoice_Properties(t *testing.T) {
	t.Run("deterministic with same seed", func(t *testing.T) {
		weights := []float64{1.0, 2.0, 3.0}
		rng1 := rand.New(rand.NewSource(42))
		rng2 := rand.New(rand.NewSource(42))

		// Should produce same sequence with same seed
		for range 10 {
			result1 := weightedRandomChoice(weights, rng1)
			result2 := weightedRandomChoice(weights, rng2)
			if result1 != result2 {
				t.Errorf("Results differ with same seed: %d != %d", result1, result2)
			}
		}
	})

	t.Run("different seeds produce different results", func(t *testing.T) {
		weights := []float64{1.0, 2.0, 3.0}
		rng1 := rand.New(rand.NewSource(42))
		rng2 := rand.New(rand.NewSource(123))

		// Should produce different sequences with different seeds
		different := false
		for range 10 {
			result1 := weightedRandomChoice(weights, rng1)
			result2 := weightedRandomChoice(weights, rng2)
			if result1 != result2 {
				different = true
				break
			}
		}

		if !different {
			t.Error("Results should differ with different seeds")
		}
	})

	t.Run("scaling weights preserves distribution", func(t *testing.T) {
		weights1 := []float64{1.0, 2.0, 3.0}
		weights2 := []float64{10.0, 20.0, 30.0} // Scaled by 10
		rng1 := rand.New(rand.NewSource(42))
		rng2 := rand.New(rand.NewSource(42))

		// Should produce same sequence when weights are scaled
		for range 10 {
			result1 := weightedRandomChoice(weights1, rng1)
			result2 := weightedRandomChoice(weights2, rng2)
			if result1 != result2 {
				t.Errorf("Results differ with scaled weights: %d != %d", result1, result2)
			}
		}
	})

	t.Run("valid index range", func(t *testing.T) {
		weights := []float64{1.0, 2.0, 3.0, 4.0, 5.0}
		rng := rand.New(rand.NewSource(42))

		// Should always return valid indices
		for range 100 {
			result := weightedRandomChoice(weights, rng)
			if result < 0 || result >= len(weights) {
				t.Errorf("weightedRandomChoice returned invalid index %d", result)
			}
		}
	})
}

// TestKmeans_InitKMeansPlusPlusCentroids tests the initKMeansPlusPlusCentroids function
func TestKmeans_InitKMeansPlusPlusCentroids(t *testing.T) {
	t.Run("no duplicate centroids", func(t *testing.T) {
		data := [][]float64{
			{1.0, 1.0},
			{2.0, 2.0},
			{3.0, 3.0},
			{4.0, 4.0},
			{5.0, 5.0},
			{6.0, 6.0},
		}

		kmeans := NewWithOptions(3, WithRandomSeed(42))

		centroids := kmeans.initKMeansPlusPlusCentroids(data)

		// Check that we got the expected number of centroids
		if len(centroids) != 3 {
			t.Errorf("Expected 3 centroids, got %d", len(centroids))
		}

		// Check that all centroids are unique (no duplicates)
		centroidSet := make(map[string]bool)
		for i, centroid := range centroids {
			centroidStr := sliceToString(centroid)
			if centroidSet[centroidStr] {
				t.Errorf("Duplicate centroid found at index %d: %v", i, centroid)
			}
			centroidSet[centroidStr] = true
		}

		// Check that all centroids are copies of data points
		for i, centroid := range centroids {
			found := slices.ContainsFunc(data, func(dataPoint []float64) bool {
				return reflect.DeepEqual(centroid, dataPoint)
			})
			if !found {
				t.Errorf("Centroid %d is not a copy of any data point: %v", i, centroid)
			}
		}
	})

	t.Run("single cluster", func(t *testing.T) {
		data := [][]float64{
			{1.0, 2.0, 3.0},
			{4.0, 5.0, 6.0},
			{7.0, 8.0, 9.0},
		}

		kmeans := NewWithOptions(1, WithRandomSeed(42))

		centroids := kmeans.initKMeansPlusPlusCentroids(data)

		if len(centroids) != 1 {
			t.Errorf("Expected 1 centroid, got %d", len(centroids))
		}

		// Check that centroid is a copy of a data point
		found := slices.ContainsFunc(data, func(dataPoint []float64) bool {
			return reflect.DeepEqual(centroids[0], dataPoint)
		})
		if !found {
			t.Errorf("Centroid is not a copy of any data point: %v", centroids[0])
		}
	})

	t.Run("clusters equal to data points", func(t *testing.T) {
		data := [][]float64{
			{1.0, 1.0},
			{2.0, 2.0},
			{3.0, 3.0},
		}

		kmeans := NewWithOptions(3, WithRandomSeed(42))

		centroids := kmeans.initKMeansPlusPlusCentroids(data)

		if len(centroids) != 3 {
			t.Errorf("Expected 3 centroids, got %d", len(centroids))
		}

		// Check that all centroids are unique
		centroidSet := make(map[string]bool)
		for i, centroid := range centroids {
			centroidStr := sliceToString(centroid)
			if centroidSet[centroidStr] {
				t.Errorf("Duplicate centroid found at index %d: %v", i, centroid)
			}
			centroidSet[centroidStr] = true
		}

		// Check that all data points are represented as centroids
		dataSet := make(map[string]bool)
		for _, dataPoint := range data {
			dataSet[sliceToString(dataPoint)] = true
		}

		for i, centroid := range centroids {
			centroidStr := sliceToString(centroid)
			if !dataSet[centroidStr] {
				t.Errorf("Centroid %d is not a data point: %v", i, centroid)
			}
		}
	})

	t.Run("deterministic with same seed", func(t *testing.T) {
		data := [][]float64{
			{1.0, 1.0},
			{2.0, 2.0},
			{3.0, 3.0},
			{4.0, 4.0},
			{5.0, 5.0},
		}

		kmeans1 := NewWithOptions(3, WithRandomSeed(42))
		kmeans2 := NewWithOptions(3, WithRandomSeed(42))

		centroids1 := kmeans1.initKMeansPlusPlusCentroids(data)
		centroids2 := kmeans2.initKMeansPlusPlusCentroids(data)

		// Should produce same centroids with same seed
		if len(centroids1) != len(centroids2) {
			t.Errorf("Different number of centroids: %d vs %d", len(centroids1), len(centroids2))
		}

		for i := range centroids1 {
			if !reflect.DeepEqual(centroids1[i], centroids2[i]) {
				t.Errorf("Centroids differ at index %d: %v vs %v", i, centroids1[i], centroids2[i])
			}
		}
	})

	t.Run("different seeds produce different results", func(t *testing.T) {
		data := [][]float64{
			{1.0, 1.0},
			{2.0, 2.0},
			{3.0, 3.0},
			{4.0, 4.0},
			{5.0, 5.0},
		}

		kmeans1 := NewWithOptions(3, WithRandomSeed(42))
		kmeans2 := NewWithOptions(3, WithRandomSeed(123456))

		centroids1 := kmeans1.initKMeansPlusPlusCentroids(data)
		centroids2 := kmeans2.initKMeansPlusPlusCentroids(data)

		// Should produce different centroids with different seeds
		different := !reflect.DeepEqual(centroids1, centroids2)
		if !different {
			t.Error("Centroids should differ with different seeds")
		}
	})

	t.Run("duplicate data points", func(t *testing.T) {
		data := [][]float64{
			{1.0, 1.0},
			{1.0, 1.0},
			{1.0, 1.0},
			{10.0, 10.0},
			{10.0, 10.0},
		}

		kmeans := NewWithOptions(3, WithRandomSeed(42))

		// This should complete successfully even if greedy selection encounters issues
		centroids := kmeans.initKMeansPlusPlusCentroids(data)

		// Verify we got the expected number of centroids
		if len(centroids) != 3 {
			t.Errorf("Expected 3 centroids, got %d", len(centroids))
		}

		// Verify all centroids are copies of data points
		for i, centroid := range centroids {
			found := slices.ContainsFunc(data, func(dataPoint []float64) bool {
				return reflect.DeepEqual(centroid, dataPoint)
			})
			if !found {
				t.Errorf("Centroid %d is not a copy of any data point: %v", i, centroid)
			}
		}

		// Verify that the number of unique centroids does not exceed the number of unique data points
		centroidSet := make(map[string]bool)
		for _, centroid := range centroids {
			centroidSet[sliceToString(centroid)] = true
		}
		uniqueDataSet := make(map[string]bool)
		for _, dataPoint := range data {
			uniqueDataSet[sliceToString(dataPoint)] = true
		}
		if len(centroidSet) > len(uniqueDataSet) {
			t.Errorf("More unique centroids than unique data points")
		}
	})

	t.Run("greedy k-means++ mode (nLocalTrials >= 2)", func(t *testing.T) {
		// Тест с k=2, теперь nLocalTrials = int(2 + math.Log(2)) = 2
		data := [][]float64{
			{1.0, 1.0},
			{2.0, 2.0},
			{3.0, 3.0},
			{4.0, 4.0},
			{5.0, 5.0},
		}

		kmeans := NewWithOptions(2, WithRandomSeed(42))

		centroids := kmeans.initKMeansPlusPlusCentroids(data)

		if len(centroids) != 2 {
			t.Errorf("Expected 2 centroids, got %d", len(centroids))
		}

		// Verify all centroids are unique
		centroidSet := make(map[string]bool)
		for i, centroid := range centroids {
			centroidStr := sliceToString(centroid)
			if centroidSet[centroidStr] {
				t.Errorf("Duplicate centroid found at index %d: %v", i, centroid)
			}
			centroidSet[centroidStr] = true
		}
	})

	t.Run("greedy k-means++ mode (nLocalTrials > 1)", func(t *testing.T) {
		// Test with k=5 to ensure we're in greedy mode (nLocalTrials = 3)
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

		centroids := kmeans.initKMeansPlusPlusCentroids(data)

		if len(centroids) != 5 {
			t.Errorf("Expected 5 centroids, got %d", len(centroids))
		}

		// Verify all centroids are unique
		centroidSet := make(map[string]bool)
		for i, centroid := range centroids {
			centroidStr := sliceToString(centroid)
			if centroidSet[centroidStr] {
				t.Errorf("Duplicate centroid found at index %d: %v", i, centroid)
			}
			centroidSet[centroidStr] = true
		}
	})

	t.Run("high dimensional data", func(t *testing.T) {
		// Test with high dimensional data points
		data := [][]float64{
			{1.0, 2.0, 3.0, 4.0, 5.0},
			{2.0, 3.0, 4.0, 5.0, 6.0},
			{3.0, 4.0, 5.0, 6.0, 7.0},
			{4.0, 5.0, 6.0, 7.0, 8.0},
			{5.0, 6.0, 7.0, 8.0, 9.0},
			{6.0, 7.0, 8.0, 9.0, 10.0},
		}

		kmeans := NewWithOptions(3, WithRandomSeed(42))

		centroids := kmeans.initKMeansPlusPlusCentroids(data)

		if len(centroids) != 3 {
			t.Errorf("Expected 3 centroids, got %d", len(centroids))
		}

		// Verify all centroids have correct dimensions
		for i, centroid := range centroids {
			if len(centroid) != 5 {
				t.Errorf("Centroid %d has wrong dimension: expected 5, got %d", i, len(centroid))
			}
		}

		// Verify all centroids are unique
		centroidSet := make(map[string]bool)
		for i, centroid := range centroids {
			centroidStr := sliceToString(centroid)
			if centroidSet[centroidStr] {
				t.Errorf("Duplicate centroid found at index %d: %v", i, centroid)
			}
			centroidSet[centroidStr] = true
		}
	})

	t.Run("data with negative coordinates", func(t *testing.T) {
		// Test with data points containing negative coordinates
		data := [][]float64{
			{-1.0, -2.0},
			{-3.0, -4.0},
			{1.0, 2.0},
			{3.0, 4.0},
			{0.0, 0.0},
		}

		kmeans := NewWithOptions(3, WithRandomSeed(42))

		centroids := kmeans.initKMeansPlusPlusCentroids(data)

		if len(centroids) != 3 {
			t.Errorf("Expected 3 centroids, got %d", len(centroids))
		}

		// Verify all centroids are unique
		centroidSet := make(map[string]bool)
		for i, centroid := range centroids {
			centroidStr := sliceToString(centroid)
			if centroidSet[centroidStr] {
				t.Errorf("Duplicate centroid found at index %d: %v", i, centroid)
			}
			centroidSet[centroidStr] = true
		}
	})

	t.Run("data with decimal coordinates", func(t *testing.T) {
		// Test with data points containing decimal coordinates
		data := [][]float64{
			{1.5, 2.7},
			{3.2, 4.8},
			{5.1, 6.3},
			{7.9, 8.4},
			{9.6, 10.2},
		}

		kmeans := NewWithOptions(3, WithRandomSeed(42))

		centroids := kmeans.initKMeansPlusPlusCentroids(data)

		if len(centroids) != 3 {
			t.Errorf("Expected 3 centroids, got %d", len(centroids))
		}

		// Verify all centroids are unique
		centroidSet := make(map[string]bool)
		for i, centroid := range centroids {
			centroidStr := sliceToString(centroid)
			if centroidSet[centroidStr] {
				t.Errorf("Duplicate centroid found at index %d: %v", i, centroid)
			}
			centroidSet[centroidStr] = true
		}
	})

	t.Run("large number of clusters", func(t *testing.T) {
		// Test with a larger number of clusters to stress test the algorithm
		data := [][]float64{
			{1.0, 1.0}, {2.0, 2.0}, {3.0, 3.0}, {4.0, 4.0}, {5.0, 5.0},
			{6.0, 6.0}, {7.0, 7.0}, {8.0, 8.0}, {9.0, 9.0}, {10.0, 10.0},
			{11.0, 11.0}, {12.0, 12.0}, {13.0, 13.0}, {14.0, 14.0}, {15.0, 15.0},
		}

		kmeans := NewWithOptions(8, WithRandomSeed(42))

		centroids := kmeans.initKMeansPlusPlusCentroids(data)

		if len(centroids) != 8 {
			t.Errorf("Expected 8 centroids, got %d", len(centroids))
		}

		// Verify all centroids are unique
		centroidSet := make(map[string]bool)
		for i, centroid := range centroids {
			centroidStr := sliceToString(centroid)
			if centroidSet[centroidStr] {
				t.Errorf("Duplicate centroid found at index %d: %v", i, centroid)
			}
			centroidSet[centroidStr] = true
		}
	})

	t.Run("standard k-means++ mode (nLocalTrials = 1)", func(t *testing.T) {
		data := [][]float64{
			{1.0, 1.0},
			{2.0, 2.0},
			{3.0, 3.0},
			{4.0, 4.0},
			{5.0, 5.0},
		}
		kmeans := NewWithOptions(2, WithRandomSeed(42), WithNCentroidsInitTrials(1))

		centroids := kmeans.initKMeansPlusPlusCentroids(data)
		if len(centroids) != 2 {
			t.Errorf("Expected 2 centroids, got %d", len(centroids))
		}
		centroidSet := make(map[string]bool)
		for i, centroid := range centroids {
			centroidStr := sliceToString(centroid)
			if centroidSet[centroidStr] {
				t.Errorf("Duplicate centroid found at index %d: %v", i, centroid)
			}
			centroidSet[centroidStr] = true
		}
	})
}

func TestCalculateTolerance(t *testing.T) {
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
			require.InDelta(t, tt.want, calculateTolerance(tt.data, tt.relativeTolerance), 1e-12)
		})
	}
}

func TestCalculateTolerance_Invariance(t *testing.T) {
	t.Parallel()

	data := [][]float64{{-2, 1}, {0, 4}, {5, 9}}
	translated := [][]float64{{98, -49}, {100, -46}, {105, -41}}
	scaled := [][]float64{{-20, 10}, {0, 40}, {50, 90}}

	tolerance := calculateTolerance(data, 1e-4)
	require.InDelta(t, tolerance, calculateTolerance(translated, 1e-4), 1e-15)
	require.InDelta(t, tolerance*100, calculateTolerance(scaled, 1e-4), 1e-15)
}

func TestConvergenceScaleInvariance(t *testing.T) {
	t.Parallel()

	data := [][]float64{{0, 0}, {2, 4}}
	oldCenters := [][]float64{{0, 0}}
	newCenters := [][]float64{{0.1, 0.2}}
	scaledData := [][]float64{{0, 0}, {20, 40}}
	scaledOldCenters := [][]float64{{0, 0}}
	scaledNewCenters := [][]float64{{1, 2}}

	converged := checkConvergence(oldCenters, newCenters, calculateTolerance(data, 0.1))
	scaledConverged := checkConvergence(
		scaledOldCenters,
		scaledNewCenters,
		calculateTolerance(scaledData, 0.1),
	)

	require.True(t, converged)
	require.Equal(t, converged, scaledConverged)
}

func TestCheckConvergence(t *testing.T) {
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
			require.Equal(t, tt.want, checkConvergence(tt.oldCenters, tt.newCenters, tt.tolerance))
		})
	}
}

// Unit tests for assignPointsToClusters
func TestAssignPointsToClusters(t *testing.T) {
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
			defer func() {
				recoverVal := recover()
				if tt.wantPanic && recoverVal == nil {
					t.Errorf("expected panic, but did not panic")
				}
				if !tt.wantPanic && recoverVal != nil {
					t.Errorf("unexpected panic: %v", recoverVal)
				}
			}()

			l := make([]int, len(tt.data))

			if !tt.wantPanic {
				got := assignPointsToClusters(tt.data, tt.centers, l)
				if !reflect.DeepEqual(got, tt.want) {
					t.Errorf("assignPointsToClusters(%v, %v) = %v; want %v", tt.data, tt.centers, got, tt.want)
				}
			} else {
				_ = assignPointsToClusters(tt.data, tt.centers, l)
			}
		})
	}
}

// Unit tests for calculateInertiaByLabels
func TestCalculateInertiaByLabels(t *testing.T) {
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
			defer func() {
				recoverVal := recover()
				if tt.wantPanic && recoverVal == nil {
					t.Errorf("expected panic, but did not panic")
				}
				if !tt.wantPanic && recoverVal != nil {
					t.Errorf("unexpected panic: %v", recoverVal)
				}
			}()
			if !tt.wantPanic {
				got := calculateInertiaByLabels(tt.data, tt.centers, tt.labels)
				if math.Abs(got-tt.want) > 1e-9 {
					t.Errorf("calculateInertiaByLabels(%v, %v, %v) = %v; want %v", tt.data, tt.centers, tt.labels, got, tt.want)
				}
			} else {
				_ = calculateInertiaByLabels(tt.data, tt.centers, tt.labels)
			}
		})
	}
}

// Unit tests for updateCentersLloyd
func TestUpdateCentersLloyd(t *testing.T) {
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

			defer func() {
				recoverVal := recover()
				if tt.wantPanic && recoverVal == nil {
					t.Errorf("expected panic, but did not panic")
				}
				if !tt.wantPanic && recoverVal != nil {
					t.Errorf("unexpected panic: %v", recoverVal)
				}
			}()
			if !tt.wantPanic {
				k := &Kmeans{NClusters: tt.nClusters}
				got := k.updateCentersLloyd(tt.data, tt.labels, tt.centers)
				if !reflect.DeepEqual(got, tt.want) {
					t.Errorf("updateCentersLloyd(%v, %v) = %v; want %v", tt.data, tt.labels, got, tt.want)
				}
				require.Equal(t, tt.wantLabels, tt.labels)
			} else {
				k := &Kmeans{NClusters: tt.nClusters}
				_ = k.updateCentersLloyd(tt.data, tt.labels, tt.centers)
			}
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
	require.Equal(t, first, second)
	require.Equal(t, calculateInertiaByLabels(data, first.Centroids, first.Labels), first.Inertia)

	counts := make([]int, 3)
	for pointIndex, label := range first.Labels {
		counts[label]++
		assignedDistance := squaredEuclideanDistance(data[pointIndex], first.Centroids[label])
		for _, centroid := range first.Centroids {
			require.LessOrEqual(t, assignedDistance, squaredEuclideanDistance(data[pointIndex], centroid))
		}
	}
	for _, count := range counts {
		require.Positive(t, count)
	}
	for _, centroid := range first.Centroids {
		require.Equal(t, []float64{100, 200}, centroid)
		require.NotEqual(t, []float64{0, 0}, centroid)
	}
}

func TestCluster_Lloyd_Basic(t *testing.T) {
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
	require.Len(t, result.Centroids, 2)
	require.Len(t, result.Labels, 4)
	require.Positive(t, result.Inertia)

	require.Equal(t, result.Labels[0], result.Labels[1])
	require.Equal(t, result.Labels[2], result.Labels[3])
}

func TestCluster_ConcurrentCallsAreReproducible(t *testing.T) {
	t.Parallel()

	data := [][]float64{
		{0, 0}, {0, 1}, {1, 0},
		{8, 8}, {8, 9}, {9, 8},
		{16, 0}, {16, 1}, {17, 0},
	}
	kmeans := NewWithOptions(3, WithRandomSeed(42), WithNInit(1))

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
			require.Len(t, result.Centroids, 2)
			require.Positive(t, result.Inertia)
		})
	}
}

func TestCluster_Lloyd_SingleDimension(t *testing.T) {
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

func TestCluster_Lloyd_SmallNumbers(t *testing.T) {
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
	require.Len(t, result.Centroids, 3)
	require.Len(t, result.Labels, 30)
}

func TestInitCentroids_Random(t *testing.T) {
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

	centroids := kmeans.initCentroids(data)

	require.Len(t, centroids, 2)
	for _, centroid := range centroids {
		require.Len(t, centroid, 2)
		found := false
		for _, point := range data {
			if reflect.DeepEqual(centroid, point) {
				found = true
				break
			}
		}
		require.True(t, found, "centroid should be from data points")
	}
}

func TestInitCentroids_KMeansPlusPlus(t *testing.T) {
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

	centroids := kmeans.initCentroids(data)

	require.Len(t, centroids, 2)
	for _, centroid := range centroids {
		require.Len(t, centroid, 2)
	}
}

func TestInitCentroids_Default(t *testing.T) {
	t.Parallel()

	kmeans := NewWithOptions(2, WithRandomSeed(42))
	require.Equal(t, InitKMeansPlusPlus, kmeans.Init)

	data := [][]float64{
		{1.0, 2.0},
		{3.0, 4.0},
		{5.0, 6.0},
	}

	centroids := kmeans.initCentroids(data)

	require.Len(t, centroids, 2)
}

func TestClusterSingle_DefaultAlgorithm(t *testing.T) {
	kmeans := NewWithOptions(2, WithRandomSeed(42))

	data := [][]float64{
		{1.0, 2.0},
		{3.0, 4.0},
		{10.0, 11.0},
		{12.0, 13.0},
	}

	result := kmeans.clusterSingle(data)

	require.Len(t, result.Centroids, 2)
}

func BenchmarkLloyd_KMeansPlusPlus(b *testing.B) {
	data := make([][]float64, 1000)
	for i := 0; i < 1000; i++ {
		if i < 500 {
			data[i] = []float64{float64(i % 10), float64(i % 10)}
		} else {
			data[i] = []float64{float64(i%10) + 50, float64(i%10) + 50}
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		kmeans := NewWithOptions(5,
			WithInitMethod(InitKMeansPlusPlus),
			WithMaxIter(100),
		)
		_, _ = kmeans.Cluster(data)
	}
}

func BenchmarkLloyd_RandomInit(b *testing.B) {
	data := make([][]float64, 1000)
	for i := 0; i < 1000; i++ {
		if i < 500 {
			data[i] = []float64{float64(i % 10), float64(i % 10)}
		} else {
			data[i] = []float64{float64(i%10) + 50, float64(i%10) + 50}
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		kmeans := NewWithOptions(5,
			WithInitMethod(InitRandom),
			WithMaxIter(100),
		)
		_, _ = kmeans.Cluster(data)
	}
}

func BenchmarkLloyd_VaryingDataSize(b *testing.B) {
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
			)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, _ = kmeans.Cluster(data)
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

func TestWeightedRandomChoice_NaN(t *testing.T) {
	rng := rand.New(rand.NewSource(42))

	weights := []float64{1.0, math.NaN(), 2.0}

	result := weightedRandomChoice(weights, rng)

	require.True(t, result >= 0 && result < len(weights))
}

func TestWeightedRandomChoice_Inf(t *testing.T) {
	rng := rand.New(rand.NewSource(42))

	weights := []float64{1.0, math.Inf(1), 2.0}

	result := weightedRandomChoice(weights, rng)

	require.True(t, result >= 0 && result < len(weights))
}

func TestWeightedRandomChoice_ZeroTotalWeight(t *testing.T) {
	rng := rand.New(rand.NewSource(42))

	weights := []float64{0.0, 0.0, 0.0}

	result := weightedRandomChoice(weights, rng)

	require.True(t, result >= 0 && result < len(weights))
}
