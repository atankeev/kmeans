package kmeans

import (
	"testing"

	"github.com/stretchr/testify/require"
)

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
