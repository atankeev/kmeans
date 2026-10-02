package kmeans

import (
	"sync"
	"testing"

	"github.com/stretchr/testify/require"
)

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
