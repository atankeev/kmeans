package kmeans_test

import (
	"sync"
	"testing"

	"github.com/atankeev/kmeans"
	"github.com/stretchr/testify/require"
)

func TestCluster_ConcurrentCallsAreReproducible(t *testing.T) {
	t.Parallel()
	requireConcurrentCallsAreReproducible(t, 3, [][]float64{{0, 0}, {0, 1}, {1, 0}, {8, 8}, {8, 9}, {9, 8}, {16, 0}, {16, 1}, {17, 0}}, kmeans.WithSeed(42), kmeans.WithRuns(10))
}

func TestCluster_ScaledArithmeticConcurrentCallsAreReproducible(t *testing.T) {
	t.Parallel()
	requireConcurrentCallsAreReproducible(t, 2, [][]float64{{-1e308}, {-1e308}, {1e308}, {1e308}}, kmeans.WithSeed(3), kmeans.WithRuns(1), kmeans.WithMaxIterations(1))
}

func requireConcurrentCallsAreReproducible(t *testing.T, k int, data [][]float64, opts ...kmeans.Option) {
	t.Helper()
	want, err := kmeans.Cluster(data, k, opts...)
	require.NoError(t, err)
	const callCount = 16
	type outcome struct {
		result *kmeans.Result
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
			result, err := kmeans.Cluster(data, k, opts...)
			outcomes <- outcome{result, err}
		}()
	}
	close(start)
	wg.Wait()
	close(outcomes)
	for outcome := range outcomes {
		require.NoError(t, outcome.err)
		require.Equal(t, want, outcome.result)
	}
	again, err := kmeans.Cluster(data, k, opts...)
	require.NoError(t, err)
	require.Equal(t, want, again)
}
