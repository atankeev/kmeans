package kmeans_test

import (
	"testing"

	"github.com/atankeev/kmeans"
)

func TestCodecovProbeSingleton(t *testing.T) {
	t.Parallel()

	result, err := kmeans.Cluster([][]float64{{3}}, 1, kmeans.WithSeed(0))
	if err != nil {
		t.Fatalf("Cluster() error = %v", err)
	}
	if len(result.Labels) != 1 || result.Labels[0] != 0 {
		t.Fatalf("Cluster() labels = %v, want [0]", result.Labels)
	}
	if len(result.Centroids) != 1 || len(result.Centroids[0]) != 1 || result.Centroids[0][0] != 3 {
		t.Fatalf("Cluster() centroids = %v, want [[3]]", result.Centroids)
	}
	if result.Inertia != 0 {
		t.Fatalf("Cluster() inertia = %v, want 0", result.Inertia)
	}
}
