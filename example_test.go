package kmeans_test

import (
	"errors"
	"fmt"

	"github.com/atankeev/kmeans"
)

func ExampleCluster() {
	data := [][]float64{
		{1.0, 2.0},
		{1.5, 1.8},
		{5.0, 8.0},
		{8.0, 8.0},
		{1.0, 0.6},
		{9.0, 11.0},
	}

	result, err := kmeans.Cluster(data, 2, kmeans.WithSeed(42))
	if err != nil {
		fmt.Println("unexpected error")
		return
	}

	// Present groups in first-appearance order, independent of cluster labels.
	groups := make([][]int, 0, 2)
	groupByLabel := make(map[int]int, 2)
	for sample, label := range result.Labels {
		group, ok := groupByLabel[label]
		if !ok {
			group = len(groups)
			groupByLabel[label] = group
			groups = append(groups, nil)
		}
		groups[group] = append(groups[group], sample)
	}
	fmt.Printf("Sample groups: %v\n", groups)
	fmt.Printf("Converged: %t\n", result.Converged)

	// Output:
	// Sample groups: [[0 1 4] [2 3 5]]
	// Converged: true
}

func ExampleCluster_invalidData() {
	data := [][]float64{{1, 2}, {3}}
	_, err := kmeans.Cluster(data, 1, kmeans.WithSeed(42))
	if errors.Is(err, kmeans.ErrInvalidData) {
		fmt.Println("invalid data")
		return
	}
	if err != nil {
		fmt.Println("unexpected error")
		return
	}
	fmt.Println("unexpected success")

	// Output: invalid data
}

func ExampleCluster_nonConverged() {
	data := [][]float64{{0}, {10}}
	result, err := kmeans.Cluster(data, 1,
		kmeans.WithSeed(42),
		kmeans.WithRuns(3),
		kmeans.WithMaxIterations(1),
		kmeans.WithTolerance(1e-12),
	)
	if err != nil {
		fmt.Println("unexpected error")
		return
	}

	// Reaching the iteration limit returns a usable non-converged result.
	fmt.Printf("Converged: %t\n", result.Converged)
	fmt.Printf("Iterations: %d\n", result.Iterations)
	fmt.Printf("Centroid: %g\n", result.Centroids[0][0])

	// Output:
	// Converged: false
	// Iterations: 1
	// Centroid: 5
}
