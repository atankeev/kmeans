# K-Means Clustering Library for Go

A K-means clustering library for Go implementing Lloyd's algorithm.

> **Note**: This library was developed with the assistance of AI assistants.

## Features

- **Clustering algorithm**: Lloyd's (standard)
- **Multiple initialization methods**: Random and K-means++
- **Configurable parameters**: Number of clusters, iterations, tolerance, etc.
- **Reproducible results**: Fixed random seed support
- **Empty-cluster recovery**: Deterministic relocation to farthest assigned samples
- **Comprehensive error handling**: Detailed validation and error messages

## Installation

```bash
go get github.com/atankeev/kmeans
```

## Quick Start

```go
package main

import (
    "fmt"
    "log"
    
    "github.com/atankeev/kmeans"
)

func main() {
    // Sample data
    data := [][]float64{
        {1.0, 2.0},
        {1.5, 1.8},
        {5.0, 8.0},
        {8.0, 8.0},
        {1.0, 0.6},
        {9.0, 11.0},
    }
    
    // Create K-means instance with 2 clusters
    kmeans := kmeans.New(2)
    
    // Perform clustering
    result, err := kmeans.Cluster(data)
    if err != nil {
        log.Fatal(err)
    }
    
    // Print results
    fmt.Printf("Cluster labels: %v\n", result.Labels)
    fmt.Printf("Centroids: %v\n", result.Centroids)
    fmt.Printf("Inertia: %.2f\n", result.Inertia)
}
```

`Cluster` requires a dataset with at least one sample and at least one feature per sample.
All samples must have the same number of features.

## Advanced Usage

### Custom Configuration

```go
// Create K-means with custom options
kmeans := kmeans.NewWithOptions(3,
    kmeans.WithInitMethod(kmeans.InitKMeansPlusPlus),
    kmeans.WithMaxIter(100),
    kmeans.WithTol(1e-6),
    kmeans.WithRandomSeed(42),
)

result, err := kmeans.Cluster(data)
```

### Using Different Initialization Methods

## API Reference

### Types

#### `Kmeans`
Main clustering struct with configurable parameters.

```go
type Kmeans struct {
    NClusters            int           // Number of clusters
    NInit                int           // Number of initializations
    MaxIter              int           // Maximum iterations
    Tol                  float64       // Convergence tolerance
    Init                 InitMethod    // Initialization method
    RandomSeed           int64         // Random seed used by each call
    NCentroidsInitTrials int           // K-means++ trials
}
```

#### `Result`
Clustering result containing labels, centroids, and inertia.

```go
type Result struct {
    Labels    []int        // Cluster labels for each point
    Centroids [][]float64  // Cluster centroids
    Inertia   float64      // Sum of squared distances
}
```

### Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `WithInitMethod(method)` | Set initialization method | `InitKMeansPlusPlus` |
| `WithMaxIter(iterations)` | Set maximum iterations | `300` |
| `WithTol(tolerance)` | Set relative convergence tolerance | `1e-4` |
| `WithRandomSeed(seed)` | Set random seed | Random |
| `WithNInit(nInit)` | Set number of initializations | `10` |
| `WithNCentroidsInitTrials(trials)` | Set number of k-means++ initialization trials | `2 + log(nClusters)` |

Configuration options always retain the value supplied by the caller. Call `Validate()` to check
configuration eagerly, or handle the validation error returned by `Cluster()`. Counts must be
positive, and the convergence tolerance must be positive and finite.

### Algorithm

#### Lloyd's Algorithm
- **Description**: Standard K-means algorithm
- **Empty clusters**: Relocated to distinct samples with the largest current assignment error; ties use sample order
- **Numerical safety**: Avoidable overflow in means, variance, distances, initialization weights, convergence, and inertia is handled with stable or scaled arithmetic. Successful results contain only finite centroids and inertia. If a required result exceeds the finite representable range or a calculation cannot be completed safely, `Cluster` returns a nil result with `ErrNumericalOverflow`. A positive inertia too small to represent may round to zero in `Result.Inertia` without an error; public zero does not prove exact coincidence of samples and centroids. Scale invariance is not guaranteed for arbitrary inputs or bit-for-bit: scaling can lose differences in input coordinates that the algorithm cannot recover.
- **Stopping behavior**: For each `Cluster` call, the effective tolerance is `Tol * mean(var(data, axis=0))`, using population variance. Each initialization stops when assignments are unchanged between consecutive iterations or when the squared Frobenius norm `sum((newCenters - oldCenters)^2)` is at most the effective tolerance. A zero-variance dataset therefore requires zero center movement unless assignments are unchanged. Runs are ranked by internally computed inertia before conversion to public `float64`, so distinct positive inertias that both round to zero still retain their internal ordering. Equal internal inertias retain the first run, with no approximate-equality tolerance or preference for convergence. The selected run supplies its labels, centroids, public inertia, and convergence status together. If that run reaches `MaxIter` without convergence, `Cluster` returns its final result together with `ErrConvergenceFailed`.
- **Best for**: General purpose clustering
- **Time Complexity**: O(n*k*i*d) where n=points, k=clusters, i=iterations, d=dimensions

### Initialization Methods

#### Random Initialization
- **Type**: `InitRandom`
- **Description**: Randomly select initial centroids
- **Best for**: Quick testing, small datasets

#### K-means++ Initialization
- **Type**: `InitKMeansPlusPlus`
- **Description**: Smart initialization for better convergence
- **Best for**: Production use, better clustering quality

## Examples

### Basic Clustering

```go
data := [][]float64{
    {1, 2}, {1.5, 1.8}, {5, 8}, {8, 8}, {1, 0.6}, {9, 11},
}

kmeans := kmeans.New(2)
result, err := kmeans.Cluster(data)
```

### High-Dimensional Data

```go
// 3D data
data := [][]float64{
    {1, 2, 3}, {1.5, 1.8, 2.5}, {5, 8, 9}, {8, 8, 7},
}

kmeans := kmeans.NewWithOptions(2,
    kmeans.WithMaxIter(200),
)
result, err := kmeans.Cluster(data)
```

### Reproducible Results

```go
kmeans := kmeans.NewWithOptions(3,
    kmeans.WithRandomSeed(42),
    kmeans.WithNInit(1), // Single run for reproducibility
)
result, err := kmeans.Cluster(data)
```

Using the same seed, options, and data produces the same result on every call, including
concurrent calls on the same `Kmeans` instance.

### Concurrency

`Cluster` can be called concurrently on the same `Kmeans` instance. Each call creates an
independent random state from `RandomSeed`, so clustering work is not serialized and seeded
results do not depend on goroutine scheduling.

Treat the instance configuration and input data as read-only while `Cluster` is running.
Concurrent mutation of exported configuration fields or input slices is not supported.

## Performance Tips

1. **Use K-means++ initialization** for better clustering quality
2. **Adjust tolerance** based on your precision requirements
3. **Use multiple initializations** (`NInit > 1`) for better results
4. **Set random seed** for reproducible results

## Error Handling

The library provides comprehensive error handling:

```go
result, err := kmeans.Cluster(data)
if errors.Is(err, kmeans.ErrConvergenceFailed) {
    // The final labels, centroids, and inertia are still available as an approximation.
    log.Printf("Clustering reached MaxIter; using result with inertia %.2f", result.Inertia)
} else if err != nil {
    switch {
    case errors.Is(err, kmeans.ErrEmptyData):
        log.Fatal("Empty dataset provided")
    case errors.Is(err, kmeans.ErrNoFeatures):
        log.Fatal("Dataset samples must contain at least one feature")
    case errors.Is(err, kmeans.ErrInvalidK):
        log.Fatal("Invalid number of clusters")
    case errors.Is(err, kmeans.ErrInvalidTol):
        log.Fatal("Tolerance must be positive and finite")
    case errors.Is(err, kmeans.ErrNonFiniteData):
        log.Fatal("Dataset contains a NaN or infinite coordinate")
    case errors.Is(err, kmeans.ErrNumericalOverflow):
        // Numerical overflow always returns a nil result.
        log.Fatal("A required clustering result cannot be represented safely")
    default:
        log.Fatal("Clustering failed:", err)
    }
}
```

With multiple initializations, convergence status describes only the selected lowest-inertia run. A lower-inertia non-converged run takes precedence over a converged run with higher inertia. Numerical overflow in any run aborts the call with a nil result, even if an earlier run was usable.
Other validation and configuration errors return a nil result.

## Testing

Run the test suite:

```bash
go test -v ./...
```

Run specific tests:

```bash
go test -v -run TestLloydKMeans
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Changelog

### v1.0.0
- Initial release
- Lloyd's algorithm
- Random and K-means++ initialization
- Comprehensive test coverage
- Full API documentation
