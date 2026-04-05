# K-Means Clustering Library for Go

A K-means clustering library for Go with support for Lloyd's and Elkan's algorithms.

> **Note**: This library was developed with the assistance of AI assistants.

## Features

- **Two clustering algorithms**: Lloyd's (standard) and Elkan's (optimized)
- **Multiple initialization methods**: Random and K-means++
- **Configurable parameters**: Number of clusters, iterations, tolerance, etc.
- **Reproducible results**: Fixed random seed support
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

## Advanced Usage

### Custom Configuration

```go
// Create K-means with custom options
kmeans := kmeans.NewWithOptions(3,
    kmeans.WithAlgorithm(kmeans.AlgorithmElkan),
    kmeans.WithInitMethod(kmeans.InitKMeansPlusPlus),
    kmeans.WithMaxIter(100),
    kmeans.WithTol(1e-6),
    kmeans.WithRandomSeed(42),
)

result, err := kmeans.Cluster(data)
```

### Using Different Algorithms

```go
// Lloyd's algorithm (default)
kmeansLloyd := kmeans.NewWithOptions(2,
    kmeans.WithAlgorithm(kmeans.AlgorithmLloyd),
)

// Elkan's algorithm (faster for large datasets)
kmeansElkan := kmeans.NewWithOptions(2,
    kmeans.WithAlgorithm(kmeans.AlgorithmElkan),
)
```

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
    RandomState          *rand.Rand    // Random state
    NCentroidsInitTrials int           // K-means++ trials
    Algorithm            Algorithm     // Clustering algorithm
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
| `WithAlgorithm(algorithm)` | Set clustering algorithm | `AlgorithmLloyd` |
| `WithInitMethod(method)` | Set initialization method | `InitKMeansPlusPlus` |
| `WithMaxIter(iterations)` | Set maximum iterations | `300` |
| `WithTol(tolerance)` | Set convergence tolerance | `1e-4` |
| `WithRandomSeed(seed)` | Set random seed | Random |
| `WithNInit(nInit)` | Set number of initializations | `10` |

### Algorithms

#### Lloyd's Algorithm
- **Type**: `AlgorithmLloyd`
- **Description**: Standard K-means algorithm
- **Best for**: General purpose clustering
- **Time Complexity**: O(n*k*i*d) where n=points, k=clusters, i=iterations, d=dimensions

#### Elkan's Algorithm
- **Type**: `AlgorithmElkan`
- **Description**: Optimized K-means with triangle inequality
- **Best for**: Large datasets with many clusters
- **Time Complexity**: O(n*k*i*d) with optimizations

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
    kmeans.WithAlgorithm(kmeans.AlgorithmElkan),
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

## Performance Tips

1. **Use Elkan's algorithm** for large datasets with many clusters
2. **Use K-means++ initialization** for better clustering quality
3. **Adjust tolerance** based on your precision requirements
4. **Use multiple initializations** (`NInit > 1`) for better results
5. **Set random seed** for reproducible results

## Error Handling

The library provides comprehensive error handling:

```go
result, err := kmeans.Cluster(data)
if err != nil {
    switch {
    case errors.Is(err, kmeans.ErrEmptyData):
        log.Fatal("Empty dataset provided")
    case errors.Is(err, kmeans.ErrInvalidK):
        log.Fatal("Invalid number of clusters")
    default:
        log.Fatal("Clustering failed:", err)
    }
}
```

## Testing

Run the test suite:

```bash
go test -v ./...
```

Run specific tests:

```bash
go test -v -run TestLloydKMeans
go test -v -run TestElkanKMeans
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Changelog

### v1.0.0
- Initial release
- Lloyd's and Elkan's algorithms
- Random and K-means++ initialization
- Comprehensive test coverage
- Full API documentation