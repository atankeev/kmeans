# K-Means Clustering Library for Go

A K-means clustering library for Go implementing Lloyd's algorithm.

> **Note**: This library was developed with the assistance of AI assistants.

## Features

- **Clustering algorithm**: Lloyd's (standard)
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
    RandomState          *rand.Rand    // Random state
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
| `WithTol(tolerance)` | Set convergence tolerance | `1e-4` |
| `WithRandomSeed(seed)` | Set random seed | Random |
| `WithNInit(nInit)` | Set number of initializations | `10` |

### Algorithm

#### Lloyd's Algorithm
- **Description**: Standard K-means algorithm
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

## Performance Tips

1. **Use K-means++ initialization** for better clustering quality
2. **Adjust tolerance** based on your precision requirements
3. **Use multiple initializations** (`NInit > 1`) for better results
4. **Set random seed** for reproducible results

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