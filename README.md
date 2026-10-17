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

Requires Go 1.24.4 or later.

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
    
    // Cluster the entire dataset into two groups.
    result, err := kmeans.Cluster(data, 2)
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
All samples must have the same number of features, every coordinate must be finite,
and `k` must be between one and the number of samples, inclusive. The API processes
the whole in-memory dataset; fitting a reusable model, prediction, and persistence
are outside its scope.

Use dense numeric vectors with Euclidean geometry: the objective is to minimize
the sum of squared distances to assigned centroids. Choose `k` and prepare feature
scaling before calling `Cluster`; automatic scaling and custom distance metrics
are not provided.

## Advanced Usage

### Custom Configuration

```go
result, err := kmeans.Cluster(data, 3,
    kmeans.WithInitMethod(kmeans.InitKMeansPlusPlus),
    kmeans.WithMaxIterations(100),
    kmeans.WithTolerance(1e-6),
    kmeans.WithSeed(42),
)

if err != nil {
    log.Fatal(err)
}
fmt.Printf("Cluster labels: %v\n", result.Labels)
```

## API Reference

### Types

#### `Cluster`

```go
func Cluster(data [][]float64, k int, opts ...Option) (*Result, error)
```

This stateless function clusters every supplied sample. Configuration and random state
belong to the call; there is no mutable algorithm object or public configuration API.
The library does not accept a context or support interruption of a running call.

#### `Result`
Clustering result for the selected initialization run. For `n` samples with `d`
features, `len(Labels) == n`, and labels are zero-based centroid indices in `[0, k)`.
`Centroids` has `k` rows with `d` coordinates each. `Centroids[Labels[i]]` is the
centroid assigned to `data[i]`. `Converged` and `Iterations`
describe only that run, rather than totals across runs. Iterations count Lloyd steps,
including the step establishing convergence, but exclude initialization and final label
reassignment. Convergence on the final allowed iteration remains convergence. Tolerance
convergence does not guarantee centroids are exact means of the final label groups.

`Inertia` is the sum of squared distances from samples to their assigned centroids.
Compare it for the same data, feature scaling, and `k`: lower inertia means a lower
value of this objective, not universally better clustering. A decrease when increasing
`k` does not by itself establish a better partition. A very small positive inertia
may round to zero; reported zero does not prove exact coincidence of samples and centroids.

```go
type Result struct {
    Labels     []int
    Centroids  [][]float64
    Inertia    float64
    Converged  bool
    Iterations int
}
```

### Configuration Options

| Option | Valid values | Default |
|--------|--------------|---------|
| `WithInitMethod(method)` | `InitKMeansPlusPlus` or `InitRandom` | `InitKMeansPlusPlus` |
| `WithMaxIterations(iterations)` | Positive integer | `DefaultMaxIterations = 300` |
| `WithTolerance(tolerance)` | Positive, finite float | `DefaultTolerance = 1e-4` |
| `WithSeed(seed)` | Any `int64`, including zero | Selected internally for each call |
| `WithRuns(runs)` | Positive integer | `DefaultRuns = 10` |
| `WithNCentroidsInitTrials(trials)` | Positive integer, even with `InitRandom` | `2 + floor(ln(k))`, resolved per call |

Only library-provided options are accepted. Nil entries and external types embedding
`Option`, including typed-nil pointers and wrappers around valid options, are rejected
with `ErrInvalidOption`. Built-in option values capture their settings and may be reused
across calls. Options apply in order, and the last supplied value for a setting wins.
Validation checks final settings: an invalid value followed by a valid replacement is
accepted; a final invalid value is an error and never silently becomes a default.
Unsupported or nil entries remain errors regardless of later entries.

Increasing runs can improve the selected result at the cost of repeating clustering.
Increasing k-means++ candidate trials can improve initialization at additional cost per
centroid; one trial performs the standard weighted choice. Increasing the iteration limit
allows more refinement, while tolerance controls the stopping threshold.

### Algorithm

#### Lloyd's Algorithm
- **Description**: Standard K-means algorithm
- **Empty clusters**: Relocated to distinct samples with the largest current assignment error; ties use sample order
- **Numerical safety**: Avoidable overflow in means, variance, distances, initialization weights, convergence, and inertia is handled with stable or scaled arithmetic. Successful results contain only finite centroids and inertia. If a required result exceeds the finite representable range or a calculation cannot be completed safely, `Cluster` returns a nil result with `ErrNumericalOverflow`. A positive inertia too small to represent may round to zero in `Result.Inertia` without an error; public zero does not prove exact coincidence of samples and centroids. Scale invariance is not guaranteed for arbitrary inputs or bit-for-bit: scaling can lose differences in input coordinates that the algorithm cannot recover.
- **Stopping behavior**: For each `Cluster` call, the effective tolerance is `tolerance * mean(var(data, axis=0))`, using population variance. Each initialization stops when assignments are unchanged between consecutive iterations or when the squared Frobenius norm `sum((newCenters - oldCenters)^2)` is at most the effective tolerance. A zero-variance dataset therefore requires zero center movement unless assignments are unchanged. Runs are ranked by internally computed inertia before conversion to public `float64`, so distinct positive inertias that both round to zero still retain their internal ordering. Equal internal inertias retain the first run, with no approximate-equality tolerance or preference for convergence. The selected run supplies its labels, centroids, public inertia, and convergence status together. If that run reaches its iteration limit without convergence, `Cluster` returns its final result with `Converged == false` and a nil error.
- **Best for**: General purpose clustering
- **Time Complexity**: `O(r * n * k * i * d)` for the main assignment/update computations across initialization runs, where `r` is runs, `n` samples, `k` clusters, `i` iterations per run (bounded by `WithMaxIterations`), and `d` features. Initialization adds work, including k-means++ candidate trials. Empty-cluster recovery adds distance computation and candidate sorting overhead. This formula is not a complete worst-case bound for `Cluster`.

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

result, err := kmeans.Cluster(data, 2)
if err != nil {
    log.Fatal(err)
}
fmt.Printf("Cluster labels: %v\n", result.Labels)
```

### High-Dimensional Data

```go
// 3D data
data := [][]float64{
    {1, 2, 3}, {1.5, 1.8, 2.5}, {5, 8, 9}, {8, 8, 7},
}

result, err := kmeans.Cluster(data, 2,
    kmeans.WithMaxIterations(200),
)
if err != nil {
    log.Fatal(err)
}
fmt.Printf("Cluster labels: %v\n", result.Labels)
```

### Reproducible Results

```go
result, err := kmeans.Cluster(data, 3,
    kmeans.WithSeed(42),
    kmeans.WithRuns(10), // Multiple initialization runs are reproducible too
)
if err != nil {
    log.Fatal(err)
}
fmt.Printf("Cluster labels: %v\n", result.Labels)
```

Within the same library version and execution environment, identical ordered data,
explicit seed, and options produce the same complete result on every call, including
concurrent calls and multiple runs. `WithSeed(0)` is deterministic. Without `WithSeed`,
the seed is chosen internally on each call and is not exposed; different seeds need not
produce different clustering results. Cross-version or cross-platform reproducibility
is not guaranteed.

### Concurrency and Ownership

`Cluster` calls are independent, and the same built-in options can be reused concurrently.
Each call has independent random state, so seeded results do not depend on scheduling.
Treat input slices as read-only during a call. The library does not modify input and
returns independent labels and centroid storage, including separate centroid rows.
Mutating one result does not affect input or another result.

## Parameter Selection

1. **`WithRuns`**: More initialization runs can find lower inertia by exploring more starting points, at the cost of repeating initialization and Lloyd iterations. Fewer runs reduce this work.
2. **`WithMaxIterations`**: Raise the limit to allow more refinement when a selected run has not converged. A lower limit caps iteration work sooner and may return a usable non-converged result; a higher limit adds no iterations to a run that already converges.
3. **`WithTolerance`**: A smaller value makes the center-movement stopping threshold stricter and may require more iterations. A larger value permits earlier stopping and less refinement. Tolerance is not a guaranteed accuracy target; unchanged assignments can also stop a run.
4. **`WithInitMethod`**: Start with the default `InitKMeansPlusPlus`, which spends more work evaluating distances during initialization to seek useful starting centers. `InitRandom` skips that distance-based selection, but may need more refinement or runs to find a comparable result. Measure total runtime on your data.
5. **`WithNCentroidsInitTrials`**: More trials evaluate more candidates for each k-means++ centroid after the first, trading initialization work for potentially better starting centers. One trial uses the standard weighted choice. This setting has no effect with `InitRandom`, but must still be positive.

Multiple initialization runs and k-means++ do not guarantee a globally optimal result.
Use `WithSeed` for reproducibility; a fixed seed is not a performance improvement.

## Error Handling

Invalid input, invalid options, and numerical overflow always return a nil result and
an error. Reaching an iteration limit is a successful, usable result with
`Converged == false`; inspect metadata after checking the error:

```go
result, err := kmeans.Cluster(data, 2, kmeans.WithSeed(42))
if err != nil {
    if errors.Is(err, kmeans.ErrInvalidData) {
        log.Print("samples must have consistent dimensions")
    }
    log.Fatal(err)
}
if !result.Converged {
    log.Printf("selected run reached its limit after %d iterations", result.Iterations)
}
fmt.Printf("Cluster labels: %v\n", result.Labels)
```

Use `errors.Is` to identify the following sentinels. Wrapped diagnostics, including
sample and dimension details, are useful but their exact wording is not a stable contract.

| Sentinel | Cause |
|----------|-------|
| `ErrInvalidOption` | Nil, wrapped, or unsupported option |
| `ErrInvalidRuns` | Non-positive run count |
| `ErrInvalidMaxIterations` | Non-positive iteration limit |
| `ErrInvalidTolerance` | Non-positive, NaN, or infinite tolerance |
| `ErrInvalidNCentroidsInitTrials` | Non-positive candidate-trial count |
| `ErrInvalidInitMethod` | Unsupported initialization method |
| `ErrInvalidK` | Cluster count outside `1 <= k <= len(data)` |
| `ErrEmptyData` | No samples |
| `ErrNoFeatures` | Samples with no features |
| `ErrInvalidData` | Inconsistent sample dimensions |
| `ErrNonFiniteData` | NaN or infinite coordinate |
| `ErrNumericalOverflow` | Required arithmetic cannot be represented or processed safely |

Numerical overflow in any initialization run aborts the call, even after an earlier
usable result. Lower internal inertia always wins, including a non-converged run over
a higher-inertia converged run.

## Migration from the Object API

This release intentionally replaces the API in place without changing the module path
or providing legacy aliases. The maintainer approved the compatibility deviation before
adoption because the library has no consumers.

Replace `New(k).Cluster(data)` or `NewWithOptions(k, opts...).Cluster(data)` with
`Cluster(data, k, opts...)`. `Kmeans` and public `Validate` are removed; provide built-in
options rather than a struct literal or mutable fields, and handle validation errors
from `Cluster`. Constructor provenance and `ErrConfigNotInitialized` are removed.

| Removed name | Replacement |
|--------------|-------------|
| `WithRandomSeed` | `WithSeed` |
| `WithNInit` | `WithRuns` |
| `WithMaxIter` | `WithMaxIterations` |
| `WithTol` | `WithTolerance` |
| `DefaultNInit` | `DefaultRuns` |
| `DefaultMaxIter` | `DefaultMaxIterations` |
| `DefaultTol` | `DefaultTolerance` |
| `ErrInvalidNInit` | `ErrInvalidRuns` |
| `ErrInvalidMaxIter` | `ErrInvalidMaxIterations` |
| `ErrInvalidTol` | `ErrInvalidTolerance` |

`WithInitMethod`, `WithNCentroidsInitTrials`, and initialization constants retain their
names. Caller-defined option functions and option wrappers are no longer supported.
Replace handling of removed `ErrConvergenceFailed` with `Result.Converged` and
`Result.Iterations`; usable non-convergence now has a nil error. Ragged data now matches
`ErrInvalidData` and still reports the offending sample and expected dimensions.

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
