# K-Means Clustering Library for Go

A Go library for clustering dense numeric vectors with Lloyd's k-means algorithm.
Call `Cluster(data, k, opts...)` to cluster a whole in-memory dataset and obtain
labels, centroids, inertia, and convergence metadata.

The objective is to minimize the sum of squared Euclidean distances to assigned
centroids. Choose `k` and prepare feature scaling before calling `Cluster`;
automatic scaling and custom distance metrics are not provided. Fitting a reusable
model, prediction, and persistence are outside its scope. Calls do not accept a
context or support interruption.

## Contents

- [Installation](#installation)
- [Quick Start and results](#quick-start)
- [Configuration](#configuration)
- [Error handling](#error-handling)
- [Concurrency and ownership](#concurrency-and-ownership)
- [Algorithm](#algorithm)
- [API documentation](#api-documentation)
- [Contributing](#contributing)
- [License](#license)

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

	fmt.Printf("Cluster labels: %v\n", result.Labels)
	fmt.Printf("Centroids: %v\n", result.Centroids)
	fmt.Printf("Inertia: %.2f\n", result.Inertia)
	fmt.Printf("Converged: %t after %d iterations\n", result.Converged, result.Iterations)
}
```

Input must contain at least one sample and at least one feature per sample. All
samples must have the same number of features, every coordinate must be finite,
and `k` must satisfy `1 <= k <= len(data)`.

### Interpreting the result

For `n` samples with `d` features, `Result.Labels` contains `n` zero-based cluster
indices in `[0, k)`, and `Result.Centroids` has `k` rows of `d` coordinates.
`result.Centroids[result.Labels[i]]` is the centroid assigned to `data[i]`.

`Inertia` is the sum of squared distances to assigned centroids. Compare it for
the same data, feature scaling, and `k`: a lower value improves this objective,
but does not universally mean better clustering. A decrease when increasing `k`
does not by itself establish a better partition. A very small positive inertia
may round to zero; zero does not prove exact coincidence of samples and centroids.

`Converged` and `Iterations` describe only the selected initialization run, not
totals across runs. Iterations count Lloyd steps, including the step establishing
convergence, but exclude initialization and final label reassignment. Convergence
on the final allowed iteration remains convergence. Tolerance convergence does
not guarantee centroids are exact means of the final label groups.

A nil error with `Converged == false` is a usable non-converged result: the selected
run reached its iteration limit. See [stopping](#stopping) and
[result selection](#result-selection) for the precise rules.

## Configuration

To customize the Quick Start, keep its `data` declaration and imports (`fmt`,
`log`, and `github.com/atankeev/kmeans`), then replace the call and everything
after it inside `main` with this fragment:

```go
result, err := kmeans.Cluster(data, 2,
    kmeans.WithInitMethod(kmeans.InitKMeansPlusPlus),
    kmeans.WithRuns(10),
    kmeans.WithMaxIterations(100),
    kmeans.WithTolerance(1e-6),
    kmeans.WithSeed(42),
)
if err != nil {
    log.Fatal(err)
}
fmt.Printf("Cluster labels: %v\n", result.Labels)
```

### Options and validation

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

### Choosing parameters

- **Initialization:** Start with the default `InitKMeansPlusPlus`. It selects the
  first centroid randomly, then samples candidates weighted by squared distance
  to the nearest existing centroid. With multiple candidate trials, it chooses
  the candidate yielding the lowest inertia. `InitRandom` copies samples at
  distinct randomly selected indexes without this distance-based selection; it
  may need more refinement or runs for comparable results. Measure total runtime
  on your data.
- **Runs:** More `WithRuns` initialization runs explore more starting points and
  can find lower inertia, at the cost of repeating initialization and Lloyd
  iterations. Fewer runs reduce this work.
- **Iterations:** Raise `WithMaxIterations` to allow more refinement when the
  selected run has not converged. A lower limit caps work sooner and may return
  a usable non-converged result; a higher limit adds no iterations to a run that
  already converges.
- **Tolerance:** A smaller `WithTolerance` makes the center-movement stopping
  threshold stricter and may require more iterations. A larger value permits
  earlier stopping and less refinement. It is not a guaranteed accuracy target;
  unchanged assignments can also stop a run.
- **Candidate trials:** More `WithNCentroidsInitTrials` trials evaluate more
  candidates for each k-means++ centroid after the first, trading initialization
  work for potentially better starting centers. One trial uses the standard
  weighted choice. This setting has no effect with `InitRandom`, but must still
  be positive.

Multiple initialization runs and k-means++ do not guarantee a globally optimal
result. A fixed seed provides reproducibility, not a performance improvement.

### Reproducibility

Within the same library version and execution environment, identical ordered
data, explicit seed, and options produce the same complete result on every call,
including concurrent calls and multiple initialization runs. `WithSeed(0)` is
deterministic. Without `WithSeed`, the seed is chosen internally on each call and
is not exposed; different seeds need not produce different clustering results.
Cross-version or cross-platform reproducibility is not guaranteed.

## Error Handling

Invalid input, invalid options, and numerical overflow always return a nil result
and an error. Reaching an iteration limit can return a successful result with
`Converged == false`; inspect metadata after checking the error.

For this fragment, keep the Quick Start's `data` declaration and imports, add
`"errors"` to its import block, and replace the call and everything after it
inside `main`:

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
sample and dimension details, are useful but their exact wording is not a stable
contract.

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

Numerical overflow in any initialization run aborts the entire call, even if an
earlier run produced a usable result.

## Concurrency and Ownership

`Cluster` calls are independent: each call owns its configuration and random
state. The same built-in options can be reused concurrently, and seeded results
do not depend on scheduling.

Treat input slices as read-only during a call. The library does not modify input
and returns independent labels and centroid storage, including separate centroid
rows. Mutating one result does not affect input or another result.

## Algorithm

### Initialization and Lloyd steps

Each initialization run starts with centroids chosen by the configured
[initialization method](#choosing-parameters). Lloyd steps assign samples to their
nearest centroids and update centroids to the means of their assigned samples.

Empty clusters take distinct samples with the largest current assignment error,
with ties resolved by sample order. A source cluster must retain at least one
sample. Labels are reassigned to the final centroids after the last update.

### Stopping

For each call, the effective tolerance is
`tolerance * mean(var(data, axis=0))`, using population variance. An initialization
run stops when either:

- Assignments are unchanged between consecutive iterations.
- The squared Frobenius norm of center movement,
  `sum((newCenters - oldCenters)^2)`, is at most the effective tolerance.

A zero-variance dataset therefore requires zero center movement unless
assignments are unchanged. If neither criterion is met by the iteration
limit, the run finishes with `Converged == false` and a nil error.

### Result selection

Runs are ranked by internally computed inertia before conversion to public
`float64`. Distinct positive inertias that both round to zero still retain their
internal ordering. Equal internal inertias retain the first run, with no
approximate-equality tolerance or preference for convergence.

The selected run supplies its labels, centroids, public inertia, and convergence
metadata together. A lower-inertia non-converged run wins over a higher-inertia
converged run.

### Numerical behavior

Stable or scaled arithmetic handles known avoidable overflow paths in means,
variance, distances, initialization weights, convergence, and inertia. Successful
results contain only finite centroids and inertia. An intermediate squared
distance outside the finite `float64` range is not itself an error when scaled
arithmetic can still make a safe decision.

If an obligatory result exceeds the finite representable range or a calculation
cannot be completed safely, the call returns a nil result with
`ErrNumericalOverflow`. In particular, final inertia above the finite `float64`
range is an error. A positive inertia too small to represent may round to zero
without an error.

Scale invariance is not guaranteed for arbitrary inputs or bit-for-bit: scaling
can lose differences in input coordinates that the algorithm cannot recover.
See the [finite arithmetic policy](docs/adr/0001-finite-arithmetic-policy.md) for
the rationale and limits.

### Computational cost

The main assignment and update computations take `O(r * n * k * i * d)` across
initialization runs, where `r` is runs, `n` samples, `k` clusters, `i` iterations
per run (bounded by `WithMaxIterations`), and `d` features. Initialization adds
work, including k-means++ candidate trials. Empty-cluster recovery adds distance
computation and candidate sorting overhead. This is not a complete worst-case
bound for `Cluster`.

## API Documentation

`Cluster(data [][]float64, k int, opts ...Option) (*Result, error)` is the public
entry point. [Cluster and Result](kmeans.go), [option factories and
defaults](options.go), and [error sentinels](errors.go) have GoDoc comments in the
local source.

From a checkout of this repository, browse the complete exported API or an
individual symbol without hosted documentation:

```bash
go doc -all .
go doc . Cluster
go doc . Result
go doc . WithTolerance
```

## Contributing

Contributions are welcome via GitHub issues and pull requests. Run the test suite
from the repository root:

```bash
go test ./...
```

Install the pinned lint tool explicitly, then run the same checks as CI:

```bash
make tools
make lint
```

Lint uses **golangci-lint v2.15.0** and **Go 1.27.2**, with version pins maintained
in the Makefile. The library's minimum supported Go version remains **1.24.4**;
the `build` CI job builds and tests on that minimum. Lint commands select their
Go toolchain through command-scoped `GOTOOLCHAIN`; Go may download it on first
use. This does not change the Go version used by other projects or ordinary Go
commands.

`make tools` downloads the official release archive, verifies its SHA-256
checksum, and installs the binary in the ignored `.bin/` directory. Supported
platforms are Linux and macOS on amd64 and arm64; use WSL on Windows. Installation
requires network access, Go, Make, a POSIX shell, `curl`, `tar`, `mktemp`, `awk`,
and either `sha256sum` or `shasum`.
`make lint` does not install or upgrade the linter. If it reports a missing or
mismatched version, run `make tools` again.

Automatic Go toolchain downloads require checksum verification. If your environment
sets `GOSUMDB=off`, use `GOSUMDB=sum.golang.org make lint` (or `make fmt`). If a
configured `GOPROXY` cannot serve toolchains or checksum lookups, select a proxy
that supports them. The Make targets do not override `GOPROXY` or `GOSUMDB`.

Both library code and tests are checked with an explicit set of linters:

| Linter | Purpose |
| --- | --- |
| `errcheck` | Find unchecked errors. |
| `govet` | Detect suspicious constructs using Go's vet analyzers. |
| `ineffassign` | Find assignments whose values are never used. |
| `staticcheck` | Detect bugs and incorrect use of Go APIs. |
| `unused` | Find unused declarations. |
| `paralleltest` | Check parallel execution of ordinary tests where safe. |
| `testifylint` | Check correct use of testify assertions. |

`make lint` also checks goimports formatting without modifying files. Apply
formatting locally with:

```bash
make fmt
```

Preserve assertions that test exact numerical guarantees. Any necessary linter
suppression must target the specific diagnostic and explain its reason; do not
blanket-exclude tests. Run `go test -race ./...` for concurrency-related changes.
The dedicated CI check is named `lint`; it runs on pull requests and pushes to
`main` with a ten-minute timeout. Its vet and formatting checks also cover those
requirements for the broader CI work in issue #31.

This library was developed with the assistance of AI assistants.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
