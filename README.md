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
}
```

Output:

```text
Sample groups: [[0 1 4] [2 3 5]]
Converged: true
```

The groups contain zero-based sample indices, ordered by their first appearance
in the input, so the display does not depend on numeric cluster labels. This
scenario and output are checked by `ExampleCluster` in [example_test.go](example_test.go).

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
run reached its iteration limit. `ExampleCluster_nonConverged` in
[example_test.go](example_test.go) demonstrates this with two samples, `0` and `10`,
one cluster, three initialization runs, and a one-iteration limit: the result has
`Converged: false`, `Iterations: 1`, and centroid `5`, with no error.
See [stopping](#stopping) and
[result selection](#result-selection) for the precise rules.

## Configuration

To customize the Quick Start, keep its `data` declaration and imports, add `"log"`
to its import block, then replace the call and everything
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

For this fragment, keep the Quick Start's imports, add `"errors"` to its import
block, and replace the body of `main`. Like `ExampleCluster_invalidData`, it
deliberately passes samples with different dimensions:

```go
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
```

Output:

```text
invalid data
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

Contributions are welcome via GitHub issues and pull requests. Using the Go
toolchain you want to verify, run uncached tests with shuffled execution order
from the repository root (POSIX shell):

```bash
GOTOOLCHAIN=local go test -shuffle=on -count=1 ./...
```

Repeat with the exact minimum version declared in `go.mod` and current stable
Go. `GOTOOLCHAIN=local` prevents Go from silently downloading a newer toolchain
to satisfy the module's minimum. In PowerShell, set `$env:GOTOOLCHAIN = "local"`
before running the `go test` command.

With a race-detector-supported toolchain and C compiler, reproduce CI's race and
atomic coverage check:

```bash
GOTOOLCHAIN=local go test -race -shuffle=on -count=1 -covermode=atomic -coverprofile=coverage.out ./...
go tool cover -func=coverage.out
```

The [Go CI workflow](https://github.com/atankeev/kmeans/actions/workflows/go.yml)
runs compatibility tests on Linux x64 with minimum and stable Go, stable-Go smoke
tests on Windows x64 and macOS ARM64, and race detection with atomic coverage on
Linux x64. Jobs record their actual Go version, OS, and architecture. Each race
job publishes function coverage in its run summary and retains a downloadable
`coverage-linux-stable` profile artifact for 30 days. Missing profiles or failed
uploads fail that job; no overall coverage threshold is enforced. The `build`
check succeeds only when every compatibility, smoke, and race/coverage job
succeeds. These checks preserve the platform-local numerical and
[Reproducibility](#reproducibility) guarantees.

Install the pinned lint tool explicitly, then run the same checks as CI:

```bash
make tools
make lint
```

Lint uses **golangci-lint v2.15.0** and **Go 1.27.2**, with version pins maintained
in the Makefile. The library's minimum supported Go version remains **1.24.4**;
CI selects the minimum test toolchain from `go.mod`. Lint commands select their
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
`main` with a ten-minute timeout and supplies CI's vet and formatting checks.

### Vulnerability scanning

Use the latest stable Go release, Make, and network access to install the pinned
scanner and scan the library from the repository root:

```bash
make vuln
```

`make vuln-tools` installs only the scanner. `make vuln` reinstalls it from the
pinned module graph before running `.bin/govulncheck ./...` from the library root.
The scanner's version, dependencies, and required Go minimum live in
`tools/govulncheck/go.mod` and `go.sum`, separate from the library's dependencies
and Go minimum. Installation uses the committed graph and checksums without
updating the module files. If Go needs to download a toolchain, the checksum and
proxy considerations described above also apply.

The **Vulnerability scan** workflow runs the `govulncheck` check on pushes and
pull requests targeting `main`, manual runs, and Mondays at 06:23 UTC (09:23 in
Europe/Istanbul). It uses stable Go; a passing scan does not establish security
on every older supported Go version. Installation errors, scanner errors,
vulnerability database/network failures, and reported vulnerabilities fail the
check, with complete diagnostics in the Actions logs.

To investigate scheduled failures, open **Actions → Vulnerability scan**, select
the failed scheduled run, and inspect the `govulncheck` job's installation and
scan step. After investigating or fixing the cause, choose **Re-run failed jobs**.
To scan the current default branch instead of the original run's commit, choose
**Run workflow** on `main` (or run `gh workflow run vulnerability.yml --ref main`).
Track any vulnerability and required Go or dependency remediation in a separate
issue; remediation can block completion of the scanning setup.

Dependabot checks the scanner module weekly on Mondays at 09:00 Europe/Istanbul
and opens reviewable PRs for `golang.org/x/vuln`, including necessary dependency
graph changes. These PRs run the same vulnerability check; review compatibility
and the scan result before merging. Other transitive dependencies do not receive
separate version-update PRs from this entry. To maintain the pin manually, update
the nested module and its checksums, then run `make vuln`; do not add a second
version pin to commands or change the root module for scanner maintenance.
The scheduled run alone is not a merge gate; required-check policy is tracked in
issue #34.

This library was developed with the assistance of AI assistants.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
