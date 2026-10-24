# Benchmark measurements and comparisons

The [Benchmarks workflow](https://github.com/atankeev/kmeans/actions/workflows/benchmarks.yml)
measures one revision every Wednesday at 06:00 UTC on `main`. To measure a branch,
choose **Actions → Benchmarks → Run workflow** and select that branch. There is no
separate SHA input. The workflow does not run on pull requests or compare revisions,
and no timing percentage blocks a PR.

Measurements use exactly Go **1.27.2**, `ubuntu-24.04`, and **`GOMAXPROCS=1`**:

```bash
go test -run '^$' -bench . -benchmem -count=5 ./...
```

No race or coverage instrumentation is enabled. Keep the benchmark compiler pin
separate from the library's minimum Go version; change it deliberately in a separate
PR. The current six leaf cases are `BenchmarkLloyd_KMeansPlusPlus`,
`BenchmarkLloyd_RandomInit`, and `BenchmarkLloyd_VaryingDataSize/size_100`,
`size_500`, `size_1000`, and `size_5000`. Each iteration measures a `Cluster` call
with seed 42 and ten initialization runs. The benchmark definitions in
[benchmark_test.go](../benchmark_test.go) specify the datasets and other settings.
A successful result with `Converged == false` is allowed. Workload expansion and
optimization investigations remain in issues #14–18.

Every run attempts to upload a **`benchmark-evidence`** artifact retained for
**90 days**. Download it from the run's summary. It contains:

- `benchmarks.txt`: raw timing (`ns/op`) and allocation (`B/op`, `allocs/op`) output.
- `environment.txt`: actual checked-out SHA, run/ref, UTC timestamp, exact Go
  version, OS/architecture, runner image version, CPU details, available and
  installed processor counts, and the explicit effective `GOMAXPROCS` override.
- `validation.txt`: confirmation of five measurements for each of the six cases,
  or diagnostics for missing, unexpected, or incomplete rows.
- `status.txt`: measurement and validation outcomes and a complete/incomplete marker.

A benchmark command failure fails the job even when output capture succeeds.
Missing cases, missing repetitions, unexpected cases, and malformed measurement
rows also fail the job. When changing the benchmark suite, update its explicit
expectations in [the validator](../scripts/validate-benchmarks.awk).
Available partial output and metadata are uploaded on failure; some files can be
absent if setup or measurement never starts. Check `status.txt` before interpreting
results. The entire job has a **15-minute** timeout; evidence collection after
cancellation, timeout, or forced runner termination is best effort.

Five CI repetitions establish that the measurement infrastructure works. They do
not satisfy the optimization investigations' requirement for at least ten
measurements per revision. Shared runners vary in CPU, load, virtualization, and
image contents. Even a fixed `ubuntu-24.04` label receives image updates. Historical
artifacts are useful evidence, but comparing different runs is not a controlled
baseline/candidate experiment.

## Compare two revisions locally

Use one idle machine, one exact measurement toolchain, the same `GOMAXPROCS`, and
equivalent benchmark cases for both revisions. Keep OS settings, power mode, CPU
affinity, and competing load consistent. Record CPU/platform information and the
actual commit SHAs. Do not run the two revisions concurrently.

The following Bash commands create detached checkouts and an output directory
outside them. Run from a repository checkout with Git, Go, Bash, and network
access. Replace `BASELINE_REF` and `CANDIDATE_REF` with the revisions being compared;
resolve them to SHAs before starting. `GOTOOLCHAIN=go1.27.2` selects the exact
compiler, downloading it if necessary, without changing `go.mod`.

```bash
set -euo pipefail
baseline_sha=$(git rev-parse 'BASELINE_REF^{commit}')
candidate_sha=$(git rev-parse 'CANDIDATE_REF^{commit}')
comparison_root=$(mktemp -d)
git worktree add --detach "$comparison_root/baseline" "$baseline_sha"
git worktree add --detach "$comparison_root/candidate" "$candidate_sha"

export GOTOOLCHAIN=go1.27.2 GOMAXPROCS=1 GOFLAGS=''
{
  printf 'Baseline: %s\nCandidate: %s\n' "$baseline_sha" "$candidate_sha"
  date -u '+Measured at: %Y-%m-%dT%H:%M:%SZ'
  go version
  go env GOOS GOARCH GOTOOLCHAIN GOFLAGS
  printf 'GOMAXPROCS: %s\n' "$GOMAXPROCS"
  uname -a
} > "$comparison_root/environment.txt"
# Also record CPU model, available processors, and power settings using your
# platform's tools (for example, lscpu and nproc on Linux).

# Two alternating batches of five give ten measurements per revision.
for batch in 1 2; do
  for revision in baseline candidate; do
    (
      cd "$comparison_root/$revision"
      go test -run '^$' -bench . -benchmem -count=5 ./...
    ) 2>&1 | tee -a "$comparison_root/$revision.txt"
  done
done
printf 'Results: %s\n' "$comparison_root"
```

Run the block in a Bash session or script where a failed command stops collection;
`pipefail` preserves a benchmark failure through `tee`. Do not compare incomplete
series. Check that every intended case has ten measurement rows in each file.
For repeated experiments, create fresh output files rather than appending to an
older experiment. Alternating batches reduces order effects but cannot eliminate
thermal drift or unrelated system activity. Reverse the starting order or collect
more alternating batches when investigating a small or inconsistent effect.

Review the definitions before comparing. Identical names do not prove equivalent
work: data, seed, options, setup excluded from timing, and the measured operation
must match. If definitions differ, use one common benchmark harness in both
detached checkouts where the API allows it, replacing conflicting benchmark
files rather than leaving duplicate names. Record the harness source/revision and
any uncommitted changes alongside results. If API or workload differences prevent
equivalence, do not claim a speedup or regression from that comparison.

## Install and interpret benchstat

The comparison tool pin is **`golang.org/x/perf/cmd/benchstat@v0.0.0-20261009192801-be2c69fb417e`**.
It is independent of the library dependency graph; installing it with an explicit
version does not edit the root `go.mod` or `go.sum`. This tool version requires
Go **1.26.0 or later** for installation. The example selects Go 1.27.2 for the tool
build as well; the measurement compiler remains the separately specified exact
Go 1.27.2 regardless of the tool's minimum requirement. The executable runs on the
platform for which it was built and needs no Go toolchain to analyze saved files.

In the same Bash session after collecting the two series:

```bash
mkdir -p "$comparison_root/tools"
GOTOOLCHAIN=go1.27.2 GOBIN="$comparison_root/tools" \
  go install golang.org/x/perf/cmd/benchstat@v0.0.0-20261009192801-be2c69fb417e
"$comparison_root/tools/benchstat" \
  "$comparison_root/baseline.txt" "$comparison_root/candidate.txt" \
  | tee "$comparison_root/comparison.txt"
```

Keep `go version -m "$comparison_root/tools/benchstat"` with the experiment to
confirm the installed tool version. Read timing and allocation tables separately:
lower `ns/op`, `B/op`, and `allocs/op` mean less measured time, allocated bytes,
and allocations per operation respectively. Inspect the central estimates,
confidence intervals, percentage changes, significance tests, and sample counts.
A `~` means the test did not detect a statistically significant difference; it
does not establish equivalence. Wide intervals or warnings about too few samples
mean the estimate is uncertain. Collect more measurements when needed and decide
whether a change is practically meaningful for the workload.

Statistical significance cannot remove systematic differences in environment,
compiler, or measured work. It cannot make separate shared-runner jobs controlled
comparisons. Preserve raw files, environment details, harness changes, tool
version, and interpretation with any reported result.

When finished, remove the detached worktrees:

```bash
git worktree remove "$comparison_root/baseline"
git worktree remove "$comparison_root/candidate"
```

Preserve the output directory for your report; worktree removal refuses
uncommitted harness changes unless you handle them first.
