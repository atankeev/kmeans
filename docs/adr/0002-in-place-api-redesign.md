---
status: accepted
---

# Replace the public API in place before adoption

The API redesign discussed in [issue #29](https://github.com/atankeev/kmeans/issues/29) replaces the previous contract in the existing module, without a `/v2` module or legacy aliases. The library has no consumers, and the maintainer explicitly approved this deviation from the backward-compatibility convention to avoid maintaining an unused legacy API. The maintainer confirmed the full design; the redesign is implemented in issue #30.

The entry point is `Cluster(data, k, opts...) (*Result, error)`, without `context.Context` or cancellation. The maintainer deliberately removed cancellation from the issue's requirements to avoid checks and added complexity in computational loops; callers consequently cannot interrupt an in-progress call through the library API.

Only library-provided options may configure a call; configuration remains private. Public `Option` is an interface with an unexported `apply(*config)` method, implemented by a private `optionFunc` type returned through library factories. Because external types can still satisfy this interface by embedding `Option`, `Cluster` also requires each option's dynamic type to be exactly `optionFunc`. Wrappers, nil options, and unsupported implementations return a nil result and `ErrInvalidOption`. Options capture values and apply them to configuration owned by each call, allowing reuse across concurrent calls while keeping configuration extension under library control.

The accepted names are `DefaultRuns`, `DefaultMaxIterations`, `DefaultTolerance`, `ErrInvalidRuns`, `ErrInvalidMaxIterations`, and `ErrInvalidTolerance`, without aliases for the old names.

Reaching the iteration limit produces a successful result with `Converged` and `Iterations`, and a nil error; `ErrConvergenceFailed` is removed. Initialization runs remain ranked by internal inertia, even when the lowest-inertia run has not converged; non-convergence does not justify selecting a worse converged result. This replaces the convergence-error contract in [ADR-0001](0001-finite-arithmetic-policy.md) while preserving its numerical overflow and run-ranking policies.
