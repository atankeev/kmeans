# Fail safely when finite arithmetic cannot be represented

For finite input, the library uses stable floating-point calculations for the known avoidable overflow paths in means, variance, distances, weighted selection, convergence, and inertia. It does not promise to recover every mathematically representable result: when required arithmetic cannot be represented or processed safely, `Cluster` returns a nil result and an error matching `ErrNumericalOverflow`. This boundary preserves practical behavior and deterministic seeded calls without introducing arbitrary-precision arithmetic; unlike convergence failure, numerical overflow never returns a usable non-converged result.

An unrepresentable intermediate squared distance is not by itself a failure. Distance comparisons, initialization weights, and convergence checks use scaled calculations when that can still produce a safe decision; overflow is reported only when an obligatory result cannot be represented or the calculation cannot be completed safely. In particular, unrepresentable final inertia is a numerical overflow.

Only the sentinel match and nil-result guarantee are stable API. Wrapped error text may identify the failing stage but is not part of the compatibility contract. Private helper signatures and behavior may change to enforce this policy.
