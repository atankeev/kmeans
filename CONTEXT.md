# K-Means Clustering

This context describes the numerical concepts exposed by the clustering library and the guarantees attached to its results.

## Language

**Finite input**:
A dataset whose every coordinate is representable as a finite `float64` value.
_Avoid_: Valid input, safe input

**Successful result**:
A clustering result returned without an error whose centroids and inertia contain only finite values.
_Avoid_: Partial result, best-effort result

**Numerical overflow**:
A failure to represent or safely process a required arithmetic result while clustering finite input.
Rounding a positive inertia to zero because it is too small to represent is not numerical overflow.
_Avoid_: Invalid data, convergence failure

**Usable non-converged result**:
A numerically valid clustering result returned with a convergence failure after the iteration limit is reached.
_Avoid_: Partial result, successful result

**Initialization run**:
One independent execution of the clustering algorithm from an initialized set of centroids; multiple runs may be compared to select the lowest-inertia result.
_Avoid_: Attempt, retry

**Inertia**:
The sum of squared distances from samples to their assigned centroids. A reported inertia of zero may reflect rounding of a very small positive value rather than exact coincidence of samples and centroids.
_Avoid_: Cost, score
