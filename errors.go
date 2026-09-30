package kmeans

import "errors"

var (
	// ErrConfigNotInitialized indicates that Kmeans was not created with New or NewWithOptions.
	ErrConfigNotInitialized = errors.New("config not initialized")
	// ErrInvalidNInit indicates that NInit is not positive.
	ErrInvalidNInit = errors.New("n_init must be positive")
	// ErrInvalidNCentroidsInitTrials indicates that NCentroidsInitTrials is not positive.
	ErrInvalidNCentroidsInitTrials = errors.New("n_centroids_init_trials must be positive")
	// ErrInvalidMaxIter indicates that MaxIter is not positive.
	ErrInvalidMaxIter = errors.New("max_iter must be positive")
	// ErrInvalidTol indicates that Tol is not positive and finite.
	ErrInvalidTol = errors.New("tol must be positive and finite")
	// ErrInvalidInitMethod indicates that the configured centroid initialization method is unsupported.
	ErrInvalidInitMethod = errors.New("invalid initialization method")
	// ErrInvalidK indicates that NClusters is outside the valid range for the data.
	ErrInvalidK = errors.New("k must be between 1 and number of samples, inclusive")
	// ErrEmptyData indicates that the input contains no samples.
	ErrEmptyData = errors.New("data cannot be empty")
	// ErrInvalidData indicates an invalid input data shape.
	ErrInvalidData = errors.New("data must be a 2D slice")
	// ErrNoFeatures indicates that samples contain no features.
	ErrNoFeatures = errors.New("data must contain at least one feature")
	// ErrNonFiniteData indicates that a coordinate is NaN or infinite.
	ErrNonFiniteData = errors.New("data contains a non-finite coordinate")
	// ErrConvergenceFailed indicates that the selected result did not converge within MaxIter.
	// Cluster returns a usable final result together with this error.
	ErrConvergenceFailed = errors.New("algorithm failed to converge within max_iterations")
	// ErrNumericalOverflow indicates that finite input produced a required arithmetic result
	// that cannot be represented or processed safely. Cluster returns a nil result.
	ErrNumericalOverflow = errors.New("numerical overflow")
)
