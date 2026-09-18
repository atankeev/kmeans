package kmeans

import "errors"

// Error definitions for the kmeans package
var (
	ErrConfigNotInitialized        = errors.New("config not initialized")
	ErrInvalidNInit                = errors.New("n_init must be positive")
	ErrInvalidNCentroidsInitTrials = errors.New("n_centroids_init_trials must be positive")
	ErrInvalidMaxIter              = errors.New("max_iter must be positive")
	ErrInvalidTol                  = errors.New("tol must be positive and finite")
	// ErrInvalidInitMethod indicates that the configured centroid initialization method is unsupported.
	ErrInvalidInitMethod = errors.New("invalid initialization method")
	ErrInvalidK          = errors.New("k must be between 1 and number of samples, inclusive")
	ErrEmptyData         = errors.New("data cannot be empty")
	ErrInvalidData       = errors.New("data must be a 2D slice")
	// ErrNoFeatures indicates that samples contain no features.
	ErrNoFeatures = errors.New("data must contain at least one feature")
	// ErrNonFiniteData indicates that a coordinate is NaN or infinite.
	ErrNonFiniteData = errors.New("data contains a non-finite coordinate")
	// ErrConvergenceFailed indicates that the selected result did not converge within MaxIter.
	// Cluster returns a usable final result together with this error.
	ErrConvergenceFailed = errors.New("algorithm failed to converge within max_iterations")
)
