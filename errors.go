package kmeans

import "errors"

// Error definitions for the kmeans package
var (
	ErrConfigNotInitialized        = errors.New("config not initialized")
	ErrInvalidNInit                = errors.New("n_init must be positive")
	ErrInvalidNCentroidsInitTrials = errors.New("n_centroids_init_trials must be positive")
	ErrInvalidMaxIter              = errors.New("max_iter must be positive")
	ErrInvalidTol                  = errors.New("tol must be positive")
	ErrInvalidRandomState          = errors.New("random_state cannot be nil")
	ErrInvalidK                    = errors.New("k must be positive and less than number of samples")
	ErrEmptyData                   = errors.New("data cannot be empty")
	ErrInvalidData                 = errors.New("data must be a 2D slice")
	ErrConvergenceFailed           = errors.New("algorithm failed to converge within max_iterations")
)
