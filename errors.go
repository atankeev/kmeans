package kmeans

import "errors"

var (
	// ErrInvalidOption indicates a nil or unsupported option, including wrappers.
	ErrInvalidOption = errors.New("invalid option")
	// ErrInvalidRuns indicates that the run count is not positive.
	ErrInvalidRuns = errors.New("runs must be positive")
	// ErrInvalidNCentroidsInitTrials indicates that NCentroidsInitTrials is not positive.
	ErrInvalidNCentroidsInitTrials = errors.New("n_centroids_init_trials must be positive")
	// ErrInvalidMaxIterations indicates that the iteration limit is not positive.
	ErrInvalidMaxIterations = errors.New("max_iterations must be positive")
	// ErrInvalidTolerance indicates that tolerance is not positive and finite.
	ErrInvalidTolerance = errors.New("tolerance must be positive and finite")
	// ErrInvalidInitMethod indicates that the configured centroid initialization method is unsupported.
	ErrInvalidInitMethod = errors.New("invalid initialization method")
	// ErrInvalidK indicates that the cluster count is outside the valid range for the data.
	ErrInvalidK = errors.New("k must be between 1 and number of samples, inclusive")
	// ErrEmptyData indicates that the input contains no samples.
	ErrEmptyData = errors.New("data cannot be empty")
	// ErrInvalidData indicates inconsistent sample dimensions.
	ErrInvalidData = errors.New("data samples must have consistent dimensions")
	// ErrNoFeatures indicates that samples contain no features.
	ErrNoFeatures = errors.New("data must contain at least one feature")
	// ErrNonFiniteData indicates that a coordinate is NaN or infinite.
	ErrNonFiniteData = errors.New("data contains a non-finite coordinate")
	// ErrNumericalOverflow indicates that finite input produced a required arithmetic result
	// that cannot be represented or processed safely. Cluster returns a nil result.
	ErrNumericalOverflow = errors.New("numerical overflow")
)
