package kmeans

import (
	"math"
	"math/rand"
)

const (
	// DefaultRuns is the default number of independent initialization runs.
	DefaultRuns = 10
	// DefaultMaxIterations is the default Lloyd iteration limit per run.
	DefaultMaxIterations = 300
	// DefaultTolerance is the default relative tolerance, scaled by the mean
	// per-feature population variance and compared with squared center movement.
	DefaultTolerance = 1e-4
)

// InitMethod identifies a centroid initialization strategy.
type InitMethod string

const (
	// InitKMeansPlusPlus selects initial centroids using k-means++.
	InitKMeansPlusPlus InitMethod = "k-means++"
	// InitRandom selects distinct sample indexes as initial centroids.
	InitRandom InitMethod = "random"
)

type config struct {
	NClusters            int
	NInit                int
	MaxIter              int
	Tol                  float64
	Init                 InitMethod
	RandomSeed           int64
	NCentroidsInitTrials int
}

// Option configures a Cluster call. Only values returned by the library's
// factories are accepted; wrappers and nil options return ErrInvalidOption.
// Built-in options may be reused across sequential and concurrent calls.
type Option interface{ apply(*config) }
type optionFunc func(*config)

func (option optionFunc) apply(k *config) { option(k) }

func newConfig(nClusters int) *config {
	trials := 2
	if nClusters > 0 {
		trials += int(math.Log(float64(nClusters)))
	}
	return &config{
		NClusters: nClusters, NInit: DefaultRuns, MaxIter: DefaultMaxIterations,
		Tol: DefaultTolerance, Init: InitKMeansPlusPlus,
		NCentroidsInitTrials: trials, RandomSeed: rand.Int63(),
	}
}

// WithRuns sets a positive number of runs (default DefaultRuns, 10).
func WithRuns(n int) Option { return optionFunc(func(k *config) { k.NInit = n }) }

// WithMaxIterations sets a positive Lloyd iteration limit per run (default
// DefaultMaxIterations, 300). Reaching it is not an error.
func WithMaxIterations(n int) Option { return optionFunc(func(k *config) { k.MaxIter = n }) }

// WithTolerance sets a positive, finite relative tolerance (default
// DefaultTolerance, 1e-4), scaled by mean per-feature population variance.
func WithTolerance(tol float64) Option { return optionFunc(func(k *config) { k.Tol = tol }) }

// WithInitMethod selects InitKMeansPlusPlus (the default) or InitRandom.
// Other values are invalid.
func WithInitMethod(init InitMethod) Option { return optionFunc(func(k *config) { k.Init = init }) }

// WithSeed sets a deterministic seed, including zero. All int64 values are valid.
// Without this option, each call selects its seed internally.
func WithSeed(seed int64) Option { return optionFunc(func(k *config) { k.RandomSeed = seed }) }

// WithNCentroidsInitTrials sets the positive number of candidate trials per
// k-means++ centroid after the first (default 2 + floor(ln(k)), per call).
// The count must be positive even when InitRandom is selected.
func WithNCentroidsInitTrials(n int) Option {
	return optionFunc(func(k *config) { k.NCentroidsInitTrials = n })
}

func (k *config) validate() error {
	if k.NClusters <= 0 {
		return ErrInvalidK
	}
	if k.NInit <= 0 {
		return ErrInvalidRuns
	}
	if k.MaxIter <= 0 {
		return ErrInvalidMaxIterations
	}
	if k.Tol <= 0 || math.IsNaN(k.Tol) || math.IsInf(k.Tol, 0) {
		return ErrInvalidTolerance
	}
	switch k.Init {
	case InitRandom, InitKMeansPlusPlus:
	default:
		return ErrInvalidInitMethod
	}
	if k.NCentroidsInitTrials <= 0 {
		return ErrInvalidNCentroidsInitTrials
	}
	return nil
}
