package kmeans

import (
	"cmp"
	"fmt"
	"math"
	"math/rand"
	"slices"
	"time"
)

// Default configuration values
const (
	// DefaultNInit is the default number of times the k-means algorithm will be run
	// with different centroid seeds to find the best result
	DefaultNInit = 10

	// DefaultMaxIter is the default maximum number of iterations of the k-means algorithm
	// for a single run before declaring convergence
	DefaultMaxIter = 300

	// DefaultTol is the default relative tolerance used to scale the mean
	// per-feature variance of the input data. Convergence is declared when the
	// squared Frobenius norm of the center shift does not exceed that threshold.
	DefaultTol = 1e-4
)

// InitMethod defines the centroid initialization strategy
type InitMethod string

const (
	InitKMeansPlusPlus InitMethod = "k-means++" // Default
	InitRandom         InitMethod = "random"
)

// Result represents the result of K-means clustering
type Result struct {
	// Cluster labels for each data point
	Labels []int

	// Cluster centroids
	Centroids [][]float64

	// Final inertia (sum of squared distances to closest centroid)
	Inertia float64
}

// Kmeans represents a K-means clustering algorithm instance.
//
// Cluster may be called concurrently on the same instance. Callers must not mutate the
// Kmeans configuration or input data while a call is in progress. Each call creates its own
// random state from RandomSeed, so equal configuration, seed, and data produce equal results.
type Kmeans struct {
	// Number of clusters (required parameter, no default)
	NClusters int

	// Number of times the k-means algorithm will be run with different centroid seeds
	NInit int

	// Maximum number of iterations of the k-means algorithm for a single run
	MaxIter int

	// Relative tolerance used to scale the mean per-feature variance of the input
	// data. The resulting threshold is compared with the squared Frobenius norm
	// of the center shift between consecutive iterations.
	Tol float64

	// Method for initialization: "k-means++" (default), "random"
	Init InitMethod

	// Random seed for reproducible results
	RandomSeed int64

	// Number of trials for k-means++ initialization
	NCentroidsInitTrials int

	// Internal fields for validation
	initialized bool
}

// Option is a functional option for configuring Kmeans
type Option func(*Kmeans)

// New creates a new Kmeans instance with required nClusters and default values for other parameters
func New(nClusters int) *Kmeans {
	return &Kmeans{
		NClusters:            nClusters,
		NInit:                DefaultNInit,
		MaxIter:              DefaultMaxIter,
		Tol:                  DefaultTol,
		Init:                 InitKMeansPlusPlus,
		NCentroidsInitTrials: int(2 + math.Log(float64(nClusters))),
		RandomSeed:           time.Now().UnixNano(),
		initialized:          true,
	}
}

// NewWithOptions creates a new Kmeans instance with required nClusters and custom options.
// Options retain the values supplied by the caller, including invalid values. Call Validate to
// check the configuration eagerly; Cluster also validates it before processing data.
func NewWithOptions(nClusters int, options ...Option) *Kmeans {
	kmeans := New(nClusters)

	for _, option := range options {
		option(kmeans)
	}

	return kmeans
}

// WithNInit sets the number of initializations
func WithNInit(nInit int) Option {
	return func(k *Kmeans) {
		k.NInit = nInit
	}
}

// WithMaxIter sets the maximum number of iterations
func WithMaxIter(maxIter int) Option {
	return func(k *Kmeans) {
		k.MaxIter = maxIter
	}
}

// WithTol sets the relative tolerance for convergence. The tolerance is scaled by
// the mean per-feature variance of the input data before clustering starts.
func WithTol(tol float64) Option {
	return func(k *Kmeans) {
		k.Tol = tol
	}
}

// WithInitMethod sets the initialization method
func WithInitMethod(init InitMethod) Option {
	return func(k *Kmeans) {
		k.Init = init
	}
}

// WithRandomSeed sets the random seed used by each clustering call.
func WithRandomSeed(seed int64) Option {
	return func(k *Kmeans) {
		k.RandomSeed = seed
	}
}

// WithNCentroidsInitTrials sets the number of trials for centroid initialization (used in k-means++)
func WithNCentroidsInitTrials(n int) Option {
	return func(k *Kmeans) {
		k.NCentroidsInitTrials = n
	}
}

// Validate checks if the Kmeans configuration is valid. Counts must be positive, and Tol must be
// positive and finite.
func (k *Kmeans) Validate() error {
	if !k.initialized {
		return ErrConfigNotInitialized
	}

	if k.NClusters <= 0 {
		return ErrInvalidK
	}

	if k.NInit <= 0 {
		return ErrInvalidNInit
	}

	if k.MaxIter <= 0 {
		return ErrInvalidMaxIter
	}

	if k.Tol <= 0 || math.IsNaN(k.Tol) || math.IsInf(k.Tol, 0) {
		return ErrInvalidTol
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

// Cluster performs K-means clustering on the given data.
//
// Cluster is safe to call concurrently on the same Kmeans instance, provided callers do not
// mutate the instance or data during a call. Each invocation starts with RandomSeed and does
// not share random state with other invocations.
//
// If the selected lowest-inertia run does not converge within MaxIter, Cluster returns
// its final result together with ErrConvergenceFailed. For configuration and data errors,
// the returned result is nil.
func (k *Kmeans) Cluster(data [][]float64) (*Result, error) {
	// Validate configuration
	if err := k.Validate(); err != nil {
		return nil, fmt.Errorf("invalid configuration: %w", err)
	}

	// Validate data
	if err := k.validateData(data); err != nil {
		return nil, fmt.Errorf("invalid data: %w", err)
	}

	labels := make([]int, len(data))
	previousLabels := make([]int, len(data))
	// Scratch buffer reused by initialization and empty-cluster recovery.
	distances := make([]float64, len(data))
	randomState := k.newRandomState()
	tolerance := calculateTolerance(data, k.Tol)
	var (
		bestResult    *Result
		bestLabels    []int
		bestConverged bool
	)

	// Run the clustering algorithm multiple times and return the best result
	for range k.NInit {
		result, converged := k.lloydKMeans(
			data,
			labels,
			previousLabels,
			distances,
			randomState,
			tolerance,
		)
		if bestResult == nil || result.Inertia < bestResult.Inertia {
			bestResult = result
			bestLabels = append(bestLabels[:0], labels...)
			bestConverged = converged
		}
	}

	bestResult.Labels = bestLabels
	if !bestConverged {
		return bestResult, ErrConvergenceFailed
	}

	return bestResult, nil
}

func (k *Kmeans) newRandomState() *rand.Rand {
	return rand.New(rand.NewSource(k.RandomSeed))
}

// validateData validates the input data.
//
// This function checks if the data is empty, has consistent dimensions,
// and if the number of clusters is not greater than the number of data points.
//
// Parameters:
//
//	data: The dataset, where each element is a point (slice of float64).
//
// Returns:
//
//	An error if the data is invalid.
//	Nil if the data is valid.
func (k *Kmeans) validateData(data [][]float64) error {
	// Validate input data
	if len(data) == 0 {
		return ErrEmptyData
	}

	// Validate that all data points have the same dimensions and finite coordinates.
	expectedDim := len(data[0])
	for pointIndex, point := range data {
		if len(point) != expectedDim {
			return fmt.Errorf("data point at index %d has dimension %d, expected %d", pointIndex, len(point), expectedDim)
		}

		for dimensionIndex, coordinate := range point {
			if math.IsNaN(coordinate) || math.IsInf(coordinate, 0) {
				return fmt.Errorf("%w at point %d, dimension %d", ErrNonFiniteData, pointIndex, dimensionIndex)
			}
		}
	}

	if expectedDim == 0 {
		return ErrNoFeatures
	}

	if k.NClusters > len(data) {
		return fmt.Errorf(
			"%w: number of clusters (%d) cannot be greater than number of samples (%d)",
			ErrInvalidK,
			k.NClusters,
			len(data),
		)
	}

	return nil
}

// initRandomCentroids initializes cluster centroids by randomly selecting data points
func (k *Kmeans) initRandomCentroids(data [][]float64, randomState *rand.Rand) [][]float64 {
	centroids := make([][]float64, 0, k.NClusters)
	usedIndices := make(map[int]struct{})

	for len(centroids) < k.NClusters {
		randomIndex := randomState.Intn(len(data))
		if _, ok := usedIndices[randomIndex]; !ok {
			centroid := make([]float64, len(data[randomIndex]))
			copy(centroid, data[randomIndex])

			centroids = append(centroids, centroid)
			usedIndices[randomIndex] = struct{}{}
		}
	}

	return centroids
}

// initKMeansPlusPlusCentroids initializes cluster centroids using the k-means++ algorithm.
// This method selects centroids with probability proportional to their squared distance
// from existing centroids, ensuring better initial placement than random selection.
// The algorithm guarantees that no duplicate data points are selected as centroids.
func (k *Kmeans) initKMeansPlusPlusCentroids(data [][]float64) [][]float64 {
	return k.initKMeansPlusPlusCentroidsWithDistances(data, make([]float64, len(data)), k.newRandomState())
}

// initKMeansPlusPlusCentroidsWithDistances initializes cluster centroids using the k-means++
// algorithm, reusing a caller-provided distances buffer (allocated once per Cluster call).
// The buffer must have len(data) capacity; it is re-initialized per run.
func (k *Kmeans) initKMeansPlusPlusCentroidsWithDistances(
	data [][]float64,
	distances []float64,
	randomState *rand.Rand,
) [][]float64 {
	nLocalTrials := k.NCentroidsInitTrials

	centroids := make([][]float64, 0, k.NClusters)

	// Track already selected indices to avoid duplicates
	usedIndices := make(map[int]struct{})

	// Squared distance from each data point to the nearest selected centroid
	if len(distances) != len(data) {
		distances = make([]float64, len(data))
	}
	for i := range distances {
		distances[i] = math.Inf(1)
	}

	// Adds a data point as a centroid and updates the nearest-centroid distances.
	addCentroid := func(idx int) {
		centroid := make([]float64, len(data[idx]))
		copy(centroid, data[idx])
		centroids = append(centroids, centroid)
		usedIndices[idx] = struct{}{}
		updateMinDistances(distances, data, centroid)
	}

	// Select first centroid randomly
	addCentroid(randomState.Intn(len(data)))

	// Select remaining centroids
	for len(centroids) < k.NClusters {
		// When nLocalTrials <= 1, use standard k-means++ initialization
		if nLocalTrials <= 1 {
			// Select next centroid with probability proportional to squared distance
			addCentroid(selectUniqueIndex(distances, usedIndices, randomState))
		} else {
			// Use greedy k-means++ initialization with nLocalTrials
			bestCentroidIdx := -1
			bestInertia := math.Inf(1)

			for range nLocalTrials {
				// Select candidate centroid with probability proportional to squared distance
				candidateIdx := selectUniqueIndex(distances, usedIndices, randomState)

				// Calculate the inertia with this candidate without materializing the
				// candidate as a centroid: min(existing distance, distance to candidate)
				inertia := candidateCost(distances, data, data[candidateIdx])

				// If this candidate is better than the best so far, update the best centroid
				if inertia < bestInertia {
					bestInertia = inertia
					bestCentroidIdx = candidateIdx
				}
			}

			// Add the best centroid to the list of centroids
			if bestCentroidIdx != -1 {
				addCentroid(bestCentroidIdx)
			} else {
				// Fallback: select a random unused data point as centroid
				addCentroid(selectUniqueIndex(distances, usedIndices, randomState))
			}
		}
	}

	return centroids
}

// initCentroids initializes the cluster centroids using the selected initialization method.
//
// This function selects the appropriate initialization method based on the configuration
// and initializes the centroids accordingly.
//
// Parameters:
//
//	data: The dataset, where each element is a point (slice of float64).
//
// Returns:
//
//	A slice of new cluster centers, where each center is a slice of float64.
func (k *Kmeans) initCentroids(data [][]float64) [][]float64 {
	return k.initCentroidsWithDistances(data, make([]float64, len(data)), k.newRandomState())
}

// initCentroidsWithDistances initializes the cluster centroids using the selected initialization
// method, reusing a caller-provided distances buffer for the k-means++ method. The buffer is
// ignored by initialization methods that do not need it.
func (k *Kmeans) initCentroidsWithDistances(
	data [][]float64,
	distances []float64,
	randomState *rand.Rand,
) [][]float64 {
	switch k.Init {
	case InitRandom:
		return k.initRandomCentroids(data, randomState)
	default:
		return k.initKMeansPlusPlusCentroidsWithDistances(data, distances, randomState)
	}
}

// clusterSingle performs K-means clustering using Lloyd's algorithm.
//
// This function performs the main loop of the algorithm, which iteratively assigns
// each data point to the nearest cluster center, updates the centers, and checks for convergence.
//
// Parameters:
//
//	data: The dataset, where each element is a point (slice of float64).
//
// Returns:
//
//	A pointer to Result containing final centroids, labels, and inertia.
func (k *Kmeans) clusterSingle(data [][]float64) *Result {
	labels := make([]int, len(data))
	previousLabels := make([]int, len(data))
	distances := make([]float64, len(data))
	result, _ := k.lloydKMeans(
		data,
		labels,
		previousLabels,
		distances,
		k.newRandomState(),
		calculateTolerance(data, k.Tol),
	)

	return result
}

// lloydKMeans performs K-means clustering using Lloyd's algorithm.
//
// The method iteratively assigns each data point to the nearest cluster center,
// then updates the cluster centers as the mean of the assigned points, until convergence
// or the maximum number of iterations is reached. The final assignments and centroids
// are returned in a Result struct, along with the final inertia (sum of squared distances).
//
// Parameters:
//
//	data: The dataset, where each element is a point (slice of float64).
//
// Returns:
//
//	A pointer to Result containing final centroids, labels, and inertia, and whether the run converged.
func (k *Kmeans) lloydKMeans(
	data [][]float64,
	labels []int,
	previousLabels []int,
	distances []float64,
	randomState *rand.Rand,
	tolerance float64,
) (*Result, bool) {
	centers := k.initCentroidsWithDistances(data, distances, randomState)
	clear(labels)

	dim := len(data[0])
	next := newCenterBuffer(k.NClusters, dim)
	counts := make([]int, k.NClusters)
	hasPreviousLabels := false
	converged := false

	for range k.MaxIter {
		// Step 1: Assign each data point to the nearest cluster center
		labels = assignPointsToClusters(data, centers, labels)

		// Step 2: Update cluster centers based on current assignments (Lloyd's update step).
		// Empty clusters take the farthest assigned samples, ordered deterministically by
		// descending assignment error and then ascending sample index.
		clearCenters(next)
		clear(counts)
		k.updateCentersLloydInto(next, counts, distances, data, labels, centers)

		// As in sklearn, unchanged assignments take precedence over the
		// tolerance-based squared Frobenius norm check.
		labelsUnchanged := hasPreviousLabels && slices.Equal(labels, previousLabels)
		converged = labelsUnchanged || checkConvergence(centers, next, tolerance)

		// Swap buffers: the just-computed centers become the current ones for the next iteration
		centers, next = next, centers

		if converged {
			break
		}

		copy(previousLabels, labels)
		hasPreviousLabels = true
	}

	// Assign points to clusters again to ensure final assignments
	labels = assignPointsToClusters(data, centers, labels)

	finalInertia := calculateInertiaByLabels(data, centers, labels)

	return &Result{
		Centroids: centers,
		Labels:    labels,
		Inertia:   finalInertia,
	}, converged
}

// updateCentersLloyd computes new cluster centers (centroids) for the Lloyd's algorithm step.
//
// For each cluster, it calculates the mean of all points assigned to that cluster.
// Empty clusters are relocated to the samples with the largest current assignment error.
//
// Parameters:
//
//	data:   The dataset, where each element is a point (slice of float64).
//	labels:  Cluster assignments for each point in data. labels[i] is the cluster index for data[i].
//	centers: Cluster centers used to produce labels.
//
// Returns:
//
//	A slice of new cluster centers, where each center is a slice of float64.
func (k *Kmeans) updateCentersLloyd(data [][]float64, labels []int, centers [][]float64) [][]float64 {
	dim := len(data[0])
	newCenters := newCenterBuffer(k.NClusters, dim)

	k.updateCentersLloydInto(
		newCenters,
		make([]int, k.NClusters),
		make([]float64, len(data)),
		data,
		labels,
		centers,
	)

	return newCenters
}

// updateCentersLloydInto computes new cluster centers (centroids) into the provided dst buffer
// for the Lloyd's algorithm step. dst must have k.NClusters rows of len(data[0]) columns and
// must be zeroed before the call, as must be counts. For each cluster, it calculates the mean of
// all points assigned to that cluster. Empty clusters take distinct farthest samples from clusters
// that can donate a sample without becoming empty themselves.
func (k *Kmeans) updateCentersLloydInto(
	dst [][]float64,
	counts []int,
	distances []float64,
	data [][]float64,
	labels []int,
	centers [][]float64,
) {
	dim := len(data[0])

	for i, point := range data {
		cluster := labels[i]
		for d := range dim {
			dst[cluster][d] += point[d]
		}
		counts[cluster]++
	}

	k.relocateEmptyClusters(dst, counts, distances, data, labels, centers)

	for i := 0; i < k.NClusters; i++ {
		for d := range dim {
			dst[i][d] /= float64(counts[i])
		}
	}
}

// relocateEmptyClusters moves distinct samples into empty clusters. Candidates are ranked by
// descending squared distance to their currently assigned center, with lower sample indexes first
// on ties. A source cluster must retain at least one sample.
func (k *Kmeans) relocateEmptyClusters(
	sums [][]float64,
	counts []int,
	distances []float64,
	data [][]float64,
	labels []int,
	centers [][]float64,
) {
	hasEmptyCluster := slices.Contains(counts, 0)
	if !hasEmptyCluster {
		return
	}

	candidates := make([]int, len(data))
	for i, point := range data {
		candidates[i] = i
		distances[i] = squaredEuclideanDistance(point, centers[labels[i]])
	}
	slices.SortFunc(candidates, func(a, b int) int {
		if byDistance := cmp.Compare(distances[b], distances[a]); byDistance != 0 {
			return byDistance
		}

		return cmp.Compare(a, b)
	})

	nextCandidate := 0
	for emptyCluster := 0; emptyCluster < k.NClusters; emptyCluster++ {
		if counts[emptyCluster] != 0 {
			continue
		}

		for counts[labels[candidates[nextCandidate]]] <= 1 {
			nextCandidate++
		}

		candidate := candidates[nextCandidate]
		nextCandidate++
		sourceCluster := labels[candidate]
		for dimension, coordinate := range data[candidate] {
			sums[sourceCluster][dimension] -= coordinate
			sums[emptyCluster][dimension] += coordinate
		}
		counts[sourceCluster]--
		counts[emptyCluster]++
		labels[candidate] = emptyCluster
	}
}

// newCenterBuffer allocates a flat-backed buffer with nClusters rows of dim columns.
func newCenterBuffer(nClusters, dim int) [][]float64 {
	buf := make([][]float64, nClusters)
	backing := make([]float64, nClusters*dim)
	for i := 0; i < nClusters; i++ {
		buf[i] = backing[i*dim : (i+1)*dim]
	}
	return buf
}

// clearCenters zeroes all values in the buffer.
func clearCenters(buf [][]float64) {
	for _, center := range buf {
		clear(center)
	}
}

// assignPointsToClusters assigns each data point to the nearest cluster center.
//
// For each point in the dataset, this function computes the squared Euclidean distance
// to each center and assigns the point to the cluster with the minimum distance.
//
// Parameters:
//
//	data:    The dataset, where each element is a point (slice of float64).
//	centers: The current cluster centers, where each center is a slice of float64.
//
// Returns:
//
//	A slice of integers where the i-th element is the index of the nearest center for data point i.
func assignPointsToClusters(data [][]float64, centers [][]float64, labels []int) []int {
	for i, point := range data {
		bestCluster := labels[i]
		if bestCluster < 0 || bestCluster >= len(centers) {
			bestCluster = 0
		}
		minDistance := squaredEuclideanDistance(point, centers[bestCluster])

		// Find the nearest center for each point. Keeping the current label on exact ties
		// lets recovered clusters remain populated when samples or centers are duplicates.
		for j, center := range centers {
			distance := squaredEuclideanDistance(point, center)
			if distance < minDistance {
				minDistance = distance
				bestCluster = j
			}
		}

		labels[i] = bestCluster
	}

	return labels
}

// squaredEuclideanDistance calculates the squared Euclidean distance between two points.
// This is more efficient than calculating the actual Euclidean distance since it avoids
// the square root operation, and for clustering purposes, the relative distances are sufficient.
func squaredEuclideanDistance(p1, p2 []float64) float64 {
	sum := 0.0

	for i := range p1 {
		diff := p1[i] - p2[i]
		// sum += diff * diff -> but more accurate
		sum = math.FMA(diff, diff, sum)
	}

	return sum
}

// calculateInertia calculates the total inertia (within-cluster sum of squares) for the given data points and cluster centers.
// Inertia is the sum of squared distances from each data point to its nearest cluster center.
// Lower inertia indicates better clustering (points are closer to their assigned centers).
func calculateInertia(data [][]float64, centers [][]float64) float64 {
	totalInertia := 0.0

	for _, point := range data {
		minDist := math.Inf(1)
		for _, center := range centers {
			dist := squaredEuclideanDistance(point, center)
			minDist = math.Min(minDist, dist)
		}
		totalInertia += minDist
	}

	return totalInertia
}

// calculateInertiaByLabels computes the total within-cluster sum of squared distances (inertia)
// for the given data points, cluster centers, and cluster assignments (labels).
//
// Parameters:
//
//	data:    A slice of data points, where each data point is a slice of float64 features.
//	centers: A slice of cluster centers, where each center is a slice of float64 features.
//	labels:  A slice of integers where labels[i] is the index of the cluster center assigned to data point i.
//
// Returns:
//
//	The total inertia, which is the sum of squared Euclidean distances from each data point to its assigned cluster center.
func calculateInertiaByLabels(data [][]float64, centers [][]float64, labels []int) float64 {
	inertia := 0.0

	for i, point := range data {
		cluster := labels[i]
		inertia += squaredEuclideanDistance(point, centers[cluster])
	}

	return inertia
}

// updateMinDistances updates distances[i] to the minimum of its current value and the squared
// Euclidean distance from data point i to the given center. Used in k-means++ initialization to
// incrementally maintain each point's distance to the nearest selected centroid.
func updateMinDistances(distances []float64, data [][]float64, center []float64) {
	for i, point := range data {
		dist := squaredEuclideanDistance(point, center)
		distances[i] = math.Min(distances[i], dist)
	}
}

// candidateCost returns the total inertia that would result from adding the given candidate
// center to the set of already selected centers, given the current per-point distances to the
// nearest selected center. This is the sum over points of min(distances[i], squared distance to
// candidate), which equals the inertia of the selected centers plus the candidate.
func candidateCost(distances []float64, data [][]float64, candidate []float64) float64 {
	total := 0.0

	for i, point := range data {
		dist := squaredEuclideanDistance(point, candidate)
		total += math.Min(distances[i], dist)
	}

	return total
}

// computeDistancesToCenters calculates the minimum squared Euclidean distance from each data point to its nearest cluster center.
// Returns a slice where distances[i] is the squared distance from data point i to its closest center.
func computeDistancesToCenters(data [][]float64, centers [][]float64, distances []float64) []float64 {
	for i, point := range data {
		minDist := math.Inf(1)
		for _, center := range centers {
			dist := squaredEuclideanDistance(point, center)
			minDist = math.Min(minDist, dist)
		}
		distances[i] = minDist
	}

	return distances
}

// weightedRandomChoice performs weighted random selection from a slice of weights.
// Returns the index of the selected element, where the probability of selecting index i
// is proportional to weights[i]. This is used in k-means++ initialization to select
// centroids with probability proportional to their squared distance from existing centroids.
func weightedRandomChoice(weights []float64, rng *rand.Rand) int {
	totalWeight := 0.0
	for _, w := range weights {
		totalWeight += w
	}

	if totalWeight == 0.0 || math.IsNaN(totalWeight) || math.IsInf(totalWeight, 0) {
		return rng.Intn(len(weights))
	}

	r := rng.Float64() * totalWeight
	cumulative := 0.0

	for i, weight := range weights {
		cumulative += weight
		if r <= cumulative {
			return i
		}
	}

	return len(weights) - 1
}

// selectUniqueIndex selects a unique index using weighted random choice, ensuring
// that the selected index has not been used before. This is used in k-means++
// initialization to prevent selecting the same data point multiple times as a centroid.
func selectUniqueIndex(weights []float64, usedIndices map[int]struct{}, rng *rand.Rand) int {
	for {
		selectedIdx := weightedRandomChoice(weights, rng)
		if _, alreadyUsed := usedIndices[selectedIdx]; !alreadyUsed {
			return selectedIdx
		}
	}
}

// calculateTolerance converts a relative tolerance into the data-dependent
// threshold used by sklearn: relativeTolerance multiplied by the mean population
// variance across features.
func calculateTolerance(data [][]float64, relativeTolerance float64) float64 {
	dim := len(data[0])
	means := make([]float64, dim)
	for _, point := range data {
		for feature, value := range point {
			means[feature] += value
		}
	}

	sampleCount := float64(len(data))
	for feature := range means {
		means[feature] /= sampleCount
	}

	var varianceSum float64
	for _, point := range data {
		for feature, value := range point {
			difference := value - means[feature]
			varianceSum += difference * difference
		}
	}

	meanVariance := varianceSum / (sampleCount * float64(dim))
	return meanVariance * relativeTolerance
}

// checkConvergence reports whether the squared Frobenius norm of the center
// shift is at or below the data-dependent tolerance.
func checkConvergence(oldCenters, newCenters [][]float64, tolerance float64) bool {
	if len(oldCenters) != len(newCenters) {
		return false
	}

	var centerShift float64
	for i := range oldCenters {
		if len(oldCenters[i]) != len(newCenters[i]) {
			return false
		}

		for feature := range oldCenters[i] {
			difference := newCenters[i][feature] - oldCenters[i][feature]
			centerShift += difference * difference
		}
	}

	return centerShift <= tolerance
}
