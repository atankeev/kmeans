package kmeans

import (
	"fmt"
	"math"
	"math/rand"
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

	// DefaultTol is the default relative tolerance with regards to Frobenius norm
	// of the difference in the cluster centers of two consecutive iterations
	// to declare convergence
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

// Kmeans represents a K-means clustering algorithm instance
type Kmeans struct {
	// Number of clusters (required parameter, no default)
	NClusters int

	// Number of times the k-means algorithm will be run with different centroid seeds
	NInit int

	// Maximum number of iterations of the k-means algorithm for a single run
	MaxIter int

	// Relative tolerance with regards to Frobenius norm of the difference in the cluster centers
	// of two consecutive iterations to declare convergence
	Tol float64

	// Method for initialization: "k-means++" (default), "random"
	Init InitMethod

	// Random state for reproducible results
	RandomState *rand.Rand

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
		RandomState:          rand.New(rand.NewSource(time.Now().UnixNano())),
		initialized:          true,
	}
}

// NewWithOptions creates a new Kmeans instance with required nClusters and custom options
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
		if nInit > 0 {
			k.NInit = nInit
		}
	}
}

// WithMaxIter sets the maximum number of iterations
func WithMaxIter(maxIter int) Option {
	return func(k *Kmeans) {
		if maxIter > 0 {
			k.MaxIter = maxIter
		}
	}
}

// WithTol sets the tolerance for convergence
func WithTol(tol float64) Option {
	return func(k *Kmeans) {
		if tol > 0 {
			k.Tol = tol
		}
	}
}

// WithInitMethod sets the initialization method
func WithInitMethod(init InitMethod) Option {
	return func(k *Kmeans) {
		k.Init = init
	}
}

// WithRandomState sets the random state for reproducible results
func WithRandomState(randomState *rand.Rand) Option {
	return func(k *Kmeans) {
		k.RandomState = randomState
	}
}

// WithRandomSeed sets the random seed (creates new random state)
func WithRandomSeed(seed int64) Option {
	return func(k *Kmeans) {
		k.RandomState = rand.New(rand.NewSource(seed))
	}
}

// WithNCentroidsInitTrials sets the number of trials for centroid initialization (used in k-means++)
func WithNCentroidsInitTrials(n int) Option {
	return func(k *Kmeans) {
		if n > 0 {
			k.NCentroidsInitTrials = n
		}
	}
}

// Validate checks if the Kmeans configuration is valid
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

	if k.Tol <= 0 {
		return ErrInvalidTol
	}

	if k.NCentroidsInitTrials <= 0 {
		return ErrInvalidNCentroidsInitTrials
	}

	if k.RandomState == nil {
		return ErrInvalidRandomState
	}

	return nil
}

// Cluster performs K-means clustering on the given data
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
	var (
		bestResult *Result
		bestLabels []int
	)

	// Run the clustering algorithm multiple times and return the best result
	for range k.NInit {
		result := k.lloydKMeans(data, labels)
		if bestResult == nil || result.Inertia < bestResult.Inertia {
			bestResult = result
			bestLabels = append(bestLabels[:0], labels...)
		}
	}

	bestResult.Labels = bestLabels

	return bestResult, nil
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

	// Validate that all data points have the same dimensions
	if len(data) > 0 {
		expectedDim := len(data[0])
		for i, point := range data {
			if len(point) != expectedDim {
				return fmt.Errorf("data point at index %d has dimension %d, expected %d", i, len(point), expectedDim)
			}
		}
	}

	if k.NClusters > len(data) {
		return fmt.Errorf("number of clusters (%d) cannot be greater than number of samples (%d)", k.NClusters, len(data))
	}

	return nil
}

// initRandomCentroids initializes cluster centroids by randomly selecting data points
func (k *Kmeans) initRandomCentroids(data [][]float64) [][]float64 {
	centroids := make([][]float64, 0, k.NClusters)
	usedIndices := make(map[int]struct{})

	for len(centroids) < k.NClusters {
		randomIndex := k.RandomState.Intn(len(data))
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
	nLocalTrials := k.NCentroidsInitTrials

	centroids := make([][]float64, 0, k.NClusters)

	// Track already selected indices to avoid duplicates
	usedIndices := make(map[int]struct{})

	// Select first centroid randomly
	firstCenterIdx := k.RandomState.Intn(len(data))
	centroids = append(centroids, make([]float64, len(data[firstCenterIdx])))
	copy(centroids[0], data[firstCenterIdx])
	usedIndices[firstCenterIdx] = struct{}{}

	distances := make([]float64, len(data))

	// Select remaining centroids
	for len(centroids) < k.NClusters {
		// Compute squared distances to nearest centroids for each data point
		distances = computeDistancesToCenters(data, centroids, distances)

		// When nLocalTrials <= 1, use standard k-means++ initialization
		if nLocalTrials <= 1 {
			// Select next centroid with probability proportional to squared distance
			nextCentroidIdx := selectUniqueIndex(distances, usedIndices, k.RandomState)

			newCentroid := make([]float64, len(data[nextCentroidIdx]))
			copy(newCentroid, data[nextCentroidIdx])
			centroids = append(centroids, newCentroid)
			usedIndices[nextCentroidIdx] = struct{}{}
		} else {
			// Use greedy k-means++ initialization with nLocalTrials
			bestCentroidIdx := -1
			bestInertia := math.Inf(1)

			for range nLocalTrials {
				// Select candidate centroid with probability proportional to squared distance
				candidateIdx := selectUniqueIndex(distances, usedIndices, k.RandomState)

				// Create temporary list of centroids with this candidate
				tempCentroids := make([][]float64, len(centroids)+1)
				copy(tempCentroids, centroids)
				tempCentroids[len(centroids)] = make([]float64, len(data[candidateIdx]))
				copy(tempCentroids[len(centroids)], data[candidateIdx])

				// Calculate inertia with this candidate
				inertia := calculateInertia(data, tempCentroids)

				// If this candidate is better than the best so far, update the best centroid
				if inertia < bestInertia {
					bestInertia = inertia
					bestCentroidIdx = candidateIdx
				}
			}

			// Add the best centroid to the list of centroids
			if bestCentroidIdx != -1 {
				newCentroid := make([]float64, len(data[bestCentroidIdx]))
				copy(newCentroid, data[bestCentroidIdx])
				centroids = append(centroids, newCentroid)
				usedIndices[bestCentroidIdx] = struct{}{}
			} else {
				// Fallback: select a random unused data point as centroid
				fallbackIdx := selectUniqueIndex(distances, usedIndices, k.RandomState)
				newCentroid := make([]float64, len(data[fallbackIdx]))
				copy(newCentroid, data[fallbackIdx])
				centroids = append(centroids, newCentroid)
				usedIndices[fallbackIdx] = struct{}{}
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
	var centers [][]float64

	switch k.Init {
	case InitRandom:
		centers = k.initRandomCentroids(data)
	case InitKMeansPlusPlus:
		centers = k.initKMeansPlusPlusCentroids(data)
	default:
		centers = k.initKMeansPlusPlusCentroids(data)
	}

	return centers
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
	return k.lloydKMeans(data, labels)
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
//	A pointer to Result containing final centroids, labels, and inertia.
func (k *Kmeans) lloydKMeans(data [][]float64, labels []int) *Result {
	centers := k.initCentroids(data)

	dim := len(data[0])
	next := newCenterBuffer(k.NClusters, dim)
	counts := make([]int, k.NClusters)

	for range k.MaxIter {
		// Step 1: Assign each data point to the nearest cluster center
		labels = assignPointsToClusters(data, centers, labels)

		// Step 2: Update cluster centers based on current assignments (Lloyd's update step)
		// This computes the mean of all points assigned to each cluster.
		clearCenters(next)
		clear(counts)
		k.updateCentersLloydInto(next, counts, data, labels)

		// Check for convergence against the previous centers before replacing them
		converged := checkConvergence(centers, next, k.Tol)

		// Swap buffers: the just-computed centers become the current ones for the next iteration
		centers, next = next, centers

		if converged {
			break
		}
	}

	// Assign points to clusters again to ensure final assignments
	labels = assignPointsToClusters(data, centers, labels)

	finalInertia := calculateInertiaByLabels(data, centers, labels)

	return &Result{
		Centroids: centers,
		Labels:    labels,
		Inertia:   finalInertia,
	}
}

// updateCentersLloyd computes new cluster centers (centroids) for the Lloyd's algorithm step.
//
// For each cluster, it calculates the mean of all points assigned to that cluster.
// If a cluster has no assigned points, its center will remain a zero vector.
//
// Parameters:
//
//	data:   The dataset, where each element is a point (slice of float64).
//	labels: Cluster assignments for each point in data. labels[i] is the cluster index for data[i].
//
// Returns:
//
//	A slice of new cluster centers, where each center is a slice of float64.
func (k *Kmeans) updateCentersLloyd(data [][]float64, labels []int) [][]float64 {
	dim := len(data[0])
	newCenters := newCenterBuffer(k.NClusters, dim)

	k.updateCentersLloydInto(newCenters, make([]int, k.NClusters), data, labels)

	return newCenters
}

// updateCentersLloydInto computes new cluster centers (centroids) into the provided dst buffer
// for the Lloyd's algorithm step. dst must have k.NClusters rows of len(data[0]) columns and
// must be zeroed before the call, as must be counts. For each cluster, it calculates the mean of
// all points assigned to that cluster. If a cluster has no assigned points, its center remains a
// zero vector.
func (k *Kmeans) updateCentersLloydInto(dst [][]float64, counts []int, data [][]float64, labels []int) {
	dim := len(data[0])

	for i, point := range data {
		cluster := labels[i]
		for d := range dim {
			dst[cluster][d] += point[d]
		}
		counts[cluster]++
	}

	for i := 0; i < k.NClusters; i++ {
		if counts[i] > 0 {
			for d := range dim {
				dst[i][d] /= float64(counts[i])
			}
		}
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
		minDistance := math.Inf(1)
		bestCluster := 0

		// Find the nearest center for each point
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

// pointsEqual compares two points (slices of float64) and returns true if they are equal within a given tolerance.
//
// Parameters:
//
//	p1:        The first point, represented as a slice of float64.
//	p2:        The second point, represented as a slice of float64.
//	tolerance: The maximum allowed absolute difference between corresponding coordinates for the points to be considered equal.
//
// Returns:
//
//	true if the points are of equal length and each corresponding coordinate differs by no more than the specified tolerance; false otherwise.
func pointsEqual(p1, p2 []float64, tolerance float64) bool {
	if len(p1) != len(p2) {
		return false
	}

	for i := range p1 {
		if math.Abs(p1[i]-p2[i]) > tolerance {
			return false
		}
	}

	return true
}

// checkConvergence determines whether the cluster centers have converged by comparing the old and new centers.
//
// Parameters:
//
//	oldCenters: A slice of previous cluster centers, where each center is a slice of float64 features.
//	newCenters: A slice of updated cluster centers, where each center is a slice of float64 features.
//	tolerance:  The maximum allowed absolute difference between corresponding coordinates for centers to be considered equal.
//
// Returns:
//
//	true if all corresponding centers are equal within the specified tolerance; false otherwise.
func checkConvergence(oldCenters, newCenters [][]float64, tolerance float64) bool {
	if len(oldCenters) != len(newCenters) {
		return false
	}

	for i := range oldCenters {
		if !pointsEqual(oldCenters[i], newCenters[i], tolerance) {
			return false
		}
	}

	return true
}
