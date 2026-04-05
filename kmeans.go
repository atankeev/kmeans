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

// Algorithm defines the clustering algorithm to use
type Algorithm string

const (
	AlgorithmLloyd = "lloyd"
	AlgorithmElkan = "elkan"
)

var (
	supportedAlgorithms = map[Algorithm]struct{}{
		AlgorithmLloyd: {},
		AlgorithmElkan: {},
	}
)

// elkanState хранит состояние алгоритма Elkan для каждой точки
type elkanState struct {
	// upperBounds[i] - верхняя граница расстояния от точки i до её текущего центра
	upperBounds []float64
	// lowerBounds - нижняя граница расстояния от точки i до центра j (flat: i*k + j)
	lowerBounds []float64
	// assignments[i] - текущий кластер точки i
	assignments []int
	// centerDistances - расстояние между центрами i и j (flat: i*k + j)
	centerDistances []float64
	// centerMovement[i] - расстояние, на которое переместился центр i
	centerMovement []float64
	// n - количество точек (для вычисления индексов)
	n int
	// k - количество кластеров (для вычисления индексов)
	k int
	// oldCenters - старые центры для проверки сходимости (переиспользуется между итерациями)
	oldCenters [][]float64
	// newCenters - новые центры (переиспользуется между итерациями)
	newCenters [][]float64
	// dim - размерность данных (для переиспользования памяти)
	dim int
}

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

	// Algorithm to use for clustering
	Algorithm Algorithm

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
		Algorithm:            AlgorithmLloyd,
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

// WithAlgorithm sets the algorithm to use for clustering
func WithAlgorithm(algorithm Algorithm) Option {
	return func(k *Kmeans) {
		k.Algorithm = algorithm
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

	if _, ok := supportedAlgorithms[k.Algorithm]; !ok {
		return ErrInvalidAlgorithm
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

	var (
		bestResult *Result
	)

	// Run the clustering algorithm multiple times and return the best result
	for range k.NInit {
		result := k.clusterSingle(data)
		if bestResult == nil || result.Inertia < bestResult.Inertia {
			bestResult = result
		}
	}

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

	// Select remaining centroids
	for len(centroids) < k.NClusters {
		// Compute squared distances to nearest centroids for each data point
		distances := computeDistancesToCenters(data, centroids)

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

// clusterSingle performs K-means clustering using the selected algorithm.
//
// This function performs the main loop of the selected algorithm, which iteratively assigns
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
	switch k.Algorithm {
	case AlgorithmLloyd:
		return k.lloydKMeans(data)
	case AlgorithmElkan:
		return k.elkanKMeans(data)
	default:
		return k.lloydKMeans(data)
	}
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
func (k *Kmeans) lloydKMeans(data [][]float64) *Result {
	centers := k.initCentroids(data)

	var (
		labels []int
	)

	for range k.MaxIter {
		// Step 1: Assign each data point to the nearest cluster center
		newLabels := assignPointsToClusters(data, centers)

		// Step 2: Update cluster centers based on current assignments (Lloyd's update step)
		// This computes the mean of all points assigned to each cluster.
		newCenters := k.updateCentersLloyd(data, newLabels)

		// Update centers and labels
		centers = newCenters
		labels = newLabels

		// Check for convergence
		converged := checkConvergence(centers, newCenters, k.Tol)
		if converged {
			break
		}
	}

	// Assign points to clusters again to ensure final assignments
	labels = assignPointsToClusters(data, centers)

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
	newCenters := make([][]float64, k.NClusters)
	clusterSizes := make([]int, k.NClusters)

	for i := 0; i < k.NClusters; i++ {
		newCenters[i] = make([]float64, dim)
	}

	for i, point := range data {
		cluster := labels[i]
		for d := range dim {
			newCenters[cluster][d] += point[d]
		}
		clusterSizes[cluster]++
	}

	for i := 0; i < k.NClusters; i++ {
		if clusterSizes[i] > 0 {
			for d := range dim {
				newCenters[i][d] /= float64(clusterSizes[i])
			}
		}
	}

	return newCenters
}

// elkanKMeans implements the Elkan k-means algorithm.
//
// This function performs the main loop of the Elkan algorithm, which iteratively assigns
// each data point to the nearest cluster center, updates the centers, and checks for convergence.
//
// Parameters:
//
//	data: The dataset, where each element is a point (slice of float64).
//
// Returns:
//
//	A pointer to Result containing final centroids, labels, and inertia.
func (k *Kmeans) elkanKMeans(data [][]float64) *Result {
	centers := k.initCentroids(data)

	// Initialize the Elkan state
	state := initializeElkanState(data, centers)

	iteration := 0

	// Main loop of the Elkan algorithm
	for iteration < k.MaxIter {
		// Assignment step with Elkan optimizations
		changed := elkanAssignStep(data, centers, state)

		if !changed && iteration > 0 {
			// Convergence reached: points don't change clusters
			break
		}

		// Save the old centers (reusing pre-allocated slice)
		for i, center := range centers {
			copy(state.oldCenters[i], center)
		}

		// Update the centers (reusing pre-allocated slice)
		newCenters := k.updateCentersElkan(data, state, state.newCenters)
		centers = newCenters

		// Update the Elkan state
		updateElkanState(state.oldCenters, newCenters, state)

		iteration++

		// Check for convergence of the centers
		converged := checkConvergence(state.oldCenters, newCenters, k.Tol)
		if converged {
			// Convergence reached: the centers have stabilized
			break
		}
	}

	finalInertia := calculateInertiaByLabels(data, centers, state.assignments)

	return &Result{
		Centroids: centers,
		Labels:    state.assignments,
		Inertia:   finalInertia,
	}
}

// updateCentersElkan recalculates the centers and tracks their movement.
//
// This function calculates the mean of all points assigned to each cluster and updates the centers.
//
// Parameters:
//
//	data:   The dataset, where each element is a point (slice of float64).
//	state:  The current Elkan state, containing bounds and assignments for each point.
//
// Returns:
//
//	A slice of new cluster centers, where each center is a slice of float64.
func (k *Kmeans) updateCentersElkan(data [][]float64, state *elkanState, reuse [][]float64) [][]float64 {
	dim := state.dim
	if dim == 0 {
		dim = len(data[0])
	}

	newCenters := reuse
	if newCenters == nil || len(newCenters) != k.NClusters {
		newCenters = make([][]float64, k.NClusters)
		for i := range k.NClusters {
			newCenters[i] = make([]float64, dim)
		}
	}

	clusterSizes := make([]int, k.NClusters)

	// Reset centers to zero
	for i := range k.NClusters {
		for d := range dim {
			newCenters[i][d] = 0
		}
	}

	// Sum the coordinates of the points for each cluster
	for i, point := range data {
		cluster := state.assignments[i]
		for d := range dim {
			newCenters[cluster][d] += point[d]
		}
		clusterSizes[cluster]++
	}

	// Calculate the mean values
	for i := range k.NClusters {
		if clusterSizes[i] > 0 {
			for d := range dim {
				newCenters[i][d] /= float64(clusterSizes[i])
			}
		}
	}

	return newCenters
}

// updateElkanState updates the state of the Elkan algorithm after the centers have moved.
//
// This function calculates the distances the centers moved and updates the distances between centers.
// It also updates the upper and lower bounds for each point.
//
// Parameters:
//
//	oldCenters: The old cluster centers, where each center is a slice of float64.
//	newCenters: The new cluster centers, where each center is a slice of float64.
//	state:      The current Elkan state, containing bounds and assignments for each point.
func updateElkanState(oldCenters, newCenters [][]float64, state *elkanState) {
	k := state.k

	// Calculate the distances the centers moved
	for i := range k {
		state.centerMovement[i] = euclideanDistance(oldCenters[i], newCenters[i])
	}

	// Update the distances between centers
	for i := range k {
		for j := i + 1; j < k; j++ {
			dist := euclideanDistance(newCenters[i], newCenters[j])
			state.centerDistances[i*k+j] = dist
			state.centerDistances[j*k+i] = dist
		}
	}

	// Update the bounds for each point
	n := state.n
	for i := range n {
		// Update the upper bound
		currentCenter := state.assignments[i]
		state.upperBounds[i] += state.centerMovement[currentCenter]

		// Update the lower bounds
		for j := range k {
			idx := i*k + j
			state.lowerBounds[idx] = math.Max(0, state.lowerBounds[idx]-state.centerMovement[j])
		}
	}
}

// elkanAssignStep performs the assignment step of the Elkan k-means algorithm with optimizations.
//
// For each data point, this function uses upper and lower bounds, as well as the triangle inequality,
// to avoid unnecessary distance calculations and efficiently determine the nearest cluster center.
//
// Parameters:
//
//	data:    The dataset, where each element is a point (slice of float64).
//	centers: The current cluster centers, where each center is a slice of float64.
//	state:   The current Elkan state, containing bounds and assignments for each point.
//
// Returns:
//
//	true if any point changed its cluster assignment during this step; false otherwise.
func elkanAssignStep(data [][]float64, centers [][]float64, state *elkanState) bool {
	k := state.k
	changed := false

	for i, point := range data {
		currentCenter := state.assignments[i]

		// Optimization 1: If the upper bound is less than or equal to half of the minimum distance between centers,
		// the point will definitely stay in the current cluster
		minCenterDistance := math.Inf(1)
		for j := range k {
			if j == currentCenter {
				continue
			}

			cd := state.centerDistances[currentCenter*k+j]
			if cd < minCenterDistance {
				minCenterDistance = cd
			}
		}

		if state.upperBounds[i] <= minCenterDistance/2 {
			continue
		}

		// Recalculate the exact distance to the current center
		actualDistance := euclideanDistance(point, centers[currentCenter])
		state.upperBounds[i] = actualDistance
		state.lowerBounds[i*k+currentCenter] = actualDistance

		// Find the best center
		bestCenter := currentCenter
		bestDistance := actualDistance

		for j := range k {
			if j == currentCenter {
				continue
			}

			lbIdx := i*k + j
			cdIdx := currentCenter*k + j

			// Optimization 2: Use the triangle inequality
			if state.upperBounds[i] > state.lowerBounds[lbIdx] &&
				state.upperBounds[i] > state.centerDistances[cdIdx]/2 {

				// Calculate the exact distance only if the optimizations didn't work
				distance := euclideanDistance(point, centers[j])
				state.lowerBounds[lbIdx] = distance

				if distance < bestDistance {
					bestDistance = distance
					bestCenter = j
				}
			}
		}

		// Update the assignment if we found the best center
		if bestCenter != currentCenter {
			state.assignments[i] = bestCenter
			state.upperBounds[i] = bestDistance
			changed = true
		}
	}

	return changed
}

// initializeElkanState initializes the state required for the Elkan variant of the k-means algorithm.
//
// This function prepares upper and lower bounds, assignments, and center distances for each data point
// and cluster center, which are used to accelerate the assignment step in Elkan's algorithm.
//
// Parameters:
//
//	data:    The dataset, where each element is a point (slice of float64).
//	centers: The initial cluster centers, where each center is a slice of float64.
//
// Returns:
//
//	A pointer to an elkanState struct containing all necessary bounds and assignments for the algorithm.
func initializeElkanState(data [][]float64, centers [][]float64) *elkanState {
	n := len(data)
	k := len(centers)
	dim := len(data[0])

	state := &elkanState{
		upperBounds:     make([]float64, n),
		lowerBounds:     make([]float64, n*k),
		assignments:     make([]int, n),
		centerDistances: make([]float64, k*k),
		centerMovement:  make([]float64, k),
		n:               n,
		k:               k,
		dim:             dim,
		oldCenters:      make([][]float64, k),
		newCenters:      make([][]float64, k),
	}

	for i := range k {
		state.oldCenters[i] = make([]float64, dim)
		state.newCenters[i] = make([]float64, dim)
	}

	// Calculate initial distances between centers
	for i := range k {
		for j := i + 1; j < k; j++ {
			dist := euclideanDistance(centers[i], centers[j])
			state.centerDistances[i*k+j] = dist
			state.centerDistances[j*k+i] = dist
		}
	}

	// Initialize bounds for each point
	for i, point := range data {
		minDist := math.Inf(1)
		bestCenter := 0

		for j, center := range centers {
			dist := euclideanDistance(point, center)
			state.lowerBounds[i*k+j] = dist

			if dist < minDist {
				minDist = dist
				bestCenter = j
			}
		}

		state.upperBounds[i] = minDist
		state.assignments[i] = bestCenter
	}

	return state
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
func assignPointsToClusters(data [][]float64, centers [][]float64) []int {
	n := len(data)
	labels := make([]int, n)

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
		sum += diff * diff
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
			if dist < minDist {
				minDist = dist
			}
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
func computeDistancesToCenters(data [][]float64, centers [][]float64) []float64 {
	distances := make([]float64, len(data))

	for i, point := range data {
		minDist := math.Inf(1)
		for _, center := range centers {
			dist := squaredEuclideanDistance(point, center)
			if dist < minDist {
				minDist = dist
			}
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

// euclideanDistance computes the Euclidean distance between two points represented as slices of float64.
//
// Parameters:
//
//	p1: The first point, a slice of float64.
//	p2: The second point, a slice of float64.
//
// Returns:
//
//	The Euclidean distance between p1 and p2.
func euclideanDistance(p1, p2 []float64) float64 {
	return math.Sqrt(squaredEuclideanDistance(p1, p2))
}
