// Package kmeans clusters finite, equal-dimensional samples with Lloyd's k-means
// algorithm. Call Cluster(data, k, opts...) to obtain labels, centroids, inertia,
// and selected-run convergence and iteration metadata.
//
// Each call has private configuration and random state. Defaults use ten runs,
// up to 300 Lloyd iterations per run, relative tolerance 1e-4, and k-means++
// initialization. Empty clusters take the farthest eligible samples. Runs are
// ranked by internal inertia; equal inertias retain the first run. Reaching the
// iteration limit is successful non-convergence; numerical overflow is an error.
package kmeans

import (
	"cmp"
	"fmt"
	"math"
	"math/rand"
	"slices"
)

// Result describes the initialization run with the lowest internal inertia.
type Result struct {
	// Labels contains one cluster index per sample. Sample i is assigned to
	// Centroids[Labels[i]].
	Labels []int
	// Centroids contains final cluster centers in independent rows. Tolerance
	// convergence does not guarantee exact means of the final label groups.
	Centroids [][]float64
	// Inertia is the sum of squared distances to assigned centroids. A tiny
	// positive inertia may round to zero; zero does not prove exact coincidence.
	Inertia float64
	// Converged reports whether the selected run satisfied a convergence check.
	Converged bool
	// Iterations counts Lloyd iterations executed by the selected run, including
	// the one establishing convergence. Initialization and final reassignment
	// are excluded; convergence at the iteration limit is valid.
	Iterations int
}

// Cluster clusters the entire dataset into k clusters. Data must be nonempty,
// have equal positive sample dimensions and finite coordinates, and satisfy
// 1 <= k <= len(data). Options apply in order; the last value for a setting wins.
// Final settings are validated without replacing invalid values with defaults.
//
// Each call owns its configuration and random state. Built-in options and
// read-only input may be shared across concurrent calls. Input is not modified;
// output storage is independent of input and other results. Callers must not
// mutate input during a call. This API does not support cancellation.
//
// Explicit seeds, including zero, reproduce results for identical ordered data
// and options within the same library version and execution environment. An
// omitted seed is selected internally for each call.
//
// Runs are ranked by internal inertia before public float64 conversion. Equal
// internal inertias retain the first run. Metadata describes the selected run;
// reaching its iteration limit returns a usable non-converged result with nil
// error. Invalid options or data and numerical overflow return a nil result.
// Overflow in any run aborts the call, even after an earlier usable result.
// Identify errors with errors.Is and exported sentinels; diagnostic wording
// is not a stable contract.
func Cluster(data [][]float64, nClusters int, opts ...Option) (*Result, error) {
	cfg := newConfig(nClusters)
	for _, option := range opts {
		builtIn, ok := option.(optionFunc)
		if !ok || builtIn == nil {
			return nil, ErrInvalidOption
		}
		builtIn.apply(cfg)
	}
	if err := cfg.validate(); err != nil {
		return nil, fmt.Errorf("invalid configuration: %w", err)
	}
	if err := validateData(data, nClusters); err != nil {
		return nil, fmt.Errorf("invalid data: %w", err)
	}
	algorithm := kmeans{config: *cfg}
	return algorithm.cluster(data)
}

// kmeans executes clustering with a validated, per-call configuration.
type kmeans struct {
	config config
}

func (k *kmeans) cluster(data [][]float64) (*Result, error) {
	labels := make([]int, len(data))
	previousLabels := make([]int, len(data))
	distances := make([]scaledValue, len(data))
	randomState := k.newRandomState()
	tolerance := calculateScaledTolerance(data, k.config.Tol)
	var (
		bestResult  *Result
		bestInertia scaledValue
		bestLabels  []int
	)
	for range k.config.NInit {
		result, inertia, _, err := k.lloydKMeans(
			data, labels, previousLabels, distances, randomState, tolerance,
		)
		if err != nil {
			return nil, err
		}
		if !resultIsFinite(result) {
			return nil, fmt.Errorf("clustering produced a non-finite result: %w", ErrNumericalOverflow)
		}
		if bestResult == nil || inertia.compare(bestInertia) < 0 {
			bestResult = result
			bestInertia = inertia
			bestLabels = append(bestLabels[:0], labels...)
		}
	}
	bestResult.Labels = bestLabels
	return bestResult, nil
}

func (k *kmeans) newRandomState() *rand.Rand {
	return rand.New(rand.NewSource(k.config.RandomSeed))
}

// initRandomCentroids copies samples at distinct randomly selected indexes.
func (k *kmeans) initRandomCentroids(data [][]float64, randomState *rand.Rand) [][]float64 {
	centroids := make([][]float64, 0, k.config.NClusters)
	usedIndices := make(map[int]struct{}, k.config.NClusters)

	for len(centroids) < k.config.NClusters {
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

// initKMeansPlusPlusCentroidsWithDistances initializes cluster centroids using the k-means++
// algorithm, reusing a caller-provided distances buffer (allocated once per Cluster call).
// The buffer is reused when len(distances) == len(data); otherwise a new buffer is
// allocated. It is re-initialized for each initialization run.
func (k *kmeans) initKMeansPlusPlusCentroidsWithDistances(
	data [][]float64,
	distances []scaledValue,
	randomState *rand.Rand,
) [][]float64 {
	nLocalTrials := k.config.NCentroidsInitTrials

	centroids := make([][]float64, 0, k.config.NClusters)

	usedIndices := make(map[int]struct{}, k.config.NClusters)

	if len(distances) != len(data) {
		distances = make([]scaledValue, len(data))
	}
	clear(distances)

	// Adds a data point as a centroid and updates the nearest-centroid distances.
	addCentroid := func(idx int) {
		centroid := make([]float64, len(data[idx]))
		copy(centroid, data[idx])
		centroids = append(centroids, centroid)
		usedIndices[idx] = struct{}{}
		updateScaledMinDistances(distances, data, centroid, len(centroids) == 1)
	}

	addCentroid(randomState.Intn(len(data)))

	for len(centroids) < k.config.NClusters {
		// A single trial uses the standard distance-weighted k-means++ choice;
		// multiple trials choose the candidate with the lowest resulting inertia.
		if nLocalTrials <= 1 {
			addCentroid(selectUniqueScaledIndex(distances, usedIndices, randomState))
			continue
		}

		bestCentroidIdx := -1
		var bestInertia scaledValue

		for range nLocalTrials {
			candidateIdx := selectUniqueScaledIndex(distances, usedIndices, randomState)

			// Calculate the inertia with this candidate without materializing the
			// candidate as a centroid: min(existing distance, distance to candidate)
			inertia := scaledCandidateCost(distances, data, data[candidateIdx])

			if bestCentroidIdx == -1 || inertia.compare(bestInertia) < 0 {
				bestInertia = inertia
				bestCentroidIdx = candidateIdx
			}
		}

		if bestCentroidIdx != -1 {
			addCentroid(bestCentroidIdx)
		} else {
			// If no candidate was selected, retain progress with an unused sample.
			addCentroid(selectUniqueScaledIndex(distances, usedIndices, randomState))
		}
	}

	return centroids
}

// initCentroidsWithDistances initializes the cluster centroids using the selected initialization
// method, reusing a caller-provided distances buffer for the k-means++ method. The buffer is
// ignored by initialization methods that do not need it.
func (k *kmeans) initCentroidsWithDistances(
	data [][]float64,
	distances []scaledValue,
	randomState *rand.Rand,
) [][]float64 {
	switch k.config.Init {
	case InitRandom:
		return k.initRandomCentroids(data, randomState)
	default:
		return k.initKMeansPlusPlusCentroidsWithDistances(data, distances, randomState)
	}
}

// lloydKMeans performs one initialization run using Lloyd's algorithm, reusing
// the labels, previousLabels, and distances buffers. It uses randomState to
// initialize centroids and tolerance as the squared center-shift threshold.
//
// It returns the final result (whose Labels shares the supplied labels buffer),
// inertia before public conversion, and convergence status.
// Exhausting the iteration limit without convergence returns a usable result,
// false, and nil error. Numerical overflow returns a nil result and an error
// wrapping ErrNumericalOverflow.
func (k *kmeans) lloydKMeans(
	data [][]float64,
	labels []int,
	previousLabels []int,
	distances []scaledValue,
	randomState *rand.Rand,
	tolerance scaledValue,
) (*Result, scaledValue, bool, error) {
	centers := k.initCentroidsWithDistances(data, distances, randomState)
	clear(labels)

	dim := len(data[0])
	next := newCenterBuffer(k.config.NClusters, dim)
	counts := make([]int, k.config.NClusters)
	hasPreviousLabels := false
	converged := false
	iterations := 0

	for range k.config.MaxIter {
		iterations++
		labels = assignPointsToClusters(data, centers, labels)

		// Empty clusters take the farthest assigned samples, ordered deterministically by
		// descending assignment error and then ascending sample index.
		clearCenters(next)
		clear(counts)
		k.updateCentersLloydInto(next, counts, distances, data, labels, centers)
		if !centersAreFinite(next) {
			return nil, scaledValue{}, false, fmt.Errorf("update centroids: %w", ErrNumericalOverflow)
		}

		// As in sklearn, unchanged assignments take precedence over the
		// tolerance-based squared Frobenius norm check.
		labelsUnchanged := hasPreviousLabels && slices.Equal(labels, previousLabels)
		converged = labelsUnchanged || checkScaledConvergence(centers, next, tolerance)

		centers, next = next, centers

		if converged {
			break
		}

		copy(previousLabels, labels)
		hasPreviousLabels = true
	}

	// Reassignment makes labels correspond to the final centers after the last update.
	labels = assignPointsToClusters(data, centers, labels)

	inertia := calculateScaledInertiaByLabels(data, centers, labels)
	finalInertia, representable := inertia.toFloat64()
	if !representable {
		return nil, scaledValue{}, false, fmt.Errorf("calculate inertia: %w", ErrNumericalOverflow)
	}

	return &Result{
		Centroids:  centers,
		Labels:     labels,
		Inertia:    finalInertia,
		Converged:  converged,
		Iterations: iterations,
	}, inertia, converged, nil
}

// updateCentersLloydInto computes new cluster centers (centroids) into the provided dst buffer
// for the Lloyd's algorithm step. dst must have k.config.NClusters rows of len(data[0]) columns and
// must be zeroed before the call, as must be counts. For each cluster, it calculates the mean of
// all points assigned to that cluster. Empty clusters take distinct farthest samples from clusters
// that can donate a sample without becoming empty themselves.
func (k *kmeans) updateCentersLloydInto(
	dst [][]float64,
	counts []int,
	distances []scaledValue,
	data [][]float64,
	labels []int,
	centers [][]float64,
) {
	for i := range data {
		cluster := labels[i]
		counts[cluster]++
	}

	k.relocateEmptyClusters(counts, distances, data, labels, centers)

	seen := make([]int, k.config.NClusters)
	dim := len(data[0])
	for i, point := range data {
		cluster := labels[i]
		seen[cluster]++
		for dimension := range dim {
			dst[cluster][dimension] = updateMean(
				dst[cluster][dimension],
				point[dimension],
				seen[cluster],
			)
		}
	}
}

// relocateEmptyClusters moves distinct samples into empty clusters. Candidates are ranked by
// descending squared distance to their currently assigned center, with lower sample indexes first
// on ties. A source cluster must retain at least one sample.
func (k *kmeans) relocateEmptyClusters(
	counts []int,
	distances []scaledValue,
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
		distances[i] = scaledSquaredDistance(point, centers[labels[i]])
	}
	slices.SortFunc(candidates, func(a, b int) int {
		if byDistance := distances[b].compare(distances[a]); byDistance != 0 {
			return byDistance
		}

		return cmp.Compare(a, b)
	})

	nextCandidate := 0
	for emptyCluster := 0; emptyCluster < k.config.NClusters; emptyCluster++ {
		if counts[emptyCluster] != 0 {
			continue
		}

		for counts[labels[candidates[nextCandidate]]] <= 1 {
			nextCandidate++
		}

		candidate := candidates[nextCandidate]
		nextCandidate++
		sourceCluster := labels[candidate]
		counts[sourceCluster]--
		counts[emptyCluster]++
		labels[candidate] = emptyCluster
	}
}

func resultIsFinite(result *Result) bool {
	if math.IsNaN(result.Inertia) || math.IsInf(result.Inertia, 0) {
		return false
	}

	return centersAreFinite(result.Centroids)
}

// validateData rejects empty data, inconsistent dimensions, non-finite coordinates,
// missing features, and a cluster count greater than the number of samples.
func validateData(data [][]float64, nClusters int) error {
	if len(data) == 0 {
		return ErrEmptyData
	}

	expectedDim := len(data[0])
	for pointIndex, point := range data {
		if len(point) != expectedDim {
			return fmt.Errorf(
				"%w: data point at index %d has dimension %d, expected %d",
				ErrInvalidData,
				pointIndex,
				len(point),
				expectedDim,
			)
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

	if nClusters > len(data) {
		return fmt.Errorf(
			"%w: number of clusters (%d) cannot be greater than number of samples (%d)",
			ErrInvalidK,
			nClusters,
			len(data),
		)
	}

	return nil
}

func centersAreFinite(centers [][]float64) bool {
	for _, center := range centers {
		for _, coordinate := range center {
			if math.IsNaN(coordinate) || math.IsInf(coordinate, 0) {
				return false
			}
		}
	}

	return true
}

// newCenterBuffer allocates a flat-backed buffer with nClusters rows of dim columns.
func newCenterBuffer(nClusters, dim int) [][]float64 {
	buf := make([][]float64, nClusters)
	backing := make([]float64, nClusters*dim)
	for i := 0; i < nClusters; i++ {
		buf[i] = backing[i*dim : (i+1)*dim : (i+1)*dim]
	}
	return buf
}

// clearCenters zeroes all values in the buffer.
func clearCenters(buf [][]float64) {
	for _, center := range buf {
		clear(center)
	}
}

// assignPointsToClusters writes the nearest center's index for each data point
// into labels and returns the same slice. The labels slice must have at least
// len(data) entries. A valid existing label is retained when its center ties
// for the minimum squared Euclidean distance.
func assignPointsToClusters(data [][]float64, centers [][]float64, labels []int) []int {
	for i, point := range data {
		bestCluster := labels[i]
		if bestCluster < 0 || bestCluster >= len(centers) {
			bestCluster = 0
		}
		minDistance := scaledSquaredDistance(point, centers[bestCluster])

		// Keeping the current label on exact ties lets recovered clusters remain
		// populated when samples or centers are duplicates.
		for j, center := range centers {
			distance := scaledSquaredDistance(point, center)
			if distance.compare(minDistance) < 0 {
				minDistance = distance
				bestCluster = j
			}
		}

		labels[i] = bestCluster
	}

	return labels
}

func calculateScaledInertiaByLabels(data [][]float64, centers [][]float64, labels []int) scaledValue {
	var inertia scaledValue
	for i, point := range data {
		inertia = inertia.add(scaledSquaredDistance(point, centers[labels[i]]))
	}

	return inertia
}

func updateScaledMinDistances(
	distances []scaledValue,
	data [][]float64,
	center []float64,
	initialize bool,
) {
	for i, point := range data {
		distance := scaledSquaredDistance(point, center)
		if initialize || distance.compare(distances[i]) < 0 {
			distances[i] = distance
		}
	}
}

func scaledCandidateCost(distances []scaledValue, data [][]float64, candidate []float64) scaledValue {
	var total scaledValue
	for i, point := range data {
		distance := scaledSquaredDistance(point, candidate)
		if distances[i].compare(distance) < 0 {
			distance = distances[i]
		}
		total = total.add(distance)
	}

	return total
}

func scaledWeightedRandomChoice(weights []scaledValue, rng *rand.Rand) int {
	var largest scaledValue
	for _, weight := range weights {
		if weight.compare(largest) > 0 {
			largest = weight
		}
	}
	if largest.fraction == 0 {
		return rng.Intn(len(weights))
	}

	total := 0.0
	for _, weight := range weights {
		total += weight.ratio(largest)
	}

	selection := rng.Float64() * total
	cumulative := 0.0
	for i, weight := range weights {
		cumulative += weight.ratio(largest)
		if selection <= cumulative {
			return i
		}
	}

	return len(weights) - 1
}

func selectUniqueScaledIndex(
	weights []scaledValue,
	usedIndices map[int]struct{},
	rng *rand.Rand,
) int {
	for {
		selectedIndex := scaledWeightedRandomChoice(weights, rng)
		if _, alreadyUsed := usedIndices[selectedIndex]; !alreadyUsed {
			return selectedIndex
		}
	}
}

func calculateScaledTolerance(data [][]float64, relativeTolerance float64) scaledValue {
	dim := len(data[0])
	means := make([]float64, dim)
	for sampleIndex, point := range data {
		for feature, value := range point {
			means[feature] = updateMean(means[feature], value, sampleIndex+1)
		}
	}

	var varianceSum scaledValue
	for _, point := range data {
		varianceSum = varianceSum.add(scaledSquaredDistance(point, means))
	}

	return varianceSum.
		divide(float64(len(data))).
		divide(float64(dim)).
		multiply(relativeTolerance)
}

func checkScaledConvergence(oldCenters, newCenters [][]float64, tolerance scaledValue) bool {
	if len(oldCenters) != len(newCenters) {
		return false
	}

	var centerShift scaledValue
	for i := range oldCenters {
		if len(oldCenters[i]) != len(newCenters[i]) {
			return false
		}

		centerShift = centerShift.add(scaledSquaredDistance(oldCenters[i], newCenters[i]))
	}

	return centerShift.compare(tolerance) <= 0
}
