package kmeans_test

import (
	"encoding/binary"
	"errors"
	"math"
	"slices"
	"testing"

	"github.com/atankeev/kmeans"
)

const maxFuzzBytes = 1024

// The first seven bytes select bounded dimensions and options. Remaining bytes
// encode little-endian float64 coordinates; missing bytes are zero padded.
// Non-finite encodings become finite in the finite-input target. The invalid
// target adds one defect to this finite base, so its error oracle does not copy
// the library's validation algorithm or depend on validation precedence.
type fuzzCase struct {
	data                        [][]float64
	k, runs, iterations, trials int
	tolerance                   float64
}

func decodeFuzzCase(encoded []byte) fuzzCase {
	var header [7]byte
	copy(header[:], encoded)
	n, d := 1+int(header[0]%16), 1+int(header[1]%4)
	c := fuzzCase{
		k: 1 + int(header[2]%byte(min(n, 4))), runs: 1 + int(header[3]%3),
		iterations: 1 + int(header[4]%10), trials: 1 + int(header[5]%3),
		tolerance: []float64{1e-4, math.SmallestNonzeroFloat64, math.MaxFloat64}[header[6]%3],
		data:      make([][]float64, n),
	}
	for i := range c.data {
		c.data[i] = make([]float64, d)
		for j := range c.data[i] {
			var coordinate [8]byte
			offset := len(header) + 8*(i*d+j)
			if offset < len(encoded) {
				copy(coordinate[:], encoded[offset:])
			}
			bits := binary.LittleEndian.Uint64(coordinate[:])
			if bits&0x7ff0000000000000 == 0x7ff0000000000000 {
				bits ^= 0x0010000000000000
			}
			c.data[i][j] = math.Float64frombits(bits)
		}
	}
	return c
}

func (c fuzzCase) options(method kmeans.InitMethod) []kmeans.Option {
	return []kmeans.Option{
		kmeans.WithSeed(42), kmeans.WithInitMethod(method), kmeans.WithRuns(c.runs),
		kmeans.WithMaxIterations(c.iterations), kmeans.WithTolerance(c.tolerance),
		kmeans.WithNCentroidsInitTrials(c.trials),
	}
}

func fuzzSeed(header [7]byte, coordinates ...float64) []byte {
	encoded := append([]byte(nil), header[:]...)
	for _, coordinate := range coordinates {
		encoded = binary.LittleEndian.AppendUint64(encoded, math.Float64bits(coordinate))
	}
	return encoded
}

func FuzzClusterFinite(f *testing.F) {
	f.Add(fuzzSeed([7]byte{3, 1, 1, 2, 9, 2}, 0, 1, 2, 3, 8, 9, 10, 11))
	f.Add(fuzzSeed([7]byte{3, 0, 3}, 7, 7, 7, 7)) // Duplicate points and k == n.
	f.Add(fuzzSeed([7]byte{1}, 0, math.Copysign(0, -1)))
	f.Add(fuzzSeed([7]byte{1, 0, 0, 0, 0, 0, 1}, math.SmallestNonzeroFloat64, -math.SmallestNonzeroFloat64))
	f.Add(fuzzSeed([7]byte{1}, math.MaxFloat64, -math.MaxFloat64))
	f.Add(fuzzSeed([7]byte{15, 3, 3, 2, 9, 2, 2}, math.MaxFloat64))
	f.Add([]byte{})
	f.Fuzz(func(t *testing.T, encoded []byte) {
		// Reject before decoder allocations, including oversized corpus entries.
		if len(encoded) > maxFuzzBytes {
			return
		}
		c := decodeFuzzCase(encoded)
		for _, method := range []kmeans.InitMethod{kmeans.InitKMeansPlusPlus, kmeans.InitRandom} {
			checkFuzzCalls(t, c, c.options(method), nil)
		}
	})
}

func cloneFuzzData(data [][]float64) [][]float64 {
	if data == nil {
		return nil
	}
	clone := make([][]float64, len(data))
	for i := range data {
		clone[i] = slices.Clone(data[i])
	}
	return clone
}

func equalFuzzData(a, b [][]float64) bool {
	if (a == nil) != (b == nil) || len(a) != len(b) {
		return false
	}
	for i := range a {
		if (a[i] == nil) != (b[i] == nil) || len(a[i]) != len(b[i]) {
			return false
		}
		for j := range a[i] {
			if math.Float64bits(a[i][j]) != math.Float64bits(b[i][j]) {
				return false
			}
		}
	}
	return true
}

func checkFuzzCalls(t *testing.T, c fuzzCase, opts []kmeans.Option, wantErr error) {
	t.Helper()
	before := cloneFuzzData(c.data)
	var previous *kmeans.Result
	var previousErr error
	for call := range 2 {
		result, err := kmeans.Cluster(c.data, c.k, opts...)
		if !equalFuzzData(before, c.data) {
			t.Fatal("Cluster modified input (including coordinate bits or shape)")
		}
		if wantErr != nil {
			if !errors.Is(err, wantErr) {
				t.Fatalf("expected validation category %v, got %v", wantErr, err)
			}
		} else if err != nil && !errors.Is(err, kmeans.ErrNumericalOverflow) {
			t.Fatalf("finite input returned unexpected error: %v", err)
		}
		if err != nil {
			if result != nil {
				t.Fatal("error returned a non-nil result")
			}
		} else {
			checkFuzzResult(t, result, c)
		}
		if call > 0 {
			if (err == nil) != (previousErr == nil) || (err != nil && !errors.Is(err, errorCategory(previousErr))) {
				t.Fatalf("seeded error categories differ: %v and %v", previousErr, err)
			}
			if err == nil && (!slices.Equal(previous.Labels, result.Labels) ||
				!equalFuzzData(previous.Centroids, result.Centroids) ||
				math.Float64bits(previous.Inertia) != math.Float64bits(result.Inertia) ||
				previous.Converged != result.Converged || previous.Iterations != result.Iterations) {
				t.Fatal("identical seeded calls produced different results")
			}
		}
		previous, previousErr = result, err
	}
}

func errorCategory(err error) error {
	// Every call above has already been checked against exactly one applicable
	// validation sentinel or ErrNumericalOverflow; unwrap only for comparison.
	for errors.Unwrap(err) != nil {
		err = errors.Unwrap(err)
	}
	return err
}

func checkFuzzResult(t *testing.T, result *kmeans.Result, c fuzzCase) {
	t.Helper()
	if result == nil {
		t.Fatal("success returned a nil result")
	}
	if len(result.Labels) != len(c.data) || len(result.Centroids) != c.k {
		t.Fatal("incorrect result dimensions")
	}
	for _, label := range result.Labels {
		if label < 0 || label >= c.k {
			t.Fatalf("label %d is outside [0, %d)", label, c.k)
		}
	}
	for _, centroid := range result.Centroids {
		if len(centroid) != len(c.data[0]) {
			t.Fatal("incorrect centroid dimensions")
		}
		for _, coordinate := range centroid {
			if math.IsNaN(coordinate) || math.IsInf(coordinate, 0) {
				t.Fatal("non-finite centroid coordinate")
			}
		}
	}
	if math.IsNaN(result.Inertia) || math.IsInf(result.Inertia, 0) || result.Inertia < 0 {
		t.Fatalf("invalid inertia: %v", result.Inertia)
	}
	if result.Iterations < 1 || result.Iterations > c.iterations {
		t.Fatalf("iteration count %d outside [1, %d]", result.Iterations, c.iterations)
	}
}

// Embedding the public interface models unsupported external option wrappers,
// including typed-nil pointers, without depending on private configuration.
type fuzzWrappedOption struct{ kmeans.Option }

const fuzzInvalidKinds = 23

func FuzzClusterInvalid(f *testing.F) {
	for defect := range byte(fuzzInvalidKinds) {
		f.Add(append([]byte{defect}, fuzzSeed([7]byte{2, 1, 1, 2, 9, 2}, 0, 1, 2, 3, 4, 5)...))
	}
	// The coordinate bits supply a non-canonical NaN payload in the NaN case.
	f.Add(append([]byte{17}, fuzzSeed([7]byte{}, math.Float64frombits(0xfff8000000000042))...))
	f.Add([]byte{})
	f.Fuzz(func(t *testing.T, encoded []byte) {
		if len(encoded) > maxFuzzBytes {
			return
		}
		var defect byte
		if len(encoded) > 0 {
			defect, encoded = encoded[0]%fuzzInvalidKinds, encoded[1:]
		}
		for _, method := range []kmeans.InitMethod{kmeans.InitKMeansPlusPlus, kmeans.InitRandom} {
			c := decodeFuzzCase(encoded)
			opts := c.options(method)
			var wantErr error
			switch defect {
			case 0, 1:
				c.k = -int(defect)
				wantErr = kmeans.ErrInvalidK
			case 2:
				c.data = c.data[:min(len(c.data), 3)]
				c.k = len(c.data) + 1
				wantErr = kmeans.ErrInvalidK
			case 3, 4:
				opts = append(opts, kmeans.WithRuns(-int(defect-3)))
				wantErr = kmeans.ErrInvalidRuns
			case 5, 6:
				opts = append(opts, kmeans.WithMaxIterations(-int(defect-5)))
				wantErr = kmeans.ErrInvalidMaxIterations
			case 7, 8, 9, 10:
				tolerance := []float64{0, -1, math.NaN(), math.Inf(1)}[defect-7]
				opts = append(opts, kmeans.WithTolerance(tolerance))
				wantErr = kmeans.ErrInvalidTolerance
			case 11:
				// Prefix prevents mutated bytes from selecting a supported method.
				opts = append(opts, kmeans.WithInitMethod(kmeans.InitMethod("unsupported:"+string(encoded))))
				wantErr = kmeans.ErrInvalidInitMethod
			case 12, 13:
				opts = append(opts, kmeans.WithNCentroidsInitTrials(-int(defect-12)))
				wantErr = kmeans.ErrInvalidNCentroidsInitTrials
			case 14:
				c.data = nil
				wantErr = kmeans.ErrEmptyData
			case 15:
				for i := range c.data {
					c.data[i] = nil
				}
				wantErr = kmeans.ErrNoFeatures
			case 16:
				if len(c.data) == 1 {
					c.data = append(c.data, nil)
				} else {
					c.data[1] = c.data[1][:len(c.data[1])-1]
				}
				wantErr = kmeans.ErrInvalidData
			case 17, 18, 19:
				coordinate := []float64{math.NaN(), math.Inf(1), math.Inf(-1)}[defect-17]
				if defect == 17 {
					// Retain mutated sign and payload while forcing a NaN.
					coordinate = math.Float64frombits(math.Float64bits(c.data[0][0]) | 0x7ff8000000000000)
				}
				c.data[0][0] = coordinate
				wantErr = kmeans.ErrNonFiniteData
			case 20:
				opts = append(opts, nil)
				wantErr = kmeans.ErrInvalidOption
			case 21:
				opts = append(opts, fuzzWrappedOption{kmeans.WithSeed(42)})
				wantErr = kmeans.ErrInvalidOption
			case 22:
				var option *fuzzWrappedOption
				opts = append(opts, option)
				wantErr = kmeans.ErrInvalidOption
			}
			checkFuzzCalls(t, c, opts, wantErr)
		}
	})
}
