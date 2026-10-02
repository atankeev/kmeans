package kmeans

import (
	"fmt"
	"testing"
)

// Each Lloyd benchmark iteration measures Cluster with seed 42 and the default NInit of 10.
func BenchmarkLloyd_KMeansPlusPlus(b *testing.B) {
	// 1,000 two-dimensional points in two groups; 5 clusters, up to 100 iterations.
	data := make([][]float64, 1000)
	for i := 0; i < 1000; i++ {
		if i < 500 {
			data[i] = []float64{float64(i % 10), float64(i % 10)}
		} else {
			data[i] = []float64{float64(i%10) + 50, float64(i%10) + 50}
		}
	}

	kmeans := NewWithOptions(5,
		WithInitMethod(InitKMeansPlusPlus),
		WithMaxIter(100),
		WithRandomSeed(42),
	)
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		if _, err := kmeans.Cluster(data); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkLloyd_RandomInit(b *testing.B) {
	// 1,000 two-dimensional points in two groups; 5 clusters, up to 100 iterations.
	data := make([][]float64, 1000)
	for i := 0; i < 1000; i++ {
		if i < 500 {
			data[i] = []float64{float64(i % 10), float64(i % 10)}
		} else {
			data[i] = []float64{float64(i%10) + 50, float64(i%10) + 50}
		}
	}

	kmeans := NewWithOptions(5,
		WithInitMethod(InitRandom),
		WithMaxIter(100),
		WithRandomSeed(42),
	)
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		if _, err := kmeans.Cluster(data); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkLloyd_VaryingDataSize(b *testing.B) {
	// Two-dimensional points in two groups; 5 clusters, up to 50 iterations.
	sizes := []int{100, 500, 1000, 5000}

	for _, size := range sizes {
		data := make([][]float64, size)
		for i := 0; i < size; i++ {
			if i < size/2 {
				data[i] = []float64{float64(i % 10), float64(i % 10)}
			} else {
				data[i] = []float64{float64(i%10) + 50, float64(i%10) + 50}
			}
		}

		b.Run(fmt.Sprintf("size_%d", size), func(b *testing.B) {
			kmeans := NewWithOptions(5,
				WithInitMethod(InitKMeansPlusPlus),
				WithMaxIter(50),
				WithRandomSeed(42),
			)
			b.ReportAllocs()
			b.ResetTimer()
			for b.Loop() {
				if _, err := kmeans.Cluster(data); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
