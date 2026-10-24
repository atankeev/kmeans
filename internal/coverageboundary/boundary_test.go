package coverageboundary

import "testing"

func TestCount(t *testing.T) {
	t.Parallel()

	if got := Count(false /* extra */); got != 14 {
		t.Fatalf("Count(false) = %d, want 14", got)
	}
}
