package kmeans

import "testing"

// testKmeans supplies private algorithm fixtures without exposing configuration.
func testKmeans(t *testing.T, k int, options ...Option) *kmeans {
	t.Helper()
	c := newConfig(k)
	for _, option := range options {
		option.apply(c)
	}
	return &kmeans{config: *c}
}
