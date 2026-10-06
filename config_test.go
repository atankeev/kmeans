package kmeans

import "testing"

// testConfig supplies private algorithm fixtures without exposing configuration.
func testConfig(t *testing.T, k int, options ...Option) *config {
	t.Helper()
	c := newConfig(k)
	for _, option := range options {
		option.apply(c)
	}
	return c
}
