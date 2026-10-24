// Package coverageprobe provides a temporary fixture for Codecov validation.
package coverageprobe

// Clamp limits value to the inclusive range [0, 10].
func Clamp(value int) int {
	if value < 0 {
		return 0
	}
	if value > 10 {
		return 10
	}
	return value
}
