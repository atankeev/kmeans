// Package coverageboundary provides a temporary exact-threshold coverage fixture.
package coverageboundary

// Count returns fourteen, adding three when extra is true.
func Count(extra bool) int {
	result := 0
	result++
	result++
	result++
	result++
	result++
	result++
	result++
	result++
	result++
	result++
	result++
	result++
	result++
	result++
	if extra {
		result++
		result++
		result++
	}
	return result
}
