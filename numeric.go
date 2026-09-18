package kmeans

import "math"

// scaledValue represents a non-negative finite value as fraction * 2^exponent.
// Non-zero fractions are normalized to [0.5, 1), while the exponent may exceed
// the range that float64 can materialize.
type scaledValue struct {
	fraction float64
	exponent int
}

func scaledFromFloat(value float64) scaledValue {
	if value == 0 {
		return scaledValue{}
	}

	fraction, exponent := math.Frexp(value)
	return scaledValue{fraction: fraction, exponent: exponent}
}

func newScaledValue(fraction float64, exponent int) scaledValue {
	if fraction == 0 {
		return scaledValue{}
	}

	normalized, adjustment := math.Frexp(fraction)
	return scaledValue{fraction: normalized, exponent: exponent + adjustment}
}

func (value scaledValue) add(other scaledValue) scaledValue {
	if value.fraction == 0 {
		return other
	}
	if other.fraction == 0 {
		return value
	}
	if value.compare(other) < 0 {
		value, other = other, value
	}

	fraction := value.fraction + math.Ldexp(other.fraction, other.exponent-value.exponent)
	return newScaledValue(fraction, value.exponent)
}

func (value scaledValue) multiply(factor float64) scaledValue {
	if value.fraction == 0 || factor == 0 {
		return scaledValue{}
	}

	fraction, exponent := math.Frexp(factor)
	return newScaledValue(value.fraction*fraction, value.exponent+exponent)
}

func (value scaledValue) divide(divisor float64) scaledValue {
	if value.fraction == 0 {
		return scaledValue{}
	}

	fraction, exponent := math.Frexp(divisor)
	return newScaledValue(value.fraction/fraction, value.exponent-exponent)
}

func (value scaledValue) square() scaledValue {
	return newScaledValue(value.fraction*value.fraction, value.exponent*2)
}

func (value scaledValue) compare(other scaledValue) int {
	if value.fraction == 0 {
		if other.fraction == 0 {
			return 0
		}

		return -1
	}
	if other.fraction == 0 {
		return 1
	}

	if value.exponent < other.exponent {
		return -1
	}
	if value.exponent > other.exponent {
		return 1
	}
	if value.fraction < other.fraction {
		return -1
	}
	if value.fraction > other.fraction {
		return 1
	}

	return 0
}

func (value scaledValue) float64() (float64, bool) {
	result := math.Ldexp(value.fraction, value.exponent)
	return result, !math.IsInf(result, 0) && !math.IsNaN(result)
}

func (value scaledValue) ratio(reference scaledValue) float64 {
	if value.fraction == 0 {
		return 0
	}

	return value.fraction / reference.fraction *
		math.Ldexp(1, value.exponent-reference.exponent)
}

func scaledAbsoluteDifference(first, second float64) scaledValue {
	if first == second {
		return scaledValue{}
	}

	if math.Signbit(first) == math.Signbit(second) {
		return scaledFromFloat(math.Abs(first - second))
	}

	firstFraction, firstExponent := math.Frexp(math.Abs(first))
	secondFraction, secondExponent := math.Frexp(math.Abs(second))
	exponent := max(firstExponent, secondExponent)
	fraction := math.Ldexp(firstFraction, firstExponent-exponent) +
		math.Ldexp(secondFraction, secondExponent-exponent)

	return newScaledValue(fraction, exponent)
}

func scaledSquaredDistance(first, second []float64) scaledValue {
	var distance scaledValue
	for dimension := range first {
		difference := scaledAbsoluteDifference(first[dimension], second[dimension])
		distance = distance.add(difference.square())
	}

	return distance
}

func updateMean(mean, value float64, count int) float64 {
	if count == 1 {
		return value
	}
	if math.Signbit(mean) == math.Signbit(value) {
		return mean + (value-mean)/float64(count)
	}

	weight := 1 / float64(count)
	return mean*(1-weight) + value*weight
}
