# Require five complete measurements of each current leaf case at GOMAXPROCS=1.
BEGIN {
    expected["BenchmarkLloyd_KMeansPlusPlus"] = 1
    expected["BenchmarkLloyd_RandomInit"] = 1
    expected["BenchmarkLloyd_VaryingDataSize/size_100"] = 1
    expected["BenchmarkLloyd_VaryingDataSize/size_500"] = 1
    expected["BenchmarkLloyd_VaryingDataSize/size_1000"] = 1
    expected["BenchmarkLloyd_VaryingDataSize/size_5000"] = 1
}
/^Benchmark/ {
    name = $1
    sub(/-1$/, "", name)
    if (!(name in expected) || NF != 8 || $2 !~ /^[1-9][0-9]*$/ ||
        $3 !~ /^[0-9]+(\.[0-9]+)?$/ || $4 != "ns/op" ||
        $5 !~ /^[0-9]+$/ || $6 != "B/op" ||
        $7 !~ /^[0-9]+$/ || $8 != "allocs/op") {
        print "Unexpected or incomplete benchmark row: " $0
        invalid = 1
        next
    }
    count[name]++
}
END {
    for (name in expected) {
        if (count[name] != 5) {
            printf "%s: expected 5 measurements, found %d\n", name, count[name]
            invalid = 1
        }
    }
    if (invalid) {
        print "Benchmark execution is incomplete or does not match the expected suite."
        exit 1
    }
    print "Complete: all six expected cases have five measurements."
}
