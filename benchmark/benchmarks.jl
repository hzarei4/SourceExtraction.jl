# using SourceExtraction
# using BenchmarkTools

# SUITE = BenchmarkGroup()
# SUITE["rand"] = @benchmarkable rand(10)

# # Write your benchmarks here.
using BenchmarkTools
using SourceExtraction

const NX = 2048
const NY = 2048

A = rand(Float32, NX, NY)

indices =
    Vector{CartesianIndex{2}}(undef, 500_000)

scores =
    Vector{Float32}(undef, 500_000)


function run_benchmark(threshold; radius=1)
    detector = LocalMaximaDetector(
        Float32(threshold);
        radius = radius,
    )

    n = detect_sources!(
        indices,
        scores,
        A,
        detector,
    )

    println()
    println(
        "threshold = ",
        threshold,
        ", radius = ",
        radius,
        ", sources = ",
        n,
    )

    @btime detect_sources!(
        $indices,
        $scores,
        $A,
        $detector,
    )
end


println("========================================")
println("2-D SOURCE DETECTION")
println("image size = $(NX) × $(NY)")
println("========================================")

run_benchmark(0.999; radius=1)
run_benchmark(0.99;  radius=1)
run_benchmark(0.9;   radius=1)

run_benchmark(0.999; radius=2)
run_benchmark(0.999; radius=3)