# source_detection_comparison.jl
#
# SourceExtraction.jl source-detection benchmark.
#
# EXACT comparisons:
#   A. Thresholded strict 3×3 local maxima:
#        SourceExtraction.detect_sources!
#        ImageFiltering.findlocalmaxima + threshold filtering
#
#   B. Raw strict 3×3 local maxima, no amplitude threshold:
#        SourceExtraction.detect_sources! with threshold=-Inf
#        ImageFiltering.findlocalmaxima
#
# RELATED comparisons (different algorithms):
#        ImageMorphology.local_maxima!
#        Photometry.PeakMesh
#
# Run:
#   julia --project=benchmark benchmark\source_detection_comparison.jl

using SourceExtraction
using BenchmarkTools
using ImageFiltering
using ImageMorphology
using Random
using Statistics
using Printf
using FFTW

const HAVE_PHOTOMETRY = Base.find_package("Photometry") !== nothing
if HAVE_PHOTOMETRY
    @eval using Photometry
end

const NX = 2048
const NY = 2048
const NSAMPLES_EXACT = 30
const NSAMPLES_RELATED = 3
const MAX_SOURCES = NX * NY
const RNG_SEED = 0x51A7CE

@inline ms(ns) = ns / 1e6

function stats(t)
    b = minimum(t)
    m = median(t)
    return (
        minimum_ns = b.time,
        median_ns = m.time,
        memory = b.memory,
        allocs = b.allocs,
    )
end

function print_result(name, r)
    @printf(
        "%-38s %10.3f ms   median %10.3f ms   %10d B   %6d allocs\n",
        name,
        ms(r.minimum_ns),
        ms(r.median_ns),
        r.memory,
        r.allocs,
    )
end

function imagefiltering_thresholded(A, threshold)
    inds = ImageFiltering.findlocalmaxima(
        A;
        window = (3, 3),
        edges = false,
    )
    filter!(I -> @inbounds(A[I] > threshold), inds)
    return inds
end

function exact_thresholded_case(A, threshold, indices, scores)
    detector = LocalMaximaDetector(
        threshold;
        radius = 1,
        strict = true,
    )

    n = detect_sources!(
        indices,
        scores,
        A,
        detector;
        border = 1,
    )

    ref = imagefiltering_thresholded(A, threshold)

    @assert Set(@view(indices[1:n])) == Set(ref)

    detect_sources!(indices, scores, A, detector; border=1)
    imagefiltering_thresholded(A, threshold)

    tse = @benchmark detect_sources!(
        $indices,
        $scores,
        $A,
        $detector;
        border = 1,
    ) samples=NSAMPLES_EXACT evals=1

    tif = @benchmark imagefiltering_thresholded(
        $A,
        $threshold,
    ) samples=NSAMPLES_EXACT evals=1

    se = stats(tse)
    imf = stats(tif)

    return (
        n = n,
        se = se,
        imf = imf,
        speedup = imf.minimum_ns / se.minimum_ns,
    )
end

function raw_localmax_case(A, indices, scores)
    # -Inf means every finite pixel passes the amplitude test.
    detector = LocalMaximaDetector(
        -Inf32;
        radius = 1,
        strict = true,
    )

    n = detect_sources!(
        indices,
        scores,
        A,
        detector;
        border = 1,
    )

    ref = ImageFiltering.findlocalmaxima(
        A;
        window = (3, 3),
        edges = false,
    )

    @assert Set(@view(indices[1:n])) == Set(ref)

    detect_sources!(indices, scores, A, detector; border=1)
    ImageFiltering.findlocalmaxima(A; window=(3,3), edges=false)

    tse = @benchmark detect_sources!(
        $indices,
        $scores,
        $A,
        $detector;
        border = 1,
    ) samples=NSAMPLES_EXACT evals=1

    tif = @benchmark ImageFiltering.findlocalmaxima(
        $A;
        window = (3, 3),
        edges = false,
    ) samples=NSAMPLES_EXACT evals=1

    se = stats(tse)
    imf = stats(tif)

    return (
        n = n,
        se = se,
        imf = imf,
        speedup = imf.minimum_ns / se.minimum_ns,
    )
end

function related_methods(A, threshold)
    println()
    println("RELATED METHODS — NOT EXACTLY THE SAME DETECTOR")
    println("-"^92)

    labels = zeros(Int, size(A))

    ImageMorphology.local_maxima!(
        labels,
        A;
        connectivity = 2,
    )

    tm = @benchmark ImageMorphology.local_maxima!(
        $labels,
        $A;
        connectivity = 2,
    ) samples=NSAMPLES_RELATED evals=1

    print_result(
        "ImageMorphology.local_maxima!",
        stats(tm),
    )

    if HAVE_PHOTOMETRY
        err = fill(
            eltype(A)(threshold),
            size(A),
        )

        # nsigma=1 => threshold numerically equals err.
        # PeakMesh is a box/grid algorithm, not sliding 3×3 local maxima.
        pm = Photometry.Detection.PeakMesh(
            (3, 3),
            1.0,
        )

        # Photometry 0.9.8 in the tested environment uses the sorting flag
        # as the fourth POSITIONAL argument.
        Photometry.Detection.extract_sources(
            pm,
            A,
            err,
            false,
        )

        tp = @benchmark Photometry.Detection.extract_sources(
            $pm,
            $A,
            $err,
            false,
        ) samples=NSAMPLES_RELATED evals=1

        print_result(
            "Photometry.PeakMesh",
            stats(tp),
        )
    end
end

function main()
    println()
    println("Source detection comparison")
    println("---------------------------")
    println("Julia:              ", VERSION)
    println("Julia threads:      ", Threads.nthreads())
    println("SourceExtraction:   ", Base.pkgversion(SourceExtraction))
    println("ImageFiltering:     ", Base.pkgversion(ImageFiltering))
    println("ImageMorphology:    ", Base.pkgversion(ImageMorphology))
    HAVE_PHOTOMETRY &&
        println("Photometry:         ", Base.pkgversion(Photometry))

    rng = Xoshiro(RNG_SEED)

    A = rand(
        rng,
        Float32,
        NX,
        NY,
    )

    indices = Vector{CartesianIndex{2}}(
        undef,
        MAX_SOURCES,
    )

    scores = Vector{Float32}(
        undef,
        MAX_SOURCES,
    )

    println()
    println("="^92)
    println("CONTROL — RAW STRICT 3×3 LOCAL MAXIMA, NO AMPLITUDE THRESHOLD")
    println("="^92)

    raw = raw_localmax_case(
        A,
        indices,
        scores,
    )

    println("sources: ", raw.n)
    print_result(
        "SourceExtraction.detect_sources!",
        raw.se,
    )
    print_result(
        "ImageFiltering.findlocalmaxima",
        raw.imf,
    )
    @printf("speedup: %.2f×\n", raw.speedup)

    thresholds = (
        0.999f0,
        0.99f0,
        0.90f0,
    )

    results = NamedTuple[]

    for threshold in thresholds
        println()
        println("="^92)
        @printf(
            "THRESHOLDED STRICT 3×3 LOCAL MAXIMA — threshold = %.3f\n",
            threshold,
        )
        println("="^92)

        r = exact_thresholded_case(
            A,
            threshold,
            indices,
            scores,
        )

        push!(results, r)

        println("sources: ", r.n)
        print_result(
            "SourceExtraction.detect_sources!",
            r.se,
        )
        print_result(
            "ImageFiltering + threshold",
            r.imf,
        )
        @printf("speedup: %.2f×\n", r.speedup)
    end

    # Realistic DoG-filtered workload.
    FFTW.set_num_threads(
        min(
            Threads.nthreads(),
            16,
        ),
    )

    dog = DoG(
        1f0,
        3f0,
    )

    prepared = prepare_filter(
        A,
        dog,
    )

    filtered = copy(
        filter_sources!(
            prepared,
            A,
        ),
    )

    println()
    println("="^92)
    println("DoG-FILTERED WORKLOAD — threshold = 0.1")
    println("="^92)

    dogres = exact_thresholded_case(
        filtered,
        0.1f0,
        indices,
        scores,
    )

    println("sources: ", dogres.n)
    print_result(
        "SourceExtraction.detect_sources!",
        dogres.se,
    )
    print_result(
        "ImageFiltering + threshold",
        dogres.imf,
    )
    @printf("speedup: %.2f×\n", dogres.speedup)

    related_methods(
        filtered,
        0.1f0,
    )

    println()
    println("="^92)
    println("SUMMARY")
    println("="^92)
    @printf(
        "Raw local-max kernel speedup:          %.2f×\n",
        raw.speedup,
    )
    for (t, r) in zip(thresholds, results)
        @printf(
            "Threshold %.3f source-detection speedup: %.2f×\n",
            t,
            r.speedup,
        )
    end
    @printf(
        "DoG workload source-detection speedup: %.2f×\n",
        dogres.speedup,
    )

    println()
    println(
        "Use the RAW result to discuss local-maximum kernel speed.",
    )
    println(
        "Use the THRESHOLDED results to discuss practical thresholded source detection.",
    )
end

main()
