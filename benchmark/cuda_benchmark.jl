using SourceExtraction
using CUDA
using BenchmarkTools

CUDA.functional() ||
    error("CUDA is not functional on this system.")


function benchmark_cuda(
    nframes;
    nx = 512,
    ny = 512,
    threshold = 0.1f0,
    radius = 1,
    dog = DoG(1f0, 3f0),
    roi_size = (11, 11),
    max_sources_per_frame = 10_000,
)

    println()
    println("="^70)
    println("CUDA SOURCE EXTRACTION BENCHMARK")
    println("="^70)
    println("Image size:       $nx × $ny")
    println("Frames:           $nframes")
    println("Total pixels:     $(nx * ny * nframes)")
    println("Element type:     Float32")
    println("DoG:              σ₁=$(dog.sigma1), σ₂=$(dog.sigma2)")
    println("Threshold:        $threshold")
    println("Detection radius: $radius")
    println("ROI size:         $(roi_size[1]) × $(roi_size[2])")
    println()

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------

    A = rand(
        Float32,
        nx,
        ny,
        nframes,
    )

    d_A = CuArray(A)

    # ------------------------------------------------------------------
    # Preparation
    # ------------------------------------------------------------------

    prepared_filter = prepare_filter(
        d_A,
        dog,
    )

    detector = LocalMaximaDetector(
        threshold;
        radius = radius,
    )

    max_sources =
        max_sources_per_frame *
        nframes

    workspace = DetectionWorkspace(
        d_A;
        max_sources = max_sources,
    )

    border = max(
        radius,
        roi_size[1] ÷ 2,
        roi_size[2] ÷ 2,
    )

    # ------------------------------------------------------------------
    # Warm-up + determine number of sources
    # ------------------------------------------------------------------

    filtered = filter_sources!(
        prepared_filter,
        d_A,
    )

    detect_sources!(
        workspace,
        filtered,
        detector;
        border = border,
    )

    CUDA.synchronize()

    n = source_count(workspace)

    println("Detected sources: $n")
    println(
        "Sources/frame:     ",
        round(
            n / nframes;
            digits = 2,
        ),
    )

    n <= max_sources ||
        error(
            "Detected $n sources but workspace capacity is only $max_sources",
        )

    d_rois = CuArray{Float32}(
        undef,
        roi_size[1],
        roi_size[2],
        n,
    )

    # Warm-up ROI extraction.
    extract_rois!(
        d_rois,
        d_A,
        workspace,
        n,
    )

    CUDA.synchronize()

    # ------------------------------------------------------------------
    # Benchmarks
    # ------------------------------------------------------------------

    println()
    println("-"^70)
    println("DoG filtering")
    println("-"^70)

    @btime CUDA.@sync filter_sources!(
        $prepared_filter,
        $d_A,
    )


    println()
    println("-"^70)
    println("Source detection")
    println("-"^70)

    @btime CUDA.@sync detect_sources!(
        $workspace,
        $filtered,
        $detector;
        border = $border,
    )


    println()
    println("-"^70)
    println("DoG + detection")
    println("-"^70)

    @btime CUDA.@sync begin
        filtered = filter_sources!(
            $prepared_filter,
            $d_A,
        )

        detect_sources!(
            $workspace,
            filtered,
            $detector;
            border = $border,
        )
    end


    println()
    println("-"^70)
    println("ROI extraction")
    println("-"^70)

    @btime CUDA.@sync extract_rois!(
        $d_rois,
        $d_A,
        $workspace,
        $n,
    )


    println()
    println("-"^70)
    println("Complete pipeline")
    println("DoG + detection + ROI extraction")
    println("-"^70)

    @btime CUDA.@sync begin
        filtered = filter_sources!(
            $prepared_filter,
            $d_A,
        )

        detect_sources!(
            $workspace,
            filtered,
            $detector;
            border = $border,
        )

        extract_rois!(
            $d_rois,
            $d_A,
            $workspace,
            $n,
        )
    end

    println()

    return nothing
end


# ======================================================================
# Run benchmark
# ======================================================================

for nframes in (1, 10, 100)
    benchmark_cuda(nframes)
end