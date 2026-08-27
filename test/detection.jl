function reference_detect_sources(
    A,
    detector;
    border = SourceExtraction.radius(detector),
)
    R = SourceExtraction.radius(detector)
    strict = SourceExtraction.isstrict(detector)
    threshold = detector.threshold

    indices = CartesianIndex{2}[]
    scores = eltype(A)[]

    nx, ny = size(A)

    for j in (1 + border):(ny - border)
        for i in (1 + border):(nx - border)

            c = A[i, j]
            c > threshold || continue

            ismax = true

            for dj in -R:R
                for di in -R:R
                    (di == 0 && dj == 0) && continue

                    v = A[i + di, j + dj]

                    if strict ? !(c > v) : !(c >= v)
                        ismax = false
                        break
                    end
                end

                ismax || break
            end

            if ismax
                push!(indices, CartesianIndex(i, j))
                push!(scores, c)
            end
        end
    end

    return indices, scores
end

@testset "Local maxima detection" begin

    A = zeros(Float32, 16, 16)

    A[5, 6] = 10f0
    A[12, 11] = 8f0

    indices = Vector{CartesianIndex{2}}(undef, 16)
    scores = Vector{Float32}(undef, 16)

    detector = LocalMaximaDetector(
        1f0;
        radius = 1,
        strict = true,
    )

    n = detect_sources!(
        indices,
        scores,
        A,
        detector,
    )

    @test n == 2

    @test indices[1] == CartesianIndex(5, 6)
    @test indices[2] == CartesianIndex(12, 11)

    @test scores[1] == 10f0
    @test scores[2] == 8f0
end


@testset "Threshold" begin

    A = zeros(Float32, 16, 16)

    A[5, 6] = 10f0
    A[12, 11] = 8f0

    indices = Vector{CartesianIndex{2}}(undef, 16)
    scores = Vector{Float32}(undef, 16)

    detector = LocalMaximaDetector(
        9f0;
        radius = 1,
    )

    n = detect_sources!(
        indices,
        scores,
        A,
        detector,
    )

    @test n == 1
    @test indices[1] == CartesianIndex(5, 6)
end


@testset "Allocation-free detection" begin

    A = zeros(Float32, 128, 128)

    A[20, 20] = 10f0
    A[60, 80] = 20f0
    A[100, 100] = 15f0

    indices = Vector{CartesianIndex{2}}(undef, 100)
    scores = Vector{Float32}(undef, 100)

    detector = LocalMaximaDetector(
        1f0;
        radius = 1,
    )

    # Compile first.
    detect_sources!(
        indices,
        scores,
        A,
        detector,
    )

    bytes = @allocated detect_sources!(
        indices,
        scores,
        A,
        detector,
    )

    @test bytes == 0
end

@testset "Randomized correctness" begin

    for radius in (1, 2, 3)
        for strict in (true, false)

            A = rand(Float32, 64, 73)

            detector = LocalMaximaDetector(
                0.7f0;
                radius = radius,
                strict = strict,
            )

            ref_indices, ref_scores =
                reference_detect_sources(A, detector)

            indices =
                Vector{CartesianIndex{2}}(undef, length(A))

            scores =
                Vector{Float32}(undef, length(A))

            n = detect_sources!(
                indices,
                scores,
                A,
                detector,
            )

            @test indices[1:n] == ref_indices
            @test scores[1:n] == ref_scores
        end
    end
end


@testset "Strict and non-strict maxima" begin

    A = zeros(Float32, 9, 9)

    A[4, 4] = 10
    A[5, 4] = 10

    indices =
        Vector{CartesianIndex{2}}(undef, 20)

    scores =
        Vector{Float32}(undef, 20)

    strict_detector = LocalMaximaDetector(
        1f0;
        radius = 1,
        strict = true,
    )

    n = detect_sources!(
        indices,
        scores,
        A,
        strict_detector,
    )

    @test n == 0

    nonstrict_detector = LocalMaximaDetector(
        1f0;
        radius = 1,
        strict = false,
    )

    n = detect_sources!(
        indices,
        scores,
        A,
        nonstrict_detector,
    )

    @test n == 2

    @test Set(indices[1:n]) ==
        Set((
            CartesianIndex(4, 4),
            CartesianIndex(5, 4),
        ))
end


@testset "Border exclusion" begin

    A = zeros(Float32, 20, 20)

    A[2, 2] = 10f0
    A[10, 10] = 20f0

    indices =
        Vector{CartesianIndex{2}}(undef, 10)

    scores =
        Vector{Float32}(undef, 10)

    detector = LocalMaximaDetector(
        1f0;
        radius = 1,
    )

    n = detect_sources!(
        indices,
        scores,
        A,
        detector;
        border = 2,
    )

    @test n == 1
    @test indices[1] == CartesianIndex(10, 10)
end

@testset "Invalid border" begin

    A = zeros(Float32, 10, 10)

    detector = LocalMaximaDetector(
        1f0;
        radius = 2,
    )

    indices =
        Vector{CartesianIndex{2}}(undef, 10)

    scores =
        Vector{Float32}(undef, 10)

    @test_throws ArgumentError detect_sources!(
        indices,
        scores,
        A,
        detector;
        border = 1,
    )
end


@testset "Batched 2-D detection" begin

    A = zeros(
        Float32,
        32,
        32,
        3,
    )

    A[5, 6, 1] = 10f0
    A[15, 20, 2] = 20f0
    A[25, 10, 3] = 30f0

    indices =
        Vector{CartesianIndex{3}}(
            undef,
            10,
        )

    scores =
        Vector{Float32}(
            undef,
            10,
        )

    detector =
        LocalMaximaDetector(
            1f0;
            radius = 1,
        )

    n = detect_sources!(
        indices,
        scores,
        A,
        detector,
    )

    @test n == 3

    @test indices[1] ==
        CartesianIndex(5, 6, 1)

    @test indices[2] ==
        CartesianIndex(15, 20, 2)

    @test indices[3] ==
        CartesianIndex(25, 10, 3)

    @test scores[1:3] ==
        Float32[10, 20, 30]
end

@testset "Allocation-free batched detection" begin

    A = rand(
        Float32,
        128,
        128,
        10,
    )

    indices =
        Vector{CartesianIndex{3}}(
            undef,
            length(A),
        )

    scores =
        Vector{Float32}(
            undef,
            length(A),
        )

    detector =
        LocalMaximaDetector(
            0.99f0;
            radius = 1,
        )

    detect_sources!(
        indices,
        scores,
        A,
        detector,
    )

    @test @allocated(
        detect_sources!(
            indices,
            scores,
            A,
            detector,
        )
    ) == 0
end