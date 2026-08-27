@testset "ROI extraction" begin

    A = zeros(Float32, 16, 16)

    A[5, 6] = 10f0
    A[12, 11] = 8f0

    indices = Vector{CartesianIndex{2}}(undef, 2)

    indices[1] = CartesianIndex(5, 6)
    indices[2] = CartesianIndex(12, 11)

    rois = zeros(Float32, 5, 5, 2)

    n = extract_rois!(
        rois,
        A,
        indices,
        2,
    )

    @test n == 2

    @test rois[3, 3, 1] == 10f0
    @test rois[3, 3, 2] == 8f0
end


@testset "Allocation-free ROI extraction" begin

    A = rand(Float32, 128, 128)

    indices = [
        CartesianIndex(20, 20),
        CartesianIndex(60, 80),
        CartesianIndex(100, 100),
    ]

    rois = zeros(Float32, 11, 11, 3)

    extract_rois!(
        rois,
        A,
        indices,
        3,
    )

    bytes = @allocated extract_rois!(
        rois,
        A,
        indices,
        3,
    )

    @test bytes == 0
end