using CUDA

if CUDA.functional()

    @testset "CUDA source detection" begin

        A = zeros(Float32, 64, 64)

        A[10, 12] = 10f0
        A[30, 40] = 20f0
        A[50, 20] = 15f0

        detector =
            LocalMaximaDetector(
                1f0;
                radius = 1,
            )

        # CPU reference

        cpu_indices =
            Vector{CartesianIndex{2}}(
                undef,
                100,
            )

        cpu_scores =
            Vector{Float32}(
                undef,
                100,
            )

        n_cpu = detect_sources!(
            cpu_indices,
            cpu_scores,
            A,
            detector,
        )

        # GPU

        d_A = CuArray(A)

        gpu_workspace =
            DetectionWorkspace(
                d_A;
                max_sources = 100,
            )

        detect_sources!(
            gpu_workspace,
            d_A,
            detector,
        )

        CUDA.synchronize()

        n_gpu =
            source_count(
                gpu_workspace,
            )

        @test n_gpu == n_cpu

        gpu_indices =
            Array(
                gpu_workspace.indices[
                    1:n_gpu
                ],
            )

        gpu_scores =
            Array(
                gpu_workspace.scores[
                    1:n_gpu
                ],
            )

        @test Set(gpu_indices) ==
              Set(cpu_indices[1:n_cpu])

        for (
            idx,
            score,
        ) in zip(
            gpu_indices,
            gpu_scores,
        )

            j = findfirst(
                ==(idx),
                cpu_indices[1:n_cpu],
            )

            @test !isnothing(j)
            @test score == cpu_scores[j]
        end
    end


    @testset "CUDA ROI extraction" begin

        A = rand(Float32, 64, 64)
        d_A = CuArray(A)

        detector = LocalMaximaDetector(
            0.7f0;
            radius = 1,
        )

        workspace = DetectionWorkspace(
            d_A;
            max_sources = 1000,
        )

        detect_sources!(
            workspace,
            d_A,
            detector;
            border = 3,
        )

        CUDA.synchronize()

        n = source_count(workspace)

        indices =
            Array(
                workspace.indices[1:n],
            )

        # CPU reference using exact same source ordering
        cpu_rois =
            Array{Float32}(
                undef,
                7,
                7,
                n,
            )

        extract_rois!(
            cpu_rois,
            A,
            indices,
            n,
        )

        # GPU
        gpu_rois =
            CuArray{Float32}(
                undef,
                7,
                7,
                n,
            )

        extract_rois!(
            gpu_rois,
            d_A,
            workspace,
            n,
        )

        CUDA.synchronize()

        @test Array(gpu_rois) == cpu_rois
    end



    @testset "CUDA batched detection" begin

    A = rand(
        Float32,
        64,
        64,
        8,
    )

    detector =
        LocalMaximaDetector(
            0.9f0;
            radius = 1,
        )

    cpu_indices =
        Vector{CartesianIndex{3}}(
            undef,
            length(A),
        )

    cpu_scores =
        Vector{Float32}(
            undef,
            length(A),
        )

    n_cpu =
        detect_sources!(
            cpu_indices,
            cpu_scores,
            A,
            detector;
            border = 3,
        )

    d_A = CuArray(A)

    workspace =
        DetectionWorkspace(
            d_A;
            max_sources = length(A),
        )

    detect_sources!(
        workspace,
        d_A,
        detector;
        border = 3,
    )

    CUDA.synchronize()

    n_gpu =
        source_count(
            workspace,
        )

    gpu_indices =
        Array(
            workspace.indices[1:n_gpu],
        )

    @test n_gpu == n_cpu

    @test Set(gpu_indices) ==
          Set(cpu_indices[1:n_cpu])
end

@testset "CUDA SourceExtractor" begin

    A =
        zeros(
            Float32,
            64,
            64,
            3,
        )

    A[10, 12, 1] = 10f0
    A[30, 40, 2] = 20f0
    A[50, 20, 3] = 15f0

    d_A =
        CuArray(A)

    extractor =
        SourceExtractor(
            d_A;
            dog = DoG(1f0, 3f0),
            detector =
                LocalMaximaDetector(
                    0.1f0;
                    radius = 1,
                ),
            roi_size = (7, 7),
            max_sources = 100,
        )

    result =
        extract_sources!(
            extractor,
            d_A,
        )

    CUDA.synchronize()

    @test source_count(result) == 3

    indices =
        Array(
            source_indices(result),
        )

    @test Set(indices) ==
          Set(
              [
                  CartesianIndex(10, 12, 1),
                  CartesianIndex(30, 40, 2),
                  CartesianIndex(50, 20, 3),
              ],
          )

    rois =
        Array(
            source_rois(result),
        )

    @test size(rois) ==
          (
              7,
              7,
              3,
          )

    for k in eachindex(indices)

        I =
            indices[k]

        @test rois[4, 4, k] ==
              A[I]
    end
end

end