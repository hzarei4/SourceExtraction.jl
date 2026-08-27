using SourceExtraction
using CUDA
using FourierTools
using SeparableFunctions
using BenchmarkTools


NFRAMES = 100

A = rand(
    Float32,
    512,
    512,
    NFRAMES,
)

d_A = CuArray(A)

dog = DoG(1f0, 3f0)

d_filter =
    prepare_filter(
        d_A,
        dog,
    )

detector =
    LocalMaximaDetector(
        0.1f0;
        radius = 1,
    )

workspace =
    DetectionWorkspace(
        d_A;
        max_sources = 2_000_000,
    )

@btime CUDA.@sync begin
    filtered =
        filter_sources!(
            $d_filter,
            $d_A,
        )

    detect_sources!(
        $workspace,
        filtered,
        $detector;
        border = 5,
    )
end
#   4.989 ms (67 allocations: 4.50 KiB)
# DetectionWorkspace{CuArray{CartesianIndex{3}, 1, CUDACore.DeviceMemory}, CuArray{Float32, 1, CUDACore.DeviceMemory}, 
# CuArray{Int32, 1, CUDACore.DeviceMemory}}(CartesianIndex{3}[CartesianIndex(69, 15, 1), CartesianIndex(296, 24, 1), 
# CartesianIndex(167, 23, 1), CartesianIndex(175, 24, 1), CartesianIndex(142, 24, 1), CartesianIndex(367, 14, 1), 
# CartesianIndex(266, 29, 1), CartesianIndex(39, 23, 1), CartesianIndex(291, 31, 1), CartesianIndex(299, 31, 1)  …  
# CartesianIndex(0, 0, 0), CartesianIndex(0, 0, 0), CartesianIndex(0, 0, 0), CartesianIndex(0, 0, 0), CartesianIndex(0, 0, 0), 
# CartesianIndex(0, 0, 0), CartesianIndex(0, 0, 0), CartesianIndex(0, 0, 0), CartesianIndex(0, 0, 0), CartesianIndex(0, 0, 0)], 
# Float32[0.105815716, 0.10354031, 0.12846854, 0.13706794, 0.14287901, 0.21234952, 0.12809724, 0.10906715, 0.11997904, 0.1053269  …  
# 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], Int32[509822])