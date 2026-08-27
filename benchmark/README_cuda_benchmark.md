CUDA reference benchmark
------------------------

Input:
    512 × 512 × 100 Float32

DoG:
    σ₁ = 1
    σ₂ = 3

Detection:
    threshold = 0.1
    radius = 1
    border = 5

ROI:
    11 × 11

Detected sources:
    509,822

Measured performance:
    detection:                   0.755 ms
    DoG + detection:             4.989 ms
    ROI extraction:              2.514 ms
    complete pipeline:           7.475 ms

Complete throughput:
    ≈ 13,378 frames/s
    ≈ 3.51 Gpixel/s

Benchmark command:
    julia --project=benchmark benchmark/cuda.jl


CUDA SourceExtraction reference
================================

512 × 512 × 1
-------------------------------
DoG                 56.4 μs
Detection           18.1 μs
DoG + detection     66.7 μs
ROI                 34.8 μs
Complete            96.1 μs

512 × 512 × 10
--------------------------------
DoG                462.9 μs
Detection           85.5 μs
DoG + detection    542.0 μs
ROI                271.2 μs
Complete           813.3 μs

512 × 512 × 100
--------------------------------
DoG                  4.266 ms
Detection             0.765 ms
DoG + detection       5.015 ms
ROI                    2.593 ms
Complete               7.574 ms