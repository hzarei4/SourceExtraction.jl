```@meta
CurrentModule = SourceExtraction
```

# SourceExtraction.jl

SourceExtraction.jl provides fast source detection and region-of-interest (ROI)
extraction for microscopy and astronomical images. It combines
Difference-of-Gaussians (DoG) filtering with thresholded local-maximum detection,
supports 2-D images and `(row, column, frame)` stacks, and can reuse its working
buffers in repeated workloads.

## Installation

Install the package directly from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/hzarei4/SourceExtraction.jl")
```

CUDA support is provided by a package extension and is activated automatically
when CUDA.jl is loaded.

## Quick start

For a single 2-D image, [`find_sources`](@ref) returns separate vectors of
horizontal (`x`, or column) and vertical (`y`, or row) coordinates:

```@example quick-start
using SourceExtraction

image = zeros(Float32, 64, 64)
image[20, 30] = 10
image[45, 42] = 8

xs, ys = find_sources(image; threshold=0.1f0)
sort!(collect(zip(xs, ys)))
```

The default threshold is 25% of the strongest DoG-filtered response. Set
`threshold_fraction` to another fraction or pass an absolute `threshold` as in
the example above.

For a stream of same-sized images, prepare a [`SourceExtractor`](@ref) once and
reuse it:

```@example quick-start
extractor = SourceExtractor(
    image;
    dog=DoG(1f0, 3f0),
    detector=LocalMaximaDetector(0.1f0),
    roi_size=(11, 11),
    max_sources=100,
)

result = extract_sources!(extractor, image)
(
    count=source_count(result),
    indices=collect(source_indices(result)),
    roi_size=size(source_rois(result)),
)
```

Results from the prepared API reference reusable storage owned by the
extractor. Copy a result view if its contents must survive the next call.

See the [user guide](guide.md) for thresholds, batched data, ROIs, CUDA, and the
low-level allocation-free interface. The complete public interface is listed in
the [API reference](api.md).

## Package features

- One-shot and prepared APIs for different workload sizes.
- Strict or non-strict local maxima with configurable radius and threshold.
- Optional fixed-size ROI extraction from the original input.
- Independent processing of every frame in 3-D image stacks.
- Reusable CPU and CUDA buffers for performance-sensitive pipelines.
