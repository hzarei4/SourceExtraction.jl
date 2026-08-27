```@meta
CurrentModule = SourceExtraction
```

# User guide

## Choosing an interface

Use [`find_sources`](@ref) when convenience matters most and an image is
processed only once. It prepares the DoG filter and detection storage on every
call, then returns ordinary CPU vectors containing `x` and `y` coordinates.

Use [`SourceExtractor`](@ref) with [`extract_sources!`](@ref) when processing
multiple arrays of the same size and backend. The prepared extractor owns its
FFT plan, detection buffers, and optional ROI storage, so subsequent calls reuse
that state.

## One-shot detection

```julia
xs, ys = find_sources(
    image;
    dog=DoG(1f0, 3f0),
    threshold_fraction=0.05,
    radius=1,
    border=1,
    max_sources=10_000,
)
```

`threshold_fraction` is multiplied by the maximum value in the DoG-filtered
image. Supply `threshold=value` to use an absolute filtered-image threshold
instead. Detection examines a `(2radius + 1) × (2radius + 1)` neighborhood and
does not consider pixels inside the excluded `border`.

The returned coordinates use conventional image coordinates: `x` is the column
and `y` is the row. The lower-level and prepared APIs instead return Julia
`CartesianIndex` values in `(row, column)` order.

## Prepared extraction and ROIs

Construct an extractor with an example array that has the same element type,
dimensions, size, and backend as later inputs:

```julia
extractor = SourceExtractor(
    image;
    dog=DoG(1f0, 3f0),
    detector=LocalMaximaDetector(0.1f0; radius=1),
    roi_size=(11, 11),
    max_sources=10_000,
)

result = extract_sources!(extractor, next_image)

indices = source_indices(result)
scores = source_scores(result)
rois = source_rois(result)
```

ROI dimensions must be positive and odd. The ROI array has shape
`(roi_rows, roi_columns, detected_sources)`, and its center pixel corresponds to
the detected index. The constructor increases the detection border as needed so
every detected source has a complete ROI.

[`source_indices`](@ref), [`source_scores`](@ref), and [`source_rois`](@ref)
return views into the extractor's storage. A later call with the same extractor
can overwrite them. Persist data explicitly when needed:

```julia
saved_indices = copy(source_indices(result))
saved_scores = copy(source_scores(result))
saved_rois = copy(source_rois(result))
```

Omit `roi_size` when ROIs are not needed. In that case
`source_rois(result) === nothing`.

## Image stacks

A 3-D input is interpreted as `(row, column, frame)`. Filtering and detection
are independent between frames, while `max_sources` is the capacity across the
whole stack. Returned indices are `CartesianIndex(row, column, frame)` values.

```julia
stack = zeros(Float32, 128, 128, 20)

extractor = SourceExtractor(
    stack;
    dog=DoG(1f0, 3f0),
    detector=LocalMaximaDetector(0.1f0),
    max_sources=20_000,
)

result = extract_sources!(extractor, stack)
```

## CUDA

Load CUDA.jl and construct the extractor from a `CuArray` to select the CUDA
implementation:

```julia
using CUDA
using SourceExtraction

gpu_image = CUDA.zeros(Float32, 2048, 2048)

extractor = SourceExtractor(
    gpu_image;
    dog=DoG(1f0, 3f0),
    detector=LocalMaximaDetector(0.1f0),
    roi_size=(11, 11),
    max_sources=100_000,
)

result = extract_sources!(extractor, gpu_image)
indices_on_cpu = Array(source_indices(result))
```

The package's CUDA extension loads automatically; no SourceExtraction-specific
setup is required. Retrieving the detected count and transferring results to the
CPU require synchronization.

## Low-level processing

The pipeline stages can also be prepared and called separately:

```julia
dog = DoG(1f0, 3f0)
prepared_filter = prepare_filter(image, dog)
filtered = filter_sources!(prepared_filter, image)

workspace = DetectionWorkspace(image; max_sources=10_000)
detector = LocalMaximaDetector(0.1f0; radius=1, strict=true)
detect_sources!(workspace, filtered, detector)

n = source_count(workspace)
rois = Array{Float32}(undef, 11, 11, 10_000)
extract_rois!(rois, image, workspace, n)
```

This interface is useful when filtering, detection, and ROI extraction need to
be scheduled independently. On the CPU, the prepared operations avoid
allocations when the supplied buffers have sufficient capacity.

## Capacity and borders

Set `max_sources` to an upper bound for detections. Detection throws an
`ArgumentError` if CPU output buffers are too small. The border must be at least
the local-maximum radius and, when extracting ROIs through `SourceExtractor`, at
least half of each ROI dimension.
