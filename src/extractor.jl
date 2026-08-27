export find_sources, find_sources!

"""
    SourceExtractor

Prepared high-level source-extraction pipeline.

A `SourceExtractor` owns all reusable state required for:

1. DoG filtering
2. local-maximum source detection
3. optional ROI extraction

The extractor is prepared for the size and backend of the array used to
construct it. Repeated calls to [`extract_sources!`](@ref) reuse the same
buffers.

If `roi_size === nothing`, no ROI storage is allocated and only source
indices and scores are produced.
"""
struct SourceExtractor{F,D,W,R,S}
    filter::F
    detector::D
    workspace::W
    rois::R
    input_size::S
    border::Int
end


"""
    SourceExtractionResult

Result returned by [`extract_sources!`](@ref).

The result references storage owned by its `SourceExtractor`. A subsequent
call to `extract_sources!` using the same extractor may overwrite the data.

Use `copy(source_indices(result))`, `copy(source_scores(result))`, or
`copy(source_rois(result))` if the data must persist independently.
"""
struct SourceExtractionResult{I,S,R}
    indices::I
    scores::S
    rois::R
    n::Int
end


"""
    SourceExtractor(
        A;
        dog,
        detector,
        roi_size=nothing,
        max_sources=10_000,
        border=nothing,
    )

Prepare a reusable source-extraction pipeline for `A`.

`A` may be either:

- `x × y`
- `x × y × frame`

For batched input, filtering and source detection are performed
independently within each frame.

`max_sources` is the total source capacity across all frames.

If `roi_size` is supplied, it must be a tuple of two positive odd integers.
The detection border is automatically made large enough to accommodate the
detection radius and ROI half-width.

A larger explicit `border` may be supplied if desired.
"""
function SourceExtractor(
    A::AbstractArray;
    dog::DoG,
    detector::LocalMaximaDetector,
    roi_size = nothing,
    max_sources::Integer = 10_000,
    border = nothing,
)
    N = ndims(A)

    N in (2, 3) ||
        throw(
            ArgumentError(
                "SourceExtractor currently supports 2-D arrays " *
                "and 3-D x × y × frame arrays",
            ),
        )

    max_sources > 0 ||
        throw(
            ArgumentError(
                "max_sources must be positive",
            ),
        )

    prepared_filter =
        prepare_filter(
            A,
            dog,
        )

    workspace =
        DetectionWorkspace(
            A;
            max_sources = max_sources,
        )

    minimum_border =
        radius(detector)

    rois =
        if isnothing(roi_size)

            nothing

        else

            length(roi_size) == 2 ||
                throw(
                    ArgumentError(
                        "roi_size must contain exactly two dimensions",
                    ),
                )

            roi_nx =
                Int(roi_size[1])

            roi_ny =
                Int(roi_size[2])

            roi_nx > 0 ||
                throw(
                    ArgumentError(
                        "ROI size in dimension 1 must be positive",
                    ),
                )

            roi_ny > 0 ||
                throw(
                    ArgumentError(
                        "ROI size in dimension 2 must be positive",
                    ),
                )

            isodd(roi_nx) ||
                throw(
                    ArgumentError(
                        "ROI size in dimension 1 must be odd",
                    ),
                )

            isodd(roi_ny) ||
                throw(
                    ArgumentError(
                        "ROI size in dimension 2 must be odd",
                    ),
                )

            minimum_border =
                max(
                    minimum_border,
                    roi_nx ÷ 2,
                    roi_ny ÷ 2,
                )

            similar(
                A,
                eltype(A),
                (
                    roi_nx,
                    roi_ny,
                    Int(max_sources),
                ),
            )
        end

    actual_border =
        if isnothing(border)
            minimum_border
        else
            Int(border)
        end

    actual_border >= minimum_border ||
        throw(
            ArgumentError(
                "border must be at least $minimum_border",
            ),
        )

    return SourceExtractor(
        prepared_filter,
        detector,
        workspace,
        rois,
        size(A),
        actual_border,
    )
end


"""
    extract_sources!(extractor, A)

Run the complete prepared source-extraction pipeline.

The operation consists of:

1. DoG filtering
2. local-maximum detection
3. optional ROI extraction from the original input array

The returned `SourceExtractionResult` references the extractor's reusable
buffers and is overwritten by subsequent calls using the same extractor.

For CUDA arrays, obtaining the source count requires a small device-to-host
synchronization before ROI extraction.
"""
function extract_sources!(
    extractor::SourceExtractor,
    A::AbstractArray,
)
    size(A) == extractor.input_size ||
        throw(
            DimensionMismatch(
                "input size $(size(A)) does not match prepared " *
                "size $(extractor.input_size)",
            ),
        )

    filtered =
        filter_sources!(
            extractor.filter,
            A,
        )

    detect_sources!(
        extractor.workspace,
        filtered,
        extractor.detector;
        border = extractor.border,
    )

    n =
        source_count(
            extractor.workspace,
        )

    if !isnothing(extractor.rois)
        extract_rois!(
            extractor.rois,
            A,
            extractor.workspace,
            n,
        )
    end

    return SourceExtractionResult(
        extractor.workspace.indices,
        extractor.workspace.scores,
        extractor.rois,
        n,
    )
end


"""
    source_indices(result)

Return a view containing the valid detected source indices.
"""
source_indices(
    result::SourceExtractionResult,
) = view(
    result.indices,
    1:result.n,
)


"""
    source_scores(result)

Return a view containing the valid source scores.
"""
source_scores(
    result::SourceExtractionResult,
) = view(
    result.scores,
    1:result.n,
)


"""
    source_rois(result)

Return a view containing the valid extracted ROIs.

Returns `nothing` if the `SourceExtractor` was constructed without
`roi_size`.
"""
source_rois(
    result::SourceExtractionResult{I,S,Nothing},
) where {I,S} = nothing


source_rois(
    result::SourceExtractionResult,
) = view(
    result.rois,
    :,
    :,
    1:result.n,
)


"""
    source_count(result)

Return the number of detected sources.
"""
source_count(
    result::SourceExtractionResult,
) = result.n


Base.length(
    result::SourceExtractionResult,
) = result.n


Base.isempty(
    result::SourceExtractionResult,
) = result.n == 0



"""
    find_sources(A; kwargs...) -> xs, ys

Detect sources in a single 2-D image and return their x and y coordinates.

This is the simple one-shot API. For repeated processing of many images,
use `SourceExtractor` and `extract_sources!` to reuse prepared buffers.

# Example

```julia
xs, ys = find_sources(image)
```

"""
function find_sources(
    A::AbstractMatrix;
    dog = DoG(1f0, 3f0),
    threshold = nothing,
    threshold_fraction::Real = 0.25,
    radius::Integer = 1,
    border::Integer = radius,
    max_sources::Integer = length(A),
    )

    # Prepare and run the DoG once.
    prepared_filter =
    prepare_filter(
    A,
    dog,
    )

    filtered =
        filter_sources!(
            prepared_filter,
            A,
        )

    # Automatic threshold relative to the strongest DoG response.
    actual_threshold =
        isnothing(threshold) ?
        threshold_fraction * maximum(filtered) :
        threshold

    detector =
        LocalMaximaDetector(
            actual_threshold;
            radius = radius,
        )

    workspace =
        DetectionWorkspace(
            A;
            max_sources = max_sources,
        )

    detect_sources!(
        workspace,
        filtered,
        detector;
        border = border,
    )

    n =
        source_count(
            workspace,
        )

    # CartesianIndex is (row, column), so:
    #
    #   x = column = I[2]
    #   y = row    = I[1]
    #
    indices =
        Array(
            workspace.indices[1:n],
        )

    xs =
        Vector{Int}(
            undef,
            n,
        )

    ys =
        Vector{Int}(
            undef,
            n,
        )

    @inbounds for k in 1:n
        I = indices[k]

        ys[k] = I[1]
        xs[k] = I[2]
    end

    return xs, ys

end

"""
    find_sources!(extractor, A) -> xs, ys

Run a prepared `SourceExtractor` and return x/y coordinates.

This method reuses the extractor's prepared buffers.
"""
function find_sources!(
    extractor::SourceExtractor,
    A::AbstractMatrix,
)
    result =
        extract_sources!(
            extractor,
            A,
        )

    n =
        source_count(result)

    indices =
        Array(
            source_indices(result),
        )

    xs =
        Vector{Int}(
            undef,
            n,
        )

    ys =
        Vector{Int}(
            undef,
            n,
        )

    @inbounds for k in 1:n
        I = indices[k]

        ys[k] = I[1]
        xs[k] = I[2]
    end

    return xs, ys
end
