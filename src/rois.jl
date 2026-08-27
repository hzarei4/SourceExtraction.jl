"""
    extract_rois!(rois, A, indices, n)

Extract fixed-size 2-D ROIs centered on the first `n` source positions.

`rois` must have dimensions

    (roi_x, roi_y, max_sources)

and both ROI dimensions must be odd.

Returns `n`.
"""
function extract_rois!(
    rois::AbstractArray{TR, 3},
    A::AbstractMatrix,
    indices::AbstractVector{CartesianIndex{2}},
    n::Integer,
) where {TR}

    roi_nx, roi_ny, roi_capacity = size(rois)

    isodd(roi_nx) ||
        throw(ArgumentError("ROI size in dimension 1 must be odd"))

    isodd(roi_ny) ||
        throw(ArgumentError("ROI size in dimension 2 must be odd"))

    n <= roi_capacity ||
        throw(ArgumentError("ROI output array is too small"))

    n <= length(indices) ||
        throw(ArgumentError("source index array contains fewer than n entries"))

    rx = roi_nx ÷ 2
    ry = roi_ny ÷ 2

    nx, ny = size(A)

    @inbounds for k in 1:n

        I = indices[k]

        cx = I[1]
        cy = I[2]

        if (
            cx - rx < 1 ||
            cx + rx > nx ||
            cy - ry < 1 ||
            cy + ry > ny
        )
            throw(ArgumentError(
                "ROI centered at $I exceeds image bounds",
            ))
        end

        for dy in -ry:ry
            for dx in -rx:rx

                rois[
                    dx + rx + 1,
                    dy + ry + 1,
                    k,
                ] = A[
                    cx + dx,
                    cy + dy,
                ]
            end
        end
    end

    return n
end


function extract_rois!(
    rois::AbstractArray{TR,3},
    A::AbstractArray{TA,3},
    indices::AbstractVector{CartesianIndex{3}},
    n::Integer,
) where {TR,TA}

    roi_nx, roi_ny, capacity = size(rois)

    isodd(roi_nx) ||
        throw(ArgumentError(
            "ROI size in dimension 1 must be odd",
        ))

    isodd(roi_ny) ||
        throw(ArgumentError(
            "ROI size in dimension 2 must be odd",
        ))

    n <= capacity ||
        throw(ArgumentError(
            "ROI output array is too small",
        ))

    rx = roi_nx ÷ 2
    ry = roi_ny ÷ 2

    @inbounds for k in 1:n

        I = indices[k]

        cx = I[1]
        cy = I[2]
        t  = I[3]

        for dy in -ry:ry
            for dx in -rx:rx

                rois[
                    dx + rx + 1,
                    dy + ry + 1,
                    k,
                ] = A[
                    cx + dx,
                    cy + dy,
                    t,
                ]
            end
        end
    end

    return n
end

"""
    extract_rois!(rois, A, workspace::DetectionWorkspace, n)

Extract ROIs using source indices stored in a detection workspace.
"""
function extract_rois!(
    rois::AbstractArray{TR,3},
    A::AbstractArray{TA,N},
    workspace::DetectionWorkspace,
    n::Integer,
) where {TR,TA,N}

    return extract_rois!(
        rois,
        A,
        workspace.indices,
        n,
    )
end