module SourceExtractionCUDAExt

using CUDA
using SourceExtraction

import SourceExtraction:
    DetectionWorkspace,
    LocalMaximaDetector,
    detect_sources!,
    source_count,
    radius

import SourceExtraction: extract_rois!

function DetectionWorkspace(
    A::CuArray{T,N};
    max_sources::Integer = 10_000,
) where {T,N}

    max_sources > 0 ||
        throw(ArgumentError("max_sources must be positive"))

    indices =
        CuArray{CartesianIndex{N}}(
            undef,
            max_sources,
        )

    scores =
        CuArray{T}(
            undef,
            max_sources,
        )

    count = CUDA.zeros(Int32, 1)

    return DetectionWorkspace(
        indices,
        scores,
        count,
    )
end



@inline function _detect_sources_2d_kernel!(
    indices,
    scores,
    count,
    A,
    detector::LocalMaximaDetector{R,Strict},
    border,
    capacity,
) where {R,Strict}

    ix =
        (blockIdx().x - 1) *
        blockDim().x +
        threadIdx().x

    iy =
        (blockIdx().y - 1) *
        blockDim().y +
        threadIdx().y

    nx, ny = size(A)

    if (
        ix <= border ||
        iy <= border ||
        ix > nx - border ||
        iy > ny - border
    )
        return
    end

    c = @inbounds A[ix, iy]

    c > detector.threshold || return

    @inbounds for dj in -R:R
        for di in -R:R

            (di == 0 && dj == 0) &&
                continue

            v = A[
                ix + di,
                iy + dj,
            ]

            if Strict
                c > v || return
            else
                c >= v || return
            end
        end
    end

    # Atomic compaction:
    #
    # atomic_add! returns the previous value.
    old = CUDA.atomic_add!(
        pointer(count),
        Int32(1),
    )

    slot = Int(old) + 1

    if slot <= capacity
        @inbounds indices[slot] =
            CartesianIndex(ix, iy)

        @inbounds scores[slot] = c
    end

    return
end

function detect_sources!(
    workspace::DetectionWorkspace{
        I,
        S,
        C,
    },
    A::CuArray{T,2},
    detector::LocalMaximaDetector{R};
    border::Integer = R,
) where {
    T,
    R,
    I<:CuArray,
    S<:CuArray,
    C<:CuArray,
}

    border >= R ||
        throw(
            ArgumentError(
                "border must be greater than or equal to the detection radius",
            ),
        )

    fill!(
        workspace.count,
        Int32(0),
    )

    nx, ny = size(A)

    threads = (16, 16)

    blocks = (
        cld(nx, threads[1]),
        cld(ny, threads[2]),
    )

    capacity =
        min(
            length(workspace.indices),
            length(workspace.scores),
        )

    @cuda threads=threads blocks=blocks _detect_sources_2d_kernel!(
            workspace.indices,
            workspace.scores,
            workspace.count,
            A,
            detector,
            border,
            capacity,
        )

    return workspace
end

function source_count(
    workspace::DetectionWorkspace{
        I,
        S,
        C,
    },
) where {
    I<:CuArray,
    S<:CuArray,
    C<:CuArray,
}

    n = CUDA.@allowscalar Int(
        workspace.count[1],
    )

    n <= length(workspace.indices) ||
        throw(
            ArgumentError(
                "source output buffer is too small: " *
                "$n sources detected for capacity " *
                "$(length(workspace.indices))",
            ),
        )

    return n
end


function _extract_rois_2d_kernel!(
    rois,
    A,
    indices,
    n,
    roi_nx,
    roi_ny,
)
    tid =
        (blockIdx().x - 1) *
        blockDim().x +
        threadIdx().x

    total = roi_nx * roi_ny * n

    if tid <= total

        q = tid - 1

        # ROI x coordinate
        ox = q % roi_nx + 1

        q ÷= roi_nx

        # ROI y coordinate
        oy = q % roi_ny + 1

        # Source number
        k = q ÷ roi_ny + 1

        rx = roi_nx ÷ 2
        ry = roi_ny ÷ 2

        center = @inbounds indices[k]

        cx = center[1]
        cy = center[2]

        x = cx + ox - rx - 1
        y = cy + oy - ry - 1

        @inbounds rois[ox, oy, k] = A[x, y]
    end

    return
end


function extract_rois!(
    rois::CuArray{TR,3},
    A::CuArray{TA,2},
    indices::CuArray{CartesianIndex{2},1},
    n::Integer,
) where {TR,TA}

    roi_nx, roi_ny, capacity = size(rois)

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

    n <= capacity ||
        throw(
            ArgumentError(
                "ROI output array is too small",
            ),
        )

    n <= length(indices) ||
        throw(
            ArgumentError(
                "source index array contains fewer than n entries",
            ),
        )

    n == 0 && return 0

    total = roi_nx * roi_ny * n

    threads = 256
    blocks = cld(total, threads)

    @cuda threads=threads blocks=blocks _extract_rois_2d_kernel!(
        rois,
        A,
        indices,
        n,
        roi_nx,
        roi_ny,
    )

    return n
end


function _detect_sources_3d_kernel!(
    indices,
    scores,
    count,
    A,
    detector::LocalMaximaDetector{R,Strict},
    border,
    capacity,
) where {R,Strict}

    ix =
        (blockIdx().x - 1) *
        blockDim().x +
        threadIdx().x

    iy =
        (blockIdx().y - 1) *
        blockDim().y +
        threadIdx().y

    t = blockIdx().z

    nx, ny, nt = size(A)

    if (
        ix <= border ||
        iy <= border ||
        ix > nx - border ||
        iy > ny - border ||
        t > nt
    )
        return
    end

    c = @inbounds A[ix, iy, t]

    c > detector.threshold || return

    @inbounds for dj in -R:R
        for di in -R:R

            (di == 0 && dj == 0) &&
                continue

            v = A[
                ix + di,
                iy + dj,
                t,
            ]

            if Strict
                c > v || return
            else
                c >= v || return
            end
        end
    end

    old = CUDA.atomic_add!(
        pointer(count),
        Int32(1),
    )

    slot = Int(old) + 1

    if slot <= capacity
        @inbounds indices[slot] =
            CartesianIndex(
                ix,
                iy,
                t,
            )

        @inbounds scores[slot] = c
    end

    return
end

function detect_sources!(
    workspace::DetectionWorkspace{I,S,C},
    A::CuArray{T,3},
    detector::LocalMaximaDetector{R};
    border::Integer = R,
) where {
    T,
    R,
    I<:CuArray,
    S<:CuArray,
    C<:CuArray,
}

    border >= R ||
        throw(ArgumentError(
            "border must be greater than or equal to the detection radius",
        ))

    fill!(
        workspace.count,
        Int32(0),
    )

    nx, ny, nt = size(A)

    threads = (16, 16, 1)

    blocks = (
        cld(nx, threads[1]),
        cld(ny, threads[2]),
        nt,
    )

    capacity = min(
        length(workspace.indices),
        length(workspace.scores),
    )

    @cuda threads=threads blocks=blocks _detect_sources_3d_kernel!(
        workspace.indices,
        workspace.scores,
        workspace.count,
        A,
        detector,
        border,
        capacity,
    )

    return workspace
end

function _extract_rois_frames_kernel!(
    rois,
    A,
    indices,
    n,
    roi_nx,
    roi_ny,
)
    tid =
        (blockIdx().x - 1) *
        blockDim().x +
        threadIdx().x

    total =
        roi_nx *
        roi_ny *
        n

    if tid <= total

        q = tid - 1

        ox = q % roi_nx + 1
        q ÷= roi_nx

        oy = q % roi_ny + 1
        k = q ÷ roi_ny + 1

        rx = roi_nx ÷ 2
        ry = roi_ny ÷ 2

        center =
            @inbounds indices[k]

        cx = center[1]
        cy = center[2]
        t  = center[3]

        x = cx + ox - rx - 1
        y = cy + oy - ry - 1

        @inbounds rois[
            ox,
            oy,
            k,
        ] = A[
            x,
            y,
            t,
        ]
    end

    return
end

function extract_rois!(
    rois::CuArray{TR,3},
    A::CuArray{TA,3},
    indices::CuArray{CartesianIndex{3},1},
    n::Integer,
) where {TR,TA}

    roi_nx, roi_ny, capacity =
        size(rois)

    n <= capacity ||
        throw(ArgumentError(
            "ROI output array is too small",
        ))

    n == 0 && return 0

    total =
        roi_nx *
        roi_ny *
        n

    threads = 256
    blocks = cld(total, threads)

    @cuda threads=threads blocks=blocks _extract_rois_frames_kernel!(
        rois,
        A,
        indices,
        n,
        roi_nx,
        roi_ny,
    )

    return n
end



end