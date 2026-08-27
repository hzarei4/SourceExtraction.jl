export DetectionWorkspace
export source_count

"""
    LocalMaximaDetector(threshold; radius=1, strict=true)

Detector for identifying local maxima above `threshold`.

A candidate pixel must be greater than all pixels inside a
`(2radius + 1) × (2radius + 1)` neighborhood when `strict=true`.

The detection radius is encoded in the detector type so that small
fixed neighborhoods can be specialized by the compiler.
"""
struct LocalMaximaDetector{R, Strict, T}
    threshold::T
end

function LocalMaximaDetector(
    threshold::T;
    radius::Integer = 1,
    strict::Bool = true,
) where {T}

    radius >= 1 ||
        throw(ArgumentError("radius must be at least 1"))

    return LocalMaximaDetector{Int(radius), strict, T}(threshold)
end

Base.show(io::IO, d::LocalMaximaDetector{R, Strict}) where {R, Strict} =
    print(
        io,
        "LocalMaximaDetector(",
        d.threshold,
        "; radius=",
        R,
        ", strict=",
        Strict,
        ")",
    )

@inline radius(::LocalMaximaDetector{R}) where {R} = R
@inline isstrict(::LocalMaximaDetector{R, Strict}) where {R, Strict} = Strict


@inline function _islocalmaximum(
    A,
    i,
    j,
    c,
    ::LocalMaximaDetector{R, Strict},
) where {R, Strict}

    @inbounds for dj in -R:R
        for di in -R:R

            (di == 0 && dj == 0) && continue

            v = A[i + di, j + dj]

            if Strict
                c > v || return false
            else
                c >= v || return false
            end
        end
    end

    return true
end


"""
    detect_sources!(
        indices,
        scores,
        A,
        detector;
        border=radius(detector),
    )

Detect local maxima in the 2-D array `A`.

Detected source positions are written into the preallocated `indices`
array and their corresponding image values into `scores`.

Returns the number of detected sources.

No allocations are required when the supplied output buffers have
sufficient capacity.
"""
function detect_sources!(
    indices::AbstractVector{CartesianIndex{2}},
    scores::AbstractVector,
    A::AbstractMatrix,
    detector::LocalMaximaDetector{R};
    border::Integer = R,
) where {R}

    border >= R ||
        throw(ArgumentError(
            "border must be greater than or equal to the detection radius",
        ))

    nx, ny = size(A)

    if nx <= 2 * border || ny <= 2 * border
        return 0
    end

    capacity = min(length(indices), length(scores))
    threshold = detector.threshold

    n = 0

    # Column-major traversal:
    # first dimension is the inner loop.
    @inbounds for j in (1 + border):(ny - border)
        for i in (1 + border):(nx - border)

            c = A[i, j]

            c > threshold || continue

            _islocalmaximum(A, i, j, c, detector) || continue

            if n == capacity
                throw(ArgumentError(
                    "source output buffer is too small",
                ))
            end

            n += 1

            indices[n] = CartesianIndex(i, j)
            scores[n] = c
        end
    end

    return n
end


"""
    DetectionWorkspace

Preallocated storage for detected source positions, scores, and count.

The underlying storage may reside on the CPU or another backend such
as CUDA.
"""
struct DetectionWorkspace{I,S,C}
    indices::I
    scores::S
    count::C
end

function DetectionWorkspace(
    A::AbstractArray{T,N};
    max_sources::Integer = 10_000,
) where {T,N}

    max_sources > 0 ||
        throw(ArgumentError("max_sources must be positive"))

    indices =
        Vector{CartesianIndex{N}}(
            undef,
            max_sources,
        )

    scores =
        Vector{T}(
            undef,
            max_sources,
        )

    count = Ref{Int}(0)

    return DetectionWorkspace(
        indices,
        scores,
        count,
    )
end


function detect_sources!(
    workspace::DetectionWorkspace,
    A::AbstractArray,
    detector::LocalMaximaDetector;
    border::Integer = radius(detector),
)
    n = detect_sources!(
        workspace.indices,
        workspace.scores,
        A,
        detector;
        border = border,
    )

    workspace.count[] = n

    return workspace
end


source_count(
    workspace::DetectionWorkspace,
) = workspace.count[]


@inline function _islocalmaximum_frame(
    A,
    i,
    j,
    t,
    c,
    ::LocalMaximaDetector{R,Strict},
) where {R,Strict}

    @inbounds for dj in -R:R
        for di in -R:R

            (di == 0 && dj == 0) && continue

            v = A[
                i + di,
                j + dj,
                t,
            ]

            if Strict
                c > v || return false
            else
                c >= v || return false
            end
        end
    end

    return true
end

"""
    detect_sources!(indices, scores, A::AbstractArray{T,3}, detector)

Detect sources independently in every frame of an `x × y × frame`
array.

Local-maxima comparisons are performed only within each frame.
"""
function detect_sources!(
    indices::AbstractVector{CartesianIndex{3}},
    scores::AbstractVector,
    A::AbstractArray{T,3},
    detector::LocalMaximaDetector{R};
    border::Integer = R,
) where {T,R}

    border >= R ||
        throw(ArgumentError(
            "border must be greater than or equal to the detection radius",
        ))

    nx, ny, nt = size(A)

    if nx <= 2border || ny <= 2border
        return 0
    end

    capacity =
        min(
            length(indices),
            length(scores),
        )

    threshold = detector.threshold

    n = 0

    @inbounds for t in 1:nt
        for j in (1 + border):(ny - border)
            for i in (1 + border):(nx - border)

                c = A[i, j, t]

                c > threshold || continue

                _islocalmaximum_frame(
                    A,
                    i,
                    j,
                    t,
                    c,
                    detector,
                ) || continue

                n == capacity &&
                    throw(ArgumentError(
                        "source output buffer is too small",
                    ))

                n += 1

                indices[n] =
                    CartesianIndex(
                        i,
                        j,
                        t,
                    )

                scores[n] = c
            end
        end
    end

    return n
end