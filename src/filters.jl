
function prepare_filter end
function filter_sources! end

"""
    DoG(sigma1, sigma2)

Difference-of-Gaussians filter specification.

`sigma1` is the width of the narrower Gaussian and `sigma2` the width
of the broader Gaussian.
"""
struct DoG{T<:Real}
    sigma1::T
    sigma2::T

    function DoG{T}(sigma1::T, sigma2::T) where {T<:Real}
        sigma1 > zero(T) ||
            throw(ArgumentError("sigma1 must be positive"))

        sigma2 > sigma1 ||
            throw(ArgumentError("sigma2 must be larger than sigma1"))

        return new{T}(sigma1, sigma2)
    end
end


function DoG(sigma1::Real, sigma2::Real)
    s1, s2 = promote(float(sigma1), float(sigma2))

    return DoG{typeof(s1)}(s1, s2)
end



struct PreparedDoG{G1,G2}
    g1::G1
    g2::G2
end


"""
    prepare_dog(array_type, size, dog)

Prepare the separable Gaussian components of a Difference-of-Gaussians
filter for a given array type and size.

This operation may allocate. The returned object is intended to be reused.
"""
function prepare_dog(
    array_type::Type,
    sz::NTuple{N,Int},
    dog::DoG,
) where {N}

    g1 = normal_sep(
        array_type,
        sz;
        sigma = dog.sigma1,
    )

    g2 = normal_sep(
        array_type,
        sz;
        sigma = dog.sigma2,
    )

    return PreparedDoG(g1, g2)
end


prepare_dog(A::AbstractArray, dog::DoG) =
    prepare_dog(typeof(A), size(A), dog)

"""
    dog_kernel!(dst, prepared)

Materialize a prepared Difference-of-Gaussians kernel into `dst`.
"""
function dog_kernel!(
    dst::AbstractArray,
    prepared::PreparedDoG,
)
    dst .= prepared.g1 .- prepared.g2
    return dst
end

function dog_kernel!(
    dst::AbstractArray,
    dog::DoG,
)
    prepared = prepare_dog(dst, dog)
    return dog_kernel!(dst, prepared)
end


struct PreparedDoGFilter{P}
    plan::P
end

"""
    prepare_filter(A, dog)

Prepare an FFT-based Difference-of-Gaussians filter for arrays
with the same size and type as `A`.

All FFT plans, Fourier-domain kernel data, and working buffers are
allocated during this step and reused during filtering.
"""
function prepare_filter(
    A::AbstractArray{T,N},
    dog::DoG;
    dims = (1, 2),
) where {T,N}

    N >= 2 ||
        throw(ArgumentError(
            "source filtering requires at least two dimensions",
        ))

    kernel = similar(
        A,
        T,
        (
            size(A, dims[1]),
            size(A, dims[2]),
        ),
    )

    prepared_dog =
        prepare_dog(
            kernel,
            dog,
        )

    dog_kernel!(
        kernel,
        prepared_dog,
    )

    _, plan =
        FourierTools.plan_conv_psf_buffer(
            A,
            kernel,
            dims,
        )

    return PreparedDoGFilter(plan)
end


"""
    filter_sources!(prepared, A)

Apply a prepared source-enhancement filter to `A`.

The returned array is an internal reusable buffer owned by `prepared`.
Its contents will be overwritten by the next call using the same
prepared filter.
"""
function filter_sources!(
    prepared::PreparedDoGFilter,
    A::AbstractArray,
)
    return prepared.plan(A)
end


"""
    filter_sources!(dst, A, prepared)

Apply the prepared filter and copy the result into `dst`.
"""
function filter_sources!(
    dst::AbstractArray,
    A::AbstractArray,
    prepared::PreparedDoGFilter,
)
    copyto!(
        dst,
        prepared.plan(A),
    )

    return dst
end