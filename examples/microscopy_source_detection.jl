using SourceExtraction
using PointSpreadFunctions
using FourierTools
using View5D
using Noise
using Random

using ImageShow


function simulate_sources(
    rng::AbstractRNG,
    psf_array::AbstractMatrix;
    n_sources::Integer,
    source_flux_range=(1_000f0, 10_000f0),
    background_flux=10f0,
    psf_radius::Integer=32,
)
    n_sources >= 0 ||
        throw(ArgumentError("n_sources must be nonnegative"))

    psf_radius >= 0 ||
        throw(ArgumentError("psf_radius must be nonnegative"))

    flux_min, flux_max = source_flux_range

    0 <= flux_min <= flux_max ||
        throw(ArgumentError(
            "source_flux_range must contain nonnegative, increasing values",
        ))

    background_flux >= 0 ||
        throw(ArgumentError("background_flux must be nonnegative"))

    psf_center = argmax(psf_array)

    psf_rows =
        (psf_center[1] - psf_radius):(psf_center[1] + psf_radius)

    psf_columns =
        (psf_center[2] - psf_radius):(psf_center[2] + psf_radius)

    first(psf_rows) >= 1 && last(psf_rows) <= size(psf_array, 1) ||
        throw(ArgumentError("psf_radius exceeds the PSF array in dimension 1"))

    first(psf_columns) >= 1 && last(psf_columns) <= size(psf_array, 2) ||
        throw(ArgumentError("psf_radius exceeds the PSF array in dimension 2"))

    psf_stamp =
        Float32.(
            @view(
                psf_array[
                    psf_rows,
                    psf_columns,
                ]
            ),
        )

    psf_sum = sum(psf_stamp)

    psf_sum > 0 ||
        throw(ArgumentError("the cropped PSF must have positive total intensity"))

    # After normalization, source flux is the expected number of photons
    # contributed by that source.
    psf_stamp ./= psf_sum

    valid_rows =
        (1 + psf_radius):(size(psf_array, 1) - psf_radius)

    valid_columns =
        (1 + psf_radius):(size(psf_array, 2) - psf_radius)

    valid_positions =
        CartesianIndices((valid_rows, valid_columns))

    n_sources <= length(valid_positions) ||
        throw(ArgumentError(
            "n_sources exceeds the number of valid, distinct source positions",
        ))

    # randperm ensures that all simulated source centers are distinct.
    selected =
        randperm(rng, length(valid_positions))[1:n_sources]

    positions =
        [valid_positions[i] for i in selected]

    fluxes =
        Float32(flux_min) .+
        (Float32(flux_max) - Float32(flux_min)) .*
        rand(rng, Float32, n_sources)

    expected_photons =
        fill(
            Float32(background_flux),
            size(psf_array),
        )

    for (position, flux) in zip(positions, fluxes)
        rows =
            (position[1] - psf_radius):(position[1] + psf_radius)

        columns =
            (position[2] - psf_radius):(position[2] + psf_radius)

        @views expected_photons[rows, columns] .+= flux .* psf_stamp
    end

    # Each pixel is sampled with its expected photon count as Poisson mean.
    image = poisson(expected_photons)

    return (
        image=image,
        expected_photons=expected_photons,
        positions=positions,
        fluxes=fluxes,
        psf_stamp=psf_stamp,
    )
end


function detected_source_rings(
    image_size::NTuple{2,Int},
    xs,
    ys;
    radius::Real=7,
    thickness::Real=2,
)
    radius > 0 ||
        throw(ArgumentError("radius must be positive"))

    thickness > 0 ||
        throw(ArgumentError("thickness must be positive"))

    length(xs) == length(ys) ||
        throw(DimensionMismatch("xs and ys must have the same length"))

    inner_radius =
        max(0f0, Float32(radius - thickness / 2))

    outer_radius =
        Float32(radius + thickness / 2)

    padding = ceil(Int, outer_radius)

    padded_size =
        (
            image_size[1] + 2padding,
            image_size[2] + 2padding,
        )

    detected_impulses =
        zeros(Float32, padded_size)

    for (x, y) in zip(xs, ys)
        row = Int(y)
        column = Int(x)

        if 1 <= row <= image_size[1] && 1 <= column <= image_size[2]
            detected_impulses[
                row + padding,
                column + padding,
            ] += 0.1f0
        end
    end

    ring_psf =
        zeros(Float32, padded_size)

    center =
        CartesianIndex(
            padded_size .÷ 2 .+ 1,
        )

    inner_radius_squared = inner_radius^2
    outer_radius_squared = outer_radius^2

    for index in CartesianIndices(ring_psf)
        row_offset = index[1] - center[1]
        column_offset = index[2] - center[2]
        radius_squared = row_offset^2 + column_offset^2

        if inner_radius_squared <= radius_squared <= outer_radius_squared
            ring_psf[index] = 1f0
        end
    end

    padded_rings =
        FourierTools.conv_psf(
            detected_impulses,
            ring_psf,
        )

    rows =
        (1 + padding):(padding + image_size[1])

    columns =
        (1 + padding):(padding + image_size[2])

    # Round-off from the FFT can produce values just outside [0, 1].
    return clamp.(
        @view(padded_rings[rows, columns]),
        0f0,
        1f0,
    )
end


Random.seed!(1234)
rng = Random.default_rng()

n_sources = 20
source_flux_range = (500f0, 10_000f0)
background_flux = 10f0

λ = 0.550 # μm
NA = 1.4
refractive_index = 1.55
sampling = (0.1, 0.1)
sz = (512, 512)

pp = PSFParams(λ, NA, refractive_index)
mypsf = psf(sz, pp; sampling=sampling)

simulation =
    simulate_sources(
        rng,
        mypsf;
        n_sources=n_sources,
        source_flux_range=source_flux_range,
        background_flux=background_flux,
    )

img = simulation.image
expected_photons = simulation.expected_photons
truth_positions = simulation.positions
truth_fluxes = simulation.fluxes

@time xs, ys =
    find_sources(
        img;
        threshold_fraction=0.05,
        max_sources=10_000,
    );

ring_overlay =
    detected_source_rings(
        size(img),
        xs,
        ys;
        radius=11,
        thickness=2,
    )

normalized_img =
    img ./ maximum(img)

img_with_rings =
    max.(normalized_img, ring_overlay)

println("Simulated sources: ", length(truth_positions))
println("Detected sources:  ", length(xs))
println("Source flux range:  ", extrema(truth_fluxes), " photons")
println("Background flux:    ", background_flux, " photons/pixel")

if isinteractive()
    @vt mypsf expected_photons img ring_overlay img_with_rings
end

save("sample_beads_image_detected_sources.tiff", Float32.(simshow(img_with_rings, γ=0.3)))