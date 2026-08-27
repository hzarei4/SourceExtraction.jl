using SourceExtraction
using Random
using FileIO
using ImageIO
using ImageCore
using ImageCore: Gray, RGB

length(ARGS) >= 1 || error("""
usage:
    julia --project=examples examples/tycho2_detection_overlay.jl \
        <Tycho2.csv> [output-directory]

example:
    julia --project=examples examples/tycho2_detection_overlay.jl \
        D:/Hossein/Data/StarDatasets/Tycho2.csv \
        examples/data/tycho2_detection
""")

catalog_path = ARGS[1]

outdir =
    length(ARGS) >= 2 ?
    ARGS[2] :
    joinpath(
        @__DIR__,
        "data",
        "tycho2_detection",
    )

mkpath(outdir)


# =====================================================================
# Read Tycho2.csv
#
# Format used by the existing PlateSolving code:
#
#     column 1: RA [deg]
#     column 2: Dec [deg]
#     column 3: magnitude
#
# A textual header is skipped automatically.
# =====================================================================

function load_tycho2_csv(
    path::AbstractString,
)
    ra  = Float32[]
    dec = Float32[]
    mag = Float32[]

    for line in eachline(path)

        s = strip(line)

        isempty(s) && continue
        startswith(s, "#") && continue

        fields =
            occursin(',', s) ?
            split(s, ',') :
            occursin(';', s) ?
            split(s, ';') :
            split(s)

        length(fields) >= 3 || continue

        r =
            tryparse(
                Float32,
                strip(fields[2]),
            )

        d =
            tryparse(
                Float32,
                strip(fields[3]),
            )

        m =
            tryparse(
                Float32,
                strip(fields[4]),
            )

        # skips the textual header
        (
            isnothing(r) ||
            isnothing(d) ||
            isnothing(m)
        ) && continue

        push!(ra, r)
        push!(dec, d)
        push!(mag, m)
    end

    return (
        ra = ra,
        dec = dec,
        mag = mag,
    )
end


# =====================================================================
# Precompute catalogue unit vectors
# =====================================================================

function prepare_catalog(catalog)

    n = length(catalog.ra)

    x =
        Vector{Float32}(
            undef,
            n,
        )

    y = similar(x)
    z = similar(x)

    @inbounds for i in 1:n

        ra =
            deg2rad(
                catalog.ra[i],
            )

        dec =
            deg2rad(
                catalog.dec[i],
            )

        cd = cos(dec)

        x[i] = cd * cos(ra)
        y[i] = cd * sin(ra)
        z[i] = sin(dec)
    end

    return (
        x = x,
        y = y,
        z = z,
        mag = catalog.mag,
    )
end


# =====================================================================
# TAN projection
# =====================================================================

function project_field(
    catalog,
    ra0_deg,
    dec0_deg,
    roll_deg,
    width,
    height,
    fov_width_deg,
)

    ra0 =
        deg2rad(
            Float32(ra0_deg),
        )

    dec0 =
        deg2rad(
            Float32(dec0_deg),
        )

    # pointing direction
    cx =
        cos(dec0) *
        cos(ra0)

    cy =
        cos(dec0) *
        sin(ra0)

    cz =
        sin(dec0)

    # increasing RA
    ex =
        -sin(ra0)

    ey =
        cos(ra0)

    ez =
        0f0

    # increasing Dec
    nx =
        -sin(dec0) *
        cos(ra0)

    ny =
        -sin(dec0) *
        sin(ra0)

    nz =
        cos(dec0)

    roll =
        deg2rad(
            Float32(roll_deg),
        )

    c = cos(roll)
    s = sin(roll)

    scale =
        (Float32(width) - 1f0) /
        (
            2f0 *
            tan(
                deg2rad(
                    Float32(fov_width_deg),
                ) / 2f0,
            )
        )

    center_x =
        (Float32(width) + 1f0) /
        2f0

    center_y =
        (Float32(height) + 1f0) /
        2f0

    xs = Float32[]
    ys = Float32[]
    mags = Float32[]

    @inbounds for i in eachindex(catalog.x)

        sx = catalog.x[i]
        sy = catalog.y[i]
        sz = catalog.z[i]

        denom =
            sx * cx +
            sy * cy +
            sz * cz

        # behind the camera
        denom > 0f0 || continue

        tx =
            (
                sx * ex +
                sy * ey +
                sz * ez
            ) / denom

        ty =
            (
                sx * nx +
                sy * ny +
                sz * nz
            ) / denom

        # camera roll
        u =
            c * tx -
            s * ty

        v =
            s * tx +
            c * ty

        px =
            center_x +
            scale * u

        py =
            center_y +
            scale * v

        if (
            1f0 <= px <= width &&
            1f0 <= py <= height
        )
            push!(xs, px)
            push!(ys, py)
            push!(
                mags,
                catalog.mag[i],
            )
        end
    end

    return (
        x = xs,
        y = ys,
        magnitude = mags,
    )
end


# =====================================================================
# Render synthetic Tycho field
# =====================================================================

function add_gaussian_star!(
    image,
    cx,
    cy,
    peak,
    sigma,
)

    radius =
        max(
            3,
            ceil(
                Int,
                4sigma,
            ),
        )

    xmin =
        max(
            1,
            floor(Int, cx) - radius,
        )

    xmax =
        min(
            size(image, 2),
            ceil(Int, cx) + radius,
        )

    ymin =
        max(
            1,
            floor(Int, cy) - radius,
        )

    ymax =
        min(
            size(image, 1),
            ceil(Int, cy) + radius,
        )

    q =
        -0.5f0 /
        Float32(sigma)^2

    @inbounds for y in ymin:ymax

        dy =
            Float32(y) -
            Float32(cy)

        for x in xmin:xmax

            dx =
                Float32(x) -
                Float32(cx)

            image[y, x] +=
                Float32(peak) *
                exp(
                    (
                        dx * dx +
                        dy * dy
                    ) * q,
                )
        end
    end

    return image
end


function render_field(
    rng,
    stars;
    width,
    height,
    max_stars = 300,
    psf_sigma = 1.2f0,
    background = 0.02f0,
    noise_sigma = 0.002f0,
)

    image =
        fill(
            Float32(background),
            height,
            width,
        )

    isempty(stars.x) &&
        return image

    # brightest stars first
    order =
        sortperm(
            stars.magnitude,
        )

    nkeep =
        min(
            max_stars,
            length(order),
        )

    keep =
        @view order[1:nkeep]

    brightest =
        minimum(
            stars.magnitude[i]
            for i in keep
        )

    @inbounds for i in keep

        magnitude =
            stars.magnitude[i]

        relative_flux =
            10f0^(
                -0.4f0 *
                (
                    magnitude -
                    brightest
                )
            )

        peak =
            clamp(
                0.9f0 *
                relative_flux,
                0.01f0,
                0.9f0,
            )

        # small PSF variation
        sigma =
            psf_sigma *
            clamp(
                1f0 +
                0.08f0 *
                randn(rng, Float32),
                0.8f0,
                1.25f0,
            )

        add_gaussian_star!(
            image,
            stars.x[i],
            stars.y[i],
            peak,
            sigma,
        )
    end

    @inbounds for i in eachindex(image)

        image[i] +=
            noise_sigma *
            randn(
                rng,
                Float32,
            )

        image[i] =
            clamp(
                image[i],
                0f0,
                1f0,
            )
    end

    return image
end


# =====================================================================
# Ring overlay
# =====================================================================

function ring_overlay(
    image,
    xs,
    ys;
    radius = 7,
    thickness = 2,
)

    height, width =
        size(image)

    overlay =
        Matrix{RGB{Float32}}(
            undef,
            height,
            width,
        )

    @inbounds for i in eachindex(image)

        v =
            clamp(
                Float32(image[i]),
                0f0,
                1f0,
            )

        overlay[i] =
            RGB{Float32}(
                v,
                v,
                v,
            )
    end

    inner2 =
        (
            radius -
            thickness / 2
        )^2

    outer2 =
        (
            radius +
            thickness / 2
        )^2

    @inbounds for k in eachindex(xs)

        cx =
            round(
                Int,
                xs[k],
            )

        cy =
            round(
                Int,
                ys[k],
            )

        xmin =
            max(
                1,
                cx - radius - thickness,
            )

        xmax =
            min(
                width,
                cx + radius + thickness,
            )

        ymin =
            max(
                1,
                cy - radius - thickness,
            )

        ymax =
            min(
                height,
                cy + radius + thickness,
            )

        for y in ymin:ymax

            dy =
                y - cy

            for x in xmin:xmax

                dx =
                    x - cx

                r2 =
                    dx * dx +
                    dy * dy

                if inner2 <= r2 <= outer2

                    overlay[y, x] =
                        RGB{Float32}(
                            1f0,
                            0f0,
                            0f0,
                        )
                end
            end
        end
    end

    return overlay
end


# =====================================================================
# Main example
# =====================================================================

println(
    "Loading Tycho2 catalogue...",
)

catalog_raw =
    load_tycho2_csv(
        catalog_path,
    )

println(
    "catalogue stars = ",
    length(catalog_raw.ra),
)

catalog =
    prepare_catalog(
        catalog_raw,
    )

rng =
    Xoshiro(
        1234,
    )

const NFRAMES = 50
const WIDTH = 512
const HEIGHT = 512
const FOV_DEG = 12f0

for frame_id in 1:NFRAMES

    local stars
    local ra
    local dec
    local roll

    # Pick a random sky field with enough Tycho stars.
    while true

        ra =
            360f0 *
            rand(
                rng,
                Float32,
            )

        # uniform sampling over the sphere
        dec =
            asind(
                2f0 *
                rand(
                    rng,
                    Float32,
                ) -
                1f0,
            )

        roll =
            360f0 *
            rand(
                rng,
                Float32,
            )

        stars =
            project_field(
                catalog,
                ra,
                dec,
                roll,
                WIDTH,
                HEIGHT,
                FOV_DEG,
            )

        length(stars.x) >= 40 &&
            break
    end

    frame =
        render_field(
            rng,
            stars;
            width = WIDTH,
            height = HEIGHT,
        )

    # ================================================================
    # SourceExtraction.jl
    #
    # This is the entire source-detection API.
    # ================================================================

    xs, ys =
        find_sources(frame, threshold_fraction = 0.05)

    # ================================================================

    frame_name =
        "frame_" *
        lpad(
            string(frame_id),
            3,
            '0',
        )

    save(
        joinpath(
            outdir,
            frame_name *
            ".png",
        ),
        Gray.(
            frame,
        ),
    )

    overlay =
        ring_overlay(
            frame,
            xs,
            ys,
        )

    save(
        joinpath(
            outdir,
            frame_name *
            "_sources.png",
        ),
        overlay,
    )

    println(
        frame_name,
        ": catalog stars = ",
        length(stars.x),
        ", detected = ",
        length(xs),
        ", center = (",
        round(
            ra;
            digits = 2,
        ),
        "°, ",
        round(
            dec;
            digits = 2,
        ),
        "°)",
    )
end

println()
println(
    "Generated $NFRAMES Tycho2 frames and source overlays in:"
)
println(
    abspath(outdir),
)