# astronomy_source_extraction_comparison.jl
#
# Practical astronomy source-extraction benchmark:
#
#   SourceExtraction.jl
#   SEP (Python/C library based on SExtractor)
#   SExtractor CLI (via WSL on Windows if necessary)
#
# The methods do NOT implement identical source definitions, so this is a
# practical pipeline comparison rather than an exact algorithmic benchmark.
#
# Run:
#   julia --project=benchmark benchmark\astronomy_source_extraction_comparison.jl
#
# Optional:
#   julia --project=benchmark benchmark\astronomy_source_extraction_comparison.jl 4096 5000

using SourceExtraction
using BenchmarkTools
using FITSIO
using FFTW
using Random
using Statistics
using Printf

# ============================================================================
# Configuration
# ============================================================================

const N = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 2048
const NSTARS = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 1500

const NSIGMA = 5f0
const BACKGROUND = 0.02f0
const NOISE_SIGMA = 0.005f0
const PSF_SIGMA = 1.2f0

const DOG = DoG(1f0, 3f0)

const JULIA_SAMPLES = 30
const EXTERNAL_SAMPLES = 10

# Use the same FFTW setting as the earlier SourceExtraction CPU benchmark.
const FFTW_THREADS = min(Sys.CPU_THREADS, 16)

const RNG_SEED = 0x5E17A57

# ============================================================================
# Synthetic astronomical image
# ============================================================================

function add_gaussian_star!(
    image::AbstractMatrix{T},
    x::T,
    y::T,
    peak::T,
    sigma::T,
) where {T<:AbstractFloat}

    r = max(3, ceil(Int, 4sigma))

    xmin = max(1, floor(Int, x) - r)
    xmax = min(size(image, 2), ceil(Int, x) + r)
    ymin = max(1, floor(Int, y) - r)
    ymax = min(size(image, 1), ceil(Int, y) + r)

    q = -T(0.5) / sigma^2

    @inbounds for iy in ymin:ymax
        dy = T(iy) - y

        for ix in xmin:xmax
            dx = T(ix) - x

            image[iy, ix] +=
                peak * exp((dx * dx + dy * dy) * q)
        end
    end

    return image
end


function make_star_field(
    rng;
    n = N,
    nstars = NSTARS,
    background = BACKGROUND,
    noise_sigma = NOISE_SIGMA,
    psf_sigma = PSF_SIGMA,
)
    image = fill(Float32(background), n, n)

    for _ in 1:nstars
        x = 10f0 + (n - 20f0) * rand(rng, Float32)
        y = 10f0 + (n - 20f0) * rand(rng, Float32)

        # Peak brightness ~0.03 ... 1 above background.
        u = rand(rng, Float32)
        peak = 10f0^(-1.5f0 * u)

        sigma =
            psf_sigma *
            clamp(
                1f0 + 0.08f0 * randn(rng, Float32),
                0.8f0,
                1.25f0,
            )

        add_gaussian_star!(
            image,
            x,
            y,
            peak,
            sigma,
        )
    end

    @inbounds for i in eachindex(image)
        image[i] +=
            noise_sigma * randn(rng, Float32)
    end

    return image
end


function make_noise_field(
    rng;
    n = N,
    background = BACKGROUND,
    noise_sigma = NOISE_SIGMA,
)
    image = fill(Float32(background), n, n)

    @inbounds for i in eachindex(image)
        image[i] +=
            noise_sigma * randn(rng, Float32)
    end

    return image
end

# ============================================================================
# Utilities
# ============================================================================

@inline megapixels(A) = length(A) / 1e6
@inline ms_from_ns(ns) = ns / 1e6

function bench_stats(trial)
    b = minimum(trial)
    m = median(trial)

    return (
        min_ms = ms_from_ns(b.time),
        median_ms = ms_from_ns(m.time),
        bytes = b.memory,
        allocs = b.allocs,
    )
end


@inline throughput_mpix_s(A, ms) =
    megapixels(A) / (ms / 1000)


function print_row(
    name;
    min_ms,
    median_ms,
    count,
    mpix_s,
    memory = nothing,
)
    memtext =
        isnothing(memory) ?
        "-" :
        memory == 0 ?
        "0 B" :
        @sprintf("%.2f MB", memory / 1e6)

    @printf(
        "%-38s %10.3f %10.3f %12d %12.1f %12s\n",
        name,
        min_ms,
        median_ms,
        count,
        mpix_s,
        memtext,
    )
end


function parse_keyvalues(text)
    result = Dict{String,String}()

    for line in split(text, '\n')
        occursin('=', line) || continue

        k, v = split(line, '='; limit=2)
        result[strip(k)] = strip(v)
    end

    return result
end


function succeeds(cmd::Cmd)
    try
        return success(
            pipeline(
                cmd;
                stdout = devnull,
                stderr = devnull,
            ),
        )
    catch
        return false
    end
end

# ============================================================================
# Robust Python 3.13 + SEP discovery
# ============================================================================

"""
Return the argv prefix for a Python interpreter that can import SEP.

On Windows we prefer:
    py -3.13
because SEP currently has a Windows wheel for Python 3.13 while a plain
`python` command may still point to the Microsoft Store execution alias.
"""
function find_sep_python()
    candidates =
        Vector{Vector{String}}()

    # Most robust option: allow the user to provide the exact interpreter.
    #
    # PowerShell example:
    #   $env:SEP_PYTHON = py -3.13 -c "import sys; print(sys.executable)"
    if haskey(ENV, "SEP_PYTHON")
        exe = strip(ENV["SEP_PYTHON"])

        !isempty(exe) &&
            push!(
                candidates,
                [exe],
            )
    end

    # Do not require Sys.which("py") here. Modern Windows Python launchers /
    # app aliases are not always discovered the same way by Julia's Sys.which,
    # even when `py -3.13` works from PowerShell.
    append!(
        candidates,
        [
            ["py", "-3.13"],
            ["py", "-3.12"],
            ["python"],
            ["python3"],
        ],
    )

    for prefix in candidates
        probe =
            Cmd(
                vcat(
                    prefix,
                    [
                        "-c",
                        "import sep, numpy; print(sep.__version__)",
                    ],
                ),
            )

        succeeds(probe) &&
            return prefix
    end

    return nothing
end

# ============================================================================
# Windows -> WSL path conversion
# ============================================================================

function windows_to_wsl_path(path::AbstractString)
    p =
        abspath(
            path,
        )

    m =
        match(
            r"^([A-Za-z]):[\\/](.*)$",
            p,
        )

    isnothing(m) &&
        return replace(
            p,
            '\\' => '/',
        )

    drive =
        lowercase(
            m.captures[1],
        )

    rest =
        replace(
            m.captures[2],
            '\\' => '/',
        )

    return "/mnt/$drive/$rest"
end

# ============================================================================
# SourceExtraction.jl
# ============================================================================

function benchmark_sourceextraction(
    image,
    noise_image,
)
    # FFTW thread count is independent of Threads.nthreads().
    FFTW.set_num_threads(
        FFTW_THREADS,
    )

    # IMPORTANT: create the FFT plan after setting the thread count.
    prepared =
        prepare_filter(
            image,
            DOG,
        )

    filtered_noise =
        copy(
            filter_sources!(
                prepared,
                noise_image,
            ),
        )

    threshold =
        NSIGMA *
        std(filtered_noise)

    filtered =
        copy(
            filter_sources!(
                prepared,
                image,
            ),
        )

    workspace =
        DetectionWorkspace(
            image;
            max_sources = length(image),
        )

    detector =
        LocalMaximaDetector(
            Float32(threshold);
            radius = 1,
            strict = true,
        )

    detect_sources!(
        workspace,
        filtered,
        detector;
        border = 1,
    )

    n =
        source_count(
            workspace,
        )

    detection_trial =
        @benchmark detect_sources!(
            $workspace,
            $filtered,
            $detector;
            border = 1,
        ) samples=JULIA_SAMPLES evals=1

    detection =
        bench_stats(
            detection_trial,
        )

    pipeline_trial =
        @benchmark begin
            filtered_local =
                filter_sources!(
                    $prepared,
                    $image,
                )

            detect_sources!(
                $workspace,
                filtered_local,
                $detector;
                border = 1,
            )
        end samples=JULIA_SAMPLES evals=1

    pipeline =
        bench_stats(
            pipeline_trial,
        )

    return (
        threshold = Float32(threshold),
        filtered = filtered,
        count = n,
        detection = detection,
        pipeline = pipeline,
    )
end

# ============================================================================
# SEP
# ============================================================================

function benchmark_sep(
    image,
    filtered,
    threshold,
    tmpdir,
)
    python_prefix =
        find_sep_python()

    isnothing(python_prefix) &&
        return (
            available = false,
            error = "no Python interpreter capable of `import sep` was found",
        )

    raw_path =
        joinpath(
            tmpdir,
            "image_f32.bin",
        )

    filtered_path =
        joinpath(
            tmpdir,
            "filtered_f32.bin",
        )

    open(raw_path, "w") do io
        write(io, image)
    end

    open(filtered_path, "w") do io
        write(io, filtered)
    end

    helper =
        joinpath(
            tmpdir,
            "sep_benchmark.py",
        )

    helper_code = raw"""
import sys
import time
import statistics

import numpy as np
import sep

n = int(sys.argv[1])
raw_path = sys.argv[2]
filtered_path = sys.argv[3]
dog_threshold = float(sys.argv[4])
nsigma = float(sys.argv[5])
noise_sigma = float(sys.argv[6])
samples = int(sys.argv[7])

# Julia matrices are column-major. Reconstruct the numerical image correctly,
# then make a C-contiguous copy for SEP.
raw = np.fromfile(
    raw_path,
    dtype=np.float32,
).reshape(
    (n, n),
    order="F",
)
raw = np.ascontiguousarray(raw)

filtered = np.fromfile(
    filtered_path,
    dtype=np.float32,
).reshape(
    (n, n),
    order="F",
)
filtered = np.ascontiguousarray(filtered)

# ----------------------------------------------------------------------
# 1. SEP minimal on the exact same DoG-filtered image and numerical
#    threshold as SourceExtraction.
#
# SEP still performs connected-component extraction + measurements,
# so this is not algorithmically identical to strict local maxima.
# ----------------------------------------------------------------------

def sep_minimal():
    return sep.extract(
        filtered,
        dog_threshold,
        minarea=1,
        filter_kernel=None,
        deblend_nthresh=1,
        deblend_cont=1.0,
        clean=False,
    )

objects_minimal = sep_minimal()

times = []
for _ in range(samples):
    t0 = time.perf_counter_ns()
    objects_minimal = sep_minimal()
    t1 = time.perf_counter_ns()
    times.append((t1 - t0) / 1e6)

minimal_min = min(times)
minimal_med = statistics.median(times)

# ----------------------------------------------------------------------
# 2. Standard SEP extract on the raw image.
#
# With err supplied, threshold is relative to the noise estimate.
# ----------------------------------------------------------------------

def sep_default():
    return sep.extract(
        raw,
        nsigma,
        err=noise_sigma,
    )

objects_default = sep_default()

times = []
for _ in range(samples):
    t0 = time.perf_counter_ns()
    objects_default = sep_default()
    t1 = time.perf_counter_ns()
    times.append((t1 - t0) / 1e6)

default_min = min(times)
default_med = statistics.median(times)

# ----------------------------------------------------------------------
# 3. SEP background estimation + extraction.
# ----------------------------------------------------------------------

def sep_background_pipeline():
    bkg = sep.Background(raw)

    # subtraction creates a temporary array, deliberately included because
    # this benchmark represents the practical background+extract pipeline.
    data_sub = raw - bkg

    return sep.extract(
        data_sub,
        nsigma,
        err=bkg.globalrms,
    )

objects_background = sep_background_pipeline()

times = []
for _ in range(samples):
    t0 = time.perf_counter_ns()
    objects_background = sep_background_pipeline()
    t1 = time.perf_counter_ns()
    times.append((t1 - t0) / 1e6)

background_min = min(times)
background_med = statistics.median(times)

print("SEP_AVAILABLE=1")
print("SEP_VERSION=" + str(sep.__version__))

print("SEP_MINIMAL_COUNT=" + str(len(objects_minimal)))
print("SEP_MINIMAL_MIN_MS=" + str(minimal_min))
print("SEP_MINIMAL_MEDIAN_MS=" + str(minimal_med))

print("SEP_DEFAULT_COUNT=" + str(len(objects_default)))
print("SEP_DEFAULT_MIN_MS=" + str(default_min))
print("SEP_DEFAULT_MEDIAN_MS=" + str(default_med))

print("SEP_BACKGROUND_COUNT=" + str(len(objects_background)))
print("SEP_BACKGROUND_MIN_MS=" + str(background_min))
print("SEP_BACKGROUND_MEDIAN_MS=" + str(background_med))
"""

    write(
        helper,
        helper_code,
    )

    args =
        vcat(
            python_prefix,
            [
                helper,
                string(N),
                raw_path,
                filtered_path,
                string(threshold),
                string(NSIGMA),
                string(NOISE_SIGMA),
                string(EXTERNAL_SAMPLES),
            ],
        )

    output =
        read(
            Cmd(args),
            String,
        )

    kv =
        parse_keyvalues(
            output,
        )

    return (
        available = true,
        python = join(python_prefix, " "),
        version = kv["SEP_VERSION"],

        minimal_count = parse(Int, kv["SEP_MINIMAL_COUNT"]),
        minimal_min_ms = parse(Float64, kv["SEP_MINIMAL_MIN_MS"]),
        minimal_median_ms = parse(Float64, kv["SEP_MINIMAL_MEDIAN_MS"]),

        default_count = parse(Int, kv["SEP_DEFAULT_COUNT"]),
        default_min_ms = parse(Float64, kv["SEP_DEFAULT_MIN_MS"]),
        default_median_ms = parse(Float64, kv["SEP_DEFAULT_MEDIAN_MS"]),

        background_count = parse(Int, kv["SEP_BACKGROUND_COUNT"]),
        background_min_ms = parse(Float64, kv["SEP_BACKGROUND_MIN_MS"]),
        background_median_ms = parse(Float64, kv["SEP_BACKGROUND_MEDIAN_MS"]),
    )
end

# ============================================================================
# SExtractor CLI, including WSL support on Windows
# ============================================================================

function find_native_sextractor()
    for name in (
        "source-extractor",
        "sex",
        "sextractor",
    )
        if (p = Sys.which(name)) !== nothing
            return (
                mode = :native,
                executable = String(p),
            )
        end
    end

    return nothing
end


function find_wsl_sextractor()
    Sys.iswindows() ||
        return nothing

    wsl =
        Sys.which("wsl")

    isnothing(wsl) &&
        return nothing

    query =
        Cmd(
            [
                String(wsl),
                "sh",
                "-lc",
                "command -v source-extractor || command -v sex || command -v sextractor",
            ],
        )

    output =
        try
            strip(
                read(
                    query,
                    String,
                ),
            )
        catch
            ""
        end

    isempty(output) &&
        return nothing

    executable =
        first(
            split(
                output,
                '\n',
            ),
        )

    return (
        mode = :wsl,
        wsl = String(wsl),
        executable = strip(executable),
    )
end


function find_sextractor()
    native =
        find_native_sextractor()

    !isnothing(native) &&
        return native

    return find_wsl_sextractor()
end


function catalog_source_count(path)
    n = 0

    for line in eachline(path)
        s = strip(line)

        isempty(s) && continue
        startswith(s, "#") && continue

        n += 1
    end

    return n
end


function benchmark_sextractor(
    image,
    tmpdir,
)
    sex =
        find_sextractor()

    isnothing(sex) &&
        return nothing

    fits_path =
        joinpath(
            tmpdir,
            "benchmark_image.fits",
        )

    FITSIO.fitswrite(
        fits_path,
        image,
    )

    params_path =
        joinpath(
            tmpdir,
            "benchmark.param",
        )

    write(
        params_path,
        "X_IMAGE\nY_IMAGE\n",
    )

    filter_path =
        joinpath(
            tmpdir,
            "benchmark.conv",
        )

    write(
        filter_path,
        """
CONV NORM
1 2 1
2 4 2
1 2 1
""",
    )

    catalog_path =
        joinpath(
            tmpdir,
            "sextractor.cat",
        )

    if sex.mode === :native
        args = String[
            sex.executable,
            fits_path,
            "-CATALOG_NAME", catalog_path,
            "-CATALOG_TYPE", "ASCII_HEAD",
            "-PARAMETERS_NAME", params_path,
            "-DETECT_MINAREA", "5",
            "-DETECT_THRESH", string(NSIGMA),
            "-ANALYSIS_THRESH", string(NSIGMA),
            "-FILTER", "Y",
            "-FILTER_NAME", filter_path,
            "-DEBLEND_NTHRESH", "32",
            "-DEBLEND_MINCONT", "0.005",
            "-CLEAN", "Y",
            "-CHECKIMAGE_TYPE", "NONE",
            "-WRITE_XML", "N",
            "-VERBOSE_TYPE", "QUIET",
        ]

        command = Cmd(args)

        # Warm-up.
        run(
            pipeline(
                command;
                stdout = devnull,
                stderr = devnull,
            ),
        )

        count =
            catalog_source_count(
                catalog_path,
            )

        times =
            Float64[]

        for _ in 1:EXTERNAL_SAMPLES
            t =
                @elapsed run(
                    pipeline(
                        command;
                        stdout = devnull,
                        stderr = devnull,
                    ),
                )

            push!(
                times,
                1000t,
            )
        end

        return (
            mode = :native,
            executable = sex.executable,
            count = count,
            min_ms = minimum(times),
            median_ms = median(times),
        )
    end

    # -----------------------------------------------------------------------
    # Fairer WSL benchmark
    #
    # Copy all inputs ONCE to WSL-native /tmp, then launch ONE WSL-side
    # Python timing harness. The Python process measures only each SExtractor
    # subprocess with time.perf_counter_ns().
    #
    # Excluded from each timing:
    #   - wsl.exe startup
    #   - Windows <-> WSL file copying
    #   - Python wrapper startup
    #   - FITS creation
    #
    # Included:
    #   - SExtractor process startup
    #   - FITS reading
    #   - background/filter/detect/deblend/measure
    #   - catalogue writing
    # -----------------------------------------------------------------------

    fits_wsl_src =
        windows_to_wsl_path(
            fits_path,
        )

    params_wsl_src =
        windows_to_wsl_path(
            params_path,
        )

    filter_wsl_src =
        windows_to_wsl_path(
            filter_path,
        )

    wrapper_path =
        joinpath(
            tmpdir,
            "sextractor_benchmark.py",
        )

    wrapper_code = raw"""
import sys
import time
import subprocess

sex = sys.argv[1]
fits_path = sys.argv[2]
catalog_path = sys.argv[3]
params_path = sys.argv[4]
filter_path = sys.argv[5]
nsigma = sys.argv[6]
samples = int(sys.argv[7])

cmd = [
    sex,
    fits_path,
    "-CATALOG_NAME", catalog_path,
    "-CATALOG_TYPE", "ASCII_HEAD",
    "-PARAMETERS_NAME", params_path,
    "-DETECT_MINAREA", "5",
    "-DETECT_THRESH", nsigma,
    "-ANALYSIS_THRESH", nsigma,
    "-FILTER", "Y",
    "-FILTER_NAME", filter_path,
    "-DEBLEND_NTHRESH", "32",
    "-DEBLEND_MINCONT", "0.005",
    "-CLEAN", "Y",
    "-CHECKIMAGE_TYPE", "NONE",
    "-WRITE_XML", "N",
    "-VERBOSE_TYPE", "QUIET",
]

# Warm-up outside timing.
subprocess.run(
    cmd,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    check=True,
)

times = []

for _ in range(samples):
    t0 = time.perf_counter_ns()

    subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )

    t1 = time.perf_counter_ns()
    times.append(t1 - t0)

with open(catalog_path, "r", encoding="utf-8", errors="replace") as f:
    count = sum(
        1
        for line in f
        if line.strip() and not line.lstrip().startswith("#")
    )

for t in times:
    print(f"TIME_NS={t}")

print(f"COUNT={count}")
"""

    write(
        wrapper_path,
        wrapper_code,
    )

    wrapper_wsl_src =
        windows_to_wsl_path(
            wrapper_path,
        )

    wsl_dir =
        "/tmp/sourceextraction_bench_" *
        string(
            getpid(),
        )

    # Prepare a native-Linux temporary directory. All of this is outside
    # the benchmark timing.
    try
        run(
            pipeline(
                Cmd(
                    [
                        sex.wsl,
                        "rm",
                        "-rf",
                        wsl_dir,
                    ],
                );
                stdout = devnull,
                stderr = devnull,
            ),
        )
    catch
        # Directory may not exist yet.
    end

    run(
        Cmd(
            [
                sex.wsl,
                "mkdir",
                "-p",
                wsl_dir,
            ],
        ),
    )

    copies = (
        (fits_wsl_src, "$wsl_dir/image.fits"),
        (params_wsl_src, "$wsl_dir/benchmark.param"),
        (filter_wsl_src, "$wsl_dir/benchmark.conv"),
        (wrapper_wsl_src, "$wsl_dir/sextractor_benchmark.py"),
    )

    for (src_wsl, dst_wsl) in copies
        run(
            Cmd(
                [
                    sex.wsl,
                    "cp",
                    src_wsl,
                    dst_wsl,
                ],
            ),
        )
    end

    # Find Python inside WSL. This is only the timing harness; it is not
    # involved in the measured SExtractor execution itself.
    python_wsl =
        try
            strip(
                read(
                    Cmd(
                        [
                            sex.wsl,
                            "sh",
                            "-lc",
                            "command -v python3 || command -v python",
                        ],
                    ),
                    String,
                ),
            )
        catch
            ""
        end

    isempty(python_wsl) &&
        error(
            "Python was not found inside WSL for the SExtractor timing harness",
        )

    output =
        try
            read(
                Cmd(
                    [
                        sex.wsl,
                        python_wsl,
                        "$wsl_dir/sextractor_benchmark.py",
                        sex.executable,
                        "$wsl_dir/image.fits",
                        "$wsl_dir/sextractor.cat",
                        "$wsl_dir/benchmark.param",
                        "$wsl_dir/benchmark.conv",
                        string(NSIGMA),
                        string(EXTERNAL_SAMPLES),
                    ],
                ),
                String,
            )
        finally
            try
                run(
                    pipeline(
                        Cmd(
                            [
                                sex.wsl,
                                "rm",
                                "-rf",
                                wsl_dir,
                            ],
                        );
                        stdout = devnull,
                        stderr = devnull,
                    ),
                )
            catch
            end
        end

    times_ms =
        Float64[]

    count =
        nothing

    for line in split(
        output,
        '\n',
    )
        if startswith(
            line,
            "TIME_NS=",
        )
            value =
                strip(
                    split(
                        line,
                        '=';
                        limit = 2,
                    )[2],
                )

            isempty(value) &&
                continue

            ns =
                parse(
                    Int64,
                    value,
                )

            push!(
                times_ms,
                ns / 1e6,
            )

        elseif startswith(
            line,
            "COUNT=",
        )
            value =
                strip(
                    split(
                        line,
                        '=';
                        limit = 2,
                    )[2],
                )

            isempty(value) &&
                continue

            count =
                parse(
                    Int,
                    value,
                )
        end
    end

    length(times_ms) == EXTERNAL_SAMPLES ||
        error(
            "Expected $EXTERNAL_SAMPLES SExtractor timings, got $(length(times_ms)). " *
            "Raw WSL output: $(repr(output))",
        )

    isnothing(count) &&
        error(
            "Could not parse SExtractor source count. " *
            "Raw WSL output: $(repr(output))",
        )

    return (
        mode = :wsl_nativefs,
        executable = sex.executable,
        count = count,
        min_ms = minimum(times_ms),
        median_ms = median(times_ms),
    )
end

# ============================================================================
# Main
# ============================================================================

function main()
    println()
    println("Astronomy source-extraction comparison")
    println("--------------------------------------")
    println("Julia:             ", VERSION)
    println("Julia threads:     ", Threads.nthreads())
    println("FFTW threads:      ", FFTW_THREADS)
    println("SourceExtraction:  ", Base.pkgversion(SourceExtraction))
    println("image:             ", N, " × ", N, " Float32")
    println("synthetic stars:   ", NSTARS)
    println("noise sigma:       ", NOISE_SIGMA)
    println("threshold level:   ", NSIGMA, "σ")

    rng =
        Xoshiro(
            RNG_SEED,
        )

    image =
        make_star_field(
            rng,
        )

    noise_image =
        make_noise_field(
            rng,
        )

    println()
    println("Benchmarking SourceExtraction.jl...")

    se =
        benchmark_sourceextraction(
            image,
            noise_image,
        )

    println(
        "DoG-domain absolute threshold = ",
        se.threshold,
    )

    mktempdir() do tmpdir
        println("Benchmarking SEP...")

        sep_result =
            try
                benchmark_sep(
                    image,
                    se.filtered,
                    se.threshold,
                    tmpdir,
                )
            catch err
                (
                    available = false,
                    error = sprint(showerror, err),
                )
            end

        if sep_result.available
            println(
                "SEP ",
                sep_result.version,
                " via ",
                sep_result.python,
            )
        end

        println("Benchmarking SExtractor CLI...")

        sex_result =
            try
                benchmark_sextractor(
                    image,
                    tmpdir,
                )
            catch err
                (
                    error = sprint(showerror, err),
                )
            end

        println()
        println("="^104)
        println("RESULTS")
        println("="^104)
        println(
            "Method                                 min [ms] median [ms]      sources       MPix/s       memory",
        )
        println("-"^104)

        print_row(
            "SourceExtraction detect only";
            min_ms = se.detection.min_ms,
            median_ms = se.detection.median_ms,
            count = se.count,
            mpix_s = throughput_mpix_s(
                image,
                se.detection.min_ms,
            ),
            memory = se.detection.bytes,
        )

        print_row(
            "SourceExtraction DoG + detect";
            min_ms = se.pipeline.min_ms,
            median_ms = se.pipeline.median_ms,
            count = se.count,
            mpix_s = throughput_mpix_s(
                image,
                se.pipeline.min_ms,
            ),
            memory = se.pipeline.bytes,
        )

        if sep_result.available
            print_row(
                "SEP minimal on same DoG image";
                min_ms = sep_result.minimal_min_ms,
                median_ms = sep_result.minimal_median_ms,
                count = sep_result.minimal_count,
                mpix_s = throughput_mpix_s(
                    image,
                    sep_result.minimal_min_ms,
                ),
            )

            print_row(
                "SEP default extract";
                min_ms = sep_result.default_min_ms,
                median_ms = sep_result.default_median_ms,
                count = sep_result.default_count,
                mpix_s = throughput_mpix_s(
                    image,
                    sep_result.default_min_ms,
                ),
            )

            print_row(
                "SEP background + extract";
                min_ms = sep_result.background_min_ms,
                median_ms = sep_result.background_median_ms,
                count = sep_result.background_count,
                mpix_s = throughput_mpix_s(
                    image,
                    sep_result.background_min_ms,
                ),
            )
        else
            println(
                "SEP                                    skipped",
            )
            println(
                "  reason: ",
                sep_result.error,
            )
        end

        if isnothing(sex_result)
            println(
                "SExtractor CLI                         skipped (not found natively or in WSL)",
            )
        elseif hasproperty(sex_result, :error)
            println(
                "SExtractor CLI                         skipped",
            )
            println(
                "  reason: ",
                sex_result.error,
            )
        else
            label =
                sex_result.mode === :wsl_nativefs ?
                "SExtractor CLI (WSL native FS)" :
                "SExtractor CLI end-to-end"

            print_row(
                label;
                min_ms = sex_result.min_ms,
                median_ms = sex_result.median_ms,
                count = sex_result.count,
                mpix_s = throughput_mpix_s(
                    image,
                    sex_result.min_ms,
                ),
            )
        end

        println()
        println("INTERPRETATION")
        println("--------------")
        println(
            "• SourceExtraction detect-only and SEP-minimal use the SAME DoG-filtered",
        )
        println(
            "  Float32 image and the SAME absolute numerical threshold.",
        )
        println(
            "• Their source definitions still differ: strict local maxima vs",
        )
        println(
            "  thresholded connected components + SEP measurements.",
        )
        println(
            "• SourceExtraction DoG+detect is an in-memory prepared pipeline.",
        )
        println(
            "• SEP default and SEP background+extract are broader source-extraction",
        )
        println(
            "  pipelines with different filtering/background semantics.",
        )
        println(
            "• For WSL, SExtractor is timed entirely inside one WSL session on",
        )
        println(
            "  WSL-native /tmp storage: repeated wsl.exe startup and /mnt/c I/O",
        )
        println(
            "  are excluded. SExtractor process startup + FITS read remain timed.",
        )
        println(
            "• FITS writing/copying into WSL is outside the timed section.",
        )
        println(
            "• Always compare source counts together with runtime.",
        )
    end
end

main()
