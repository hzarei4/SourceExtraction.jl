using SourceExtraction
using Test

@testset "SourceExtraction.jl" begin
    include("detection.jl")
    include("filters.jl")
    include("rois.jl")
    include("cuda.jl")
    include("extractor.jl")
end