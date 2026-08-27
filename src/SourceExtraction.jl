module SourceExtraction


using SeparableFunctions
using FourierTools

export LocalMaximaDetector
export detect_sources!
export extract_rois!

export DoG
export prepare_dog
export dog_kernel!

export prepare_filter
export filter_sources!


export SourceExtractor
export SourceExtractionResult
export extract_sources!
export source_indices
export source_scores
export source_rois
export source_count

include("detection.jl")
include("filters.jl")
include("rois.jl")
include("extractor.jl")

end