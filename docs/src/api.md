```@meta
CurrentModule = SourceExtraction
```

# API reference

## Convenience interface

```@docs
find_sources
find_sources!
```

## Prepared pipeline

```@docs
SourceExtractor
SourceExtractionResult
extract_sources!
source_count
source_indices
source_scores
source_rois
```

## Detection

```@docs
LocalMaximaDetector
DetectionWorkspace
detect_sources!
```

## Difference-of-Gaussians filtering

```@docs
DoG
prepare_dog
dog_kernel!
prepare_filter
filter_sources!
```

## ROI extraction

```@docs
extract_rois!
```

## Index

```@index
```
