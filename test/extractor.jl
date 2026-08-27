@testset "SourceExtractor" begin

    @testset "2-D detection without ROIs" begin

        A =
            zeros(
                Float32,
                32,
                32,
            )

        A[8, 9] = 10f0
        A[21, 23] = 20f0

        dog =
            DoG(
                1f0,
                3f0,
            )

        detector =
            LocalMaximaDetector(
                0.1f0;
                radius = 1,
            )

        extractor =
            SourceExtractor(
                A;
                dog = dog,
                detector = detector,
                max_sources = 10,
            )

        result =
            extract_sources!(
                extractor,
                A,
            )

        @test source_count(result) == 2
        @test length(result) == 2
        @test !isempty(result)

        @test Set(source_indices(result)) ==
              Set(
                  [
                      CartesianIndex(8, 9),
                      CartesianIndex(21, 23),
                  ],
              )

        @test length(source_scores(result)) == 2

        @test source_rois(result) === nothing
    end


    @testset "2-D detection with ROIs" begin

        A =
            zeros(
                Float32,
                32,
                32,
            )

        A[10, 12] = 10f0
        A[22, 20] = 20f0

        dog =
            DoG(
                1f0,
                3f0,
            )

        detector =
            LocalMaximaDetector(
                0.1f0;
                radius = 1,
            )

        extractor =
            SourceExtractor(
                A;
                dog = dog,
                detector = detector,
                roi_size = (7, 7),
                max_sources = 10,
            )

        @test extractor.border == 3

        result =
            extract_sources!(
                extractor,
                A,
            )

        @test source_count(result) == 2

        indices =
            source_indices(result)

        rois =
            source_rois(result)

        @test size(rois) ==
              (
                  7,
                  7,
                  2,
              )

        for k in eachindex(indices)

            I =
                indices[k]

            @test rois[4, 4, k] ==
                  A[I]
        end
    end


    @testset "Batched source extraction" begin

        A =
            zeros(
                Float32,
                32,
                32,
                3,
            )

        A[8, 9, 1] = 10f0
        A[15, 20, 2] = 20f0
        A[25, 12, 3] = 30f0

        dog =
            DoG(
                1f0,
                3f0,
            )

        detector =
            LocalMaximaDetector(
                0.1f0;
                radius = 1,
            )

        extractor =
            SourceExtractor(
                A;
                dog = dog,
                detector = detector,
                roi_size = (5, 5),
                max_sources = 10,
            )

        result =
            extract_sources!(
                extractor,
                A,
            )

        @test source_count(result) == 3

        @test Set(source_indices(result)) ==
              Set(
                  [
                      CartesianIndex(8, 9, 1),
                      CartesianIndex(15, 20, 2),
                      CartesianIndex(25, 12, 3),
                  ],
              )

        indices =
            source_indices(result)

        rois =
            source_rois(result)

        for k in eachindex(indices)

            I =
                indices[k]

            @test rois[3, 3, k] ==
                  A[I]
        end
    end


    @testset "Input size validation" begin

        A =
            zeros(
                Float32,
                32,
                32,
            )

        extractor =
            SourceExtractor(
                A;
                dog = DoG(1f0, 3f0),
                detector =
                    LocalMaximaDetector(
                        0.1f0;
                        radius = 1,
                    ),
            )

        B =
            zeros(
                Float32,
                64,
                64,
            )

        @test_throws DimensionMismatch extract_sources!(
            extractor,
            B,
        )
    end


    @testset "ROI validation" begin

        A =
            zeros(
                Float32,
                32,
                32,
            )

        dog =
            DoG(
                1f0,
                3f0,
            )

        detector =
            LocalMaximaDetector(
                0.1f0;
                radius = 1,
            )

        @test_throws ArgumentError SourceExtractor(
            A;
            dog = dog,
            detector = detector,
            roi_size = (6, 7),
        )

        @test_throws ArgumentError SourceExtractor(
            A;
            dog = dog,
            detector = detector,
            roi_size = (7, 6),
        )

        @test_throws ArgumentError SourceExtractor(
            A;
            dog = dog,
            detector = detector,
            roi_size = (7, 7),
            border = 2,
        )
    end


    @testset "Reusable buffers" begin

        A =
            zeros(
                Float32,
                32,
                32,
            )

        A[10, 10] = 10f0

        extractor =
            SourceExtractor(
                A;
                dog = DoG(1f0, 3f0),
                detector =
                    LocalMaximaDetector(
                        0.1f0;
                        radius = 1,
                    ),
                roi_size = (5, 5),
                max_sources = 10,
            )

        result1 =
            extract_sources!(
                extractor,
                A,
            )

        indices_storage =
            result1.indices

        scores_storage =
            result1.scores

        rois_storage =
            result1.rois

        result2 =
            extract_sources!(
                extractor,
                A,
            )

        @test result2.indices === indices_storage
        @test result2.scores === scores_storage
        @test result2.rois === rois_storage
    end

end