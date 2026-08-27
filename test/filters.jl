@testset "DoG filter" begin

    @testset "Construction" begin
        dog = DoG(1f0, 3f0)

        @test dog.sigma1 === 1f0
        @test dog.sigma2 === 3f0
    end


    @testset "Invalid parameters" begin
        @test_throws ArgumentError DoG(0f0, 3f0)
        @test_throws ArgumentError DoG(-1f0, 3f0)
        @test_throws ArgumentError DoG(3f0, 1f0)
        @test_throws ArgumentError DoG(1f0, 1f0)
    end


    @testset "Kernel generation" begin
        dog = DoG(1f0, 3f0)

        kernel = similar(
            zeros(Float32, 65, 65),
        )

        dog_kernel!(
            kernel,
            dog,
        )

        @test size(kernel) == (65, 65)
        @test eltype(kernel) == Float32

        center = CartesianIndex(
            size(kernel) .÷ 2 .+ 1,
        )

        @test kernel[center] > 0

        # A properly normalized DoG should have approximately zero DC.
        @test isapprox(
            sum(kernel),
            0f0;
            atol = 1f-5,
        )
    end
end

@testset "Prepared DoG" begin

    dog = DoG(1f0, 3f0)

    kernel1 = zeros(Float32, 65, 65)
    kernel2 = similar(kernel1)

    prepared = prepare_dog(
        kernel1,
        dog,
    )

    dog_kernel!(
        kernel1,
        dog,
    )

    dog_kernel!(
        kernel2,
        prepared,
    )

    @test kernel1 ≈ kernel2

    # Compile
    dog_kernel!(
        kernel2,
        prepared,
    )

    bytes = @allocated dog_kernel!(
        kernel2,
        prepared,
    )

    @test bytes == 0
end



@testset "DoG impulse response" begin
    sz = (65, 65)

    image = zeros(Float32, sz)
    image[33, 33] = 1f0

    dog = DoG(1f0, 3f0)

    kernel = zeros(Float32, sz)
    prepared_dog = prepare_dog(kernel, dog)
    dog_kernel!(kernel, prepared_dog)

    filtered = similar(image)

    prepared_filter = prepare_filter(
        image,
        dog,
    )

    filter_sources!(
        filtered,
        image,
        prepared_filter,
    )

    @test filtered ≈ kernel atol=1f-5
end


@testset "DoG removes constant background" begin
    image = fill(10f0, 128, 128)
    filtered = similar(image)

    dog = DoG(1f0, 3f0)
    prepared = prepare_filter(image, dog)

    filter_sources!(
        filtered,
        image,
        prepared,
    )

    @test maximum(abs, filtered) < 1f-4
end



@testset "Prepared DoG filtering" begin

    @testset "Impulse response" begin

        sz = (65, 65)

        image = zeros(Float32, sz)
        image[33, 33] = 1f0

        dog = DoG(1f0, 3f0)

        kernel = zeros(Float32, sz)

        prepared_dog = prepare_dog(
            kernel,
            dog,
        )

        dog_kernel!(
            kernel,
            prepared_dog,
        )

        prepared_filter = prepare_filter(
            image,
            dog,
        )

        filtered = filter_sources!(
            prepared_filter,
            image,
        )

        @test filtered ≈ kernel atol=1f-5
    end


    @testset "Constant background suppression" begin

        image = fill(
            10f0,
            128,
            128,
        )

        dog = DoG(
            1f0,
            3f0,
        )

        prepared_filter = prepare_filter(
            image,
            dog,
        )

        filtered = filter_sources!(
            prepared_filter,
            image,
        )

        @test maximum(abs, filtered) < 1f-4
    end


    @testset "Destination method" begin

        image = rand(
            Float32,
            128,
            128,
        )

        dog = DoG(
            1f0,
            3f0,
        )

        prepared_filter = prepare_filter(
            image,
            dog,
        )

        reference = copy(
            filter_sources!(
                prepared_filter,
                image,
            ),
        )

        dst = similar(image)

        filter_sources!(
            dst,
            image,
            prepared_filter,
        )

        @test dst ≈ reference
    end


    @testset "Allocation-free filtering" begin

        image = rand(
            Float32,
            128,
            128,
        )

        dog = DoG(
            1f0,
            3f0,
        )

        prepared_filter = prepare_filter(
            image,
            dog,
        )

        # Warm-up.
        filter_sources!(
            prepared_filter,
            image,
        )

        bytes = @allocated filter_sources!(
            prepared_filter,
            image,
        )

        @test bytes == 0
    end

end