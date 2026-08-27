using SourceExtraction
using Documenter

DocMeta.setdocmeta!(SourceExtraction, :DocTestSetup, :(using SourceExtraction); recursive=true)

makedocs(;
    modules=[SourceExtraction],
    authors="Hossein Zarei Oshtolagh",
    sitename="SourceExtraction.jl",
    doctest=true,
    checkdocs=:exports,
    format=Documenter.HTML(;
        canonical="https://hzarei4.github.io/SourceExtraction.jl/",
        edit_link="main",
        repolink="https://github.com/hzarei4/SourceExtraction.jl",
    ),
    pages=[
        "Home" => "index.md",
        "User guide" => "guide.md",
        "API reference" => "api.md",
    ],
)

deploydocs(;
    repo="github.com/hzarei4/SourceExtraction.jl.git",
    devbranch="main",
)
