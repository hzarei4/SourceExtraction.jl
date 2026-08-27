using SourceExtraction
using Documenter

DocMeta.setdocmeta!(SourceExtraction, :DocTestSetup, :(using SourceExtraction); recursive=true)

makedocs(;
    modules=[SourceExtraction],
    authors="Hossein Zarei Oshtolagh",
    sitename="SourceExtraction.jl",
    format=Documenter.HTML(;
        canonical="https://hzarei4.github.io/SourceExtraction.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/hzarei4/SourceExtraction.jl",
    devbranch="main",
)
