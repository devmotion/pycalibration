using Pkg: Pkg

Pkg.add([
    Pkg.PackageSpec(;
        name="CalibrationErrors",
        uuid="33913031-fe46-5864-950f-100836f47845",
        version="0.6",
    ),
    Pkg.PackageSpec(;
        name="CalibrationTests", uuid="2818745e-0823-50c7-bc2d-405ac343d48b", version="0.6"
    ),
])
