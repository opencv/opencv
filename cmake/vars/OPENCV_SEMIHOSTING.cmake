set(CV_TRACE OFF)

# These third parties libraries are incompatible with the semihosting
# toolchain.
set(WITH_JPEG OFF)
set(WITH_OPENEXR OFF)
set(WITH_TIFF OFF)

# Turn off `libpng` for some linking issues.
set(WITH_PNG OFF)
