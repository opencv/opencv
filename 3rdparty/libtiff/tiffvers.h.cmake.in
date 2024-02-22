/* tiffvers.h version information is updated according to version information
 * in configure.ac */

/* clang-format off */

/* clang-format disabled because FindTIFF.cmake is very sensitive to the
 * formatting of below line being a single line.
 * Furthermore, configure_file variables of type "@VAR@" are
 * modified by clang-format and won't be substituted by CMake.
 */
#define TIFFLIB_VERSION_STR "LIBTIFF, Version @LIBTIFF_VERSION@\nCopyright (c) 1988-1996 Sam Leffler\nCopyright (c) 1991-1996 Silicon Graphics, Inc."
/*
 * This define can be used in code that requires
 * compilation-related definitions specific to a
 * version or versions of the library.  Runtime
 * version checking should be done based on the
 * string returned by TIFFGetVersion.
 */
#define TIFFLIB_VERSION @LIBTIFF_RELEASE_DATE@

/* The following defines have been added in 4.5.0 */
#define TIFFLIB_MAJOR_VERSION @LIBTIFF_MAJOR_VERSION@
#define TIFFLIB_MINOR_VERSION @LIBTIFF_MINOR_VERSION@
#define TIFFLIB_MICRO_VERSION @LIBTIFF_MICRO_VERSION@
#define TIFFLIB_VERSION_STR_MAJ_MIN_MIC "@LIBTIFF_VERSION@"

/* Macro added in 4.5.0. Returns TRUE if the current libtiff version is
 * greater or equal to major.minor.micro
 */
#define TIFFLIB_AT_LEAST(major, minor, micro) \
    (TIFFLIB_MAJOR_VERSION > (major) || \
     (TIFFLIB_MAJOR_VERSION == (major) && TIFFLIB_MINOR_VERSION > (minor)) || \
     (TIFFLIB_MAJOR_VERSION == (major) && TIFFLIB_MINOR_VERSION == (minor) && \
      TIFFLIB_MICRO_VERSION >= (micro)))

/* clang-format on */
