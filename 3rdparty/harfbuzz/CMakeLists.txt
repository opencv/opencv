# ----------------------------------------------------------------------------
#  CMake file for the bundled HarfBuzz text-shaping library. See root CMakeLists.txt
#
#  OpenCV vendors a minimal subset of HarfBuzz (core + HB_HAS_RASTER) as a
#  normal static library: each .cc is a separate translation unit. The subset
#  is produced by hb_extract.py (see that script to update HarfBuzz).
# ----------------------------------------------------------------------------
project(${HARFBUZZ_LIBRARY} CXX)

ocv_include_directories("${CMAKE_CURRENT_SOURCE_DIR}/src")

# Every .cc copied by hb_extract.py is a real translation unit (the script
# never copies non-TU .cc files), so a plain recursive glob is exactly right.
file(GLOB_RECURSE lib_srcs src/*.cc)
file(GLOB_RECURSE lib_hdrs src/*.h src/*.hh)

# HarfBuzz is built with HB_TINY for minimum footprint. Because we request the
# software rasterizer (HB_HAS_RASTER, defined below), HB_TINY does NOT disable
# the draw/paint/color APIs nor CFF (PostScript/.otf
# outlines) -- see hb-config.hh: HB_NO_DRAW/COLOR/PAINT and the TINY->HB_NO_CFF
# rule only trigger when no HB_HAS_* backend is requested. So CFF (CJK/Indic),
# COLR/CPAL color and our rasterizer all survive HB_TINY.
#
# Two features HB_TINY (via HB_LEAN) would otherwise drop are restored through
# HarfBuzz's config-override hook (hb-opencv-config.hh): thread-safety (HB_NO_MT)
# and variable fonts (HB_NO_VAR). No platform backends (CoreText/DirectWrite/...).

add_library(${HARFBUZZ_LIBRARY} STATIC ${OPENCV_3RDPARTY_EXCLUDE_FROM_ALL} ${lib_srcs} ${lib_hdrs})

target_compile_definitions(${HARFBUZZ_LIBRARY} PRIVATE
    HB_TINY
    HB_HAS_RASTER   # request the software rasterizer: keeps draw/paint/color + CFF under HB_TINY
    "HB_CONFIG_OVERRIDE_H=\"hb-opencv-config.hh\"")

set_target_properties(${HARFBUZZ_LIBRARY} PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
)

ocv_warnings_disable(CMAKE_CXX_FLAGS
    -Wunused-variable -Wunused-function -Wunused-parameter
    -Wunused-but-set-variable    # clang15/gcc
    -Wshadow
    -Wmissing-declarations       # gcc
    -Wmissing-prototypes         # clang
    -Wimplicit-fallthrough
    -Wextra-semi                 # clang
    -Wdeprecated-copy            # gcc/clang
    -Wsuggest-override
    -Wcast-function-type
    -Wclass-memaccess            # gcc
)
ocv_warnings_disable(CMAKE_CXX_FLAGS /wd4244 /wd4267 /wd4127 /wd4456 /wd4459) # MSVC

set_target_properties(${HARFBUZZ_LIBRARY}
  PROPERTIES OUTPUT_NAME ${HARFBUZZ_LIBRARY}
  DEBUG_POSTFIX "${OPENCV_DEBUG_POSTFIX}"
  COMPILE_PDB_NAME ${HARFBUZZ_LIBRARY}
  COMPILE_PDB_NAME_DEBUG "${HARFBUZZ_LIBRARY}${OPENCV_DEBUG_POSTFIX}"
  ARCHIVE_OUTPUT_DIRECTORY ${3P_LIBRARY_OUTPUT_PATH}
  )

if(ENABLE_SOLUTION_FOLDERS)
  set_target_properties(${HARFBUZZ_LIBRARY} PROPERTIES FOLDER "3rdparty")
endif()

if(NOT BUILD_SHARED_LIBS)
  ocv_install_target(${HARFBUZZ_LIBRARY} EXPORT OpenCVModules ARCHIVE DESTINATION ${OPENCV_3P_LIB_INSTALL_PATH} COMPONENT dev OPTIONAL)
endif()
