# --- HarfBuzz text-shaping library (bundled) ---
#
# HarfBuzz is always built from the bundled subset in 3rdparty/harfbuzz
# (core + HB_HAS_RASTER). It provides OpenType shaping and rasterization for
# the text-rendering code in the imgproc module (putText / getTextSize).

if(WITH_HARFBUZZ)
  ocv_clear_vars(HARFBUZZ_LIBRARY HARFBUZZ_LIBRARIES HARFBUZZ_INCLUDE_DIR)
  set(HARFBUZZ_LIBRARY libharfbuzz CACHE INTERNAL "")
  set(HARFBUZZ_LIBRARIES ${HARFBUZZ_LIBRARY})

  add_subdirectory("${OpenCV_SOURCE_DIR}/3rdparty/harfbuzz")
  set(HARFBUZZ_INCLUDE_DIR "${${HARFBUZZ_LIBRARY}_SOURCE_DIR}/src" CACHE INTERNAL "")
  set(HAVE_HARFBUZZ 1)
endif()
