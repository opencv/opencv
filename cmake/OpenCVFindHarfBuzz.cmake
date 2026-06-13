# HarfBuzz text shaping + rasterization for cv::putText / cv::getTextSize.
#
# Behaviour (mirrors BUILD_PNG / BUILD_ZLIB etc.):
#   WITH_HARFBUZZ=ON (default)
#       Try to use a system HarfBuzz; if it is missing or too old to provide the
#       hb-raster software-rasterizer API we depend on, build the bundled subset
#       in 3rdparty/harfbuzz.
#   WITH_HARFBUZZ=ON  BUILD_HARFBUZZ=ON
#       Skip the system search and always build the bundled subset.
#   WITH_HARFBUZZ=OFF
#       Disabled entirely; BUILD_HARFBUZZ has no effect (as with BUILD_PNG etc.).

if(WITH_HARFBUZZ)
  include(CheckCXXSourceCompiles)

  ocv_clear_vars(HAVE_HARFBUZZ HARFBUZZ_IS_BUNDLED)

  if(NOT BUILD_HARFBUZZ)
    # --- look for a system HarfBuzz via pkg-config ---
    ocv_clear_internal_cache_vars(HARFBUZZ_LIBRARY HARFBUZZ_LIBRARIES HARFBUZZ_INCLUDE_DIR)
    ocv_check_modules(HARFBUZZ harfbuzz)

    if(HAVE_HARFBUZZ)
      # The hb-raster software rasterizer is a recent addition and may be
      # shipped as a separate library: e.g. Homebrew/Linux split it into
      # libharfbuzz-raster (with its own harfbuzz-raster.pc) while the plain
      # harfbuzz.pc links only -lharfbuzz (no hb_raster_* symbols). Pull in the
      # raster module when present; otherwise assume a monolithic build.
      ocv_check_modules(HARFBUZZ_RASTER harfbuzz-raster)
      if(HAVE_HARFBUZZ_RASTER)
        # harfbuzz-raster.pc "Requires: harfbuzz", so its resolved flags already
        # include the core library and include dir.
        set(_hb_inc  "${HARFBUZZ_RASTER_INCLUDE_DIRS}")
        set(_hb_libs "${HARFBUZZ_RASTER_LIBRARIES}")
      else()
        set(_hb_inc  "${HARFBUZZ_INCLUDE_DIRS}")
        set(_hb_libs "${HARFBUZZ_LIBRARIES}")
      endif()

      # Verify the header and symbol actually compile and link with the chosen
      # libraries; otherwise fall back to the bundled copy.
      ocv_clear_vars(HAVE_HARFBUZZ)
      set(CMAKE_REQUIRED_INCLUDES "${_hb_inc}")
      set(CMAKE_REQUIRED_LIBRARIES "${_hb_libs}")
      check_cxx_source_compiles("
        #include <hb.h>
        #include <hb-raster.h>
        int main() {
          hb_raster_draw_t* rd = hb_raster_draw_create_or_fail();
          hb_raster_draw_render(rd);
          return rd ? 0 : 1;
        }" HARFBUZZ_HAS_RASTER)
      unset(CMAKE_REQUIRED_INCLUDES)
      unset(CMAKE_REQUIRED_LIBRARIES)

      if(HARFBUZZ_HAS_RASTER)
        set(HARFBUZZ_INCLUDE_DIR "${_hb_inc}" CACHE INTERNAL "")
        set(HARFBUZZ_LIBRARIES "${_hb_libs}" CACHE INTERNAL "")
        set(HAVE_HARFBUZZ 1)
      else()
        message(STATUS "HarfBuzz: found system version ${HARFBUZZ_VERSION} but it lacks the hb-raster API; building the bundled copy instead")
      endif()
    endif()
  endif()

  if(NOT HAVE_HARFBUZZ)
    # --- build the bundled HarfBuzz subset (HB_TINY + hb-raster) ---
    ocv_clear_vars(HARFBUZZ_LIBRARY HARFBUZZ_LIBRARIES HARFBUZZ_INCLUDE_DIR)
    set(HARFBUZZ_LIBRARY libharfbuzz CACHE INTERNAL "")
    set(HARFBUZZ_LIBRARIES ${HARFBUZZ_LIBRARY})
    add_subdirectory("${OpenCV_SOURCE_DIR}/3rdparty/harfbuzz")
    set(HARFBUZZ_INCLUDE_DIR "${${HARFBUZZ_LIBRARY}_SOURCE_DIR}/src" CACHE INTERNAL "")
    set(HARFBUZZ_VERSION "build (14.2.1)" CACHE INTERNAL "")
    set(HARFBUZZ_IS_BUNDLED YES)
    set(HAVE_HARFBUZZ 1)
  endif()
endif()
