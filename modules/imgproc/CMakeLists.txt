set(the_description "Image Processing")
ocv_add_dispatched_file(accum SSE4_1 AVX AVX2)
ocv_add_dispatched_file(bilateral_filter SSE2 AVX2)
ocv_add_dispatched_file(box_filter SSE2 SSE4_1 AVX2)
ocv_add_dispatched_file(filter SSE2 SSE4_1 AVX2)
ocv_add_dispatched_file(color_hsv SSE2 SSE4_1 AVX2)
ocv_add_dispatched_file(color_rgb SSE2 SSE4_1 AVX2)
ocv_add_dispatched_file(color_yuv SSE2 SSE4_1 AVX2)
ocv_add_dispatched_file(median_blur SSE2 SSE4_1 AVX2 AVX512_SKX)
ocv_add_dispatched_file(morph SSE2 SSE4_1 AVX2)
ocv_add_dispatched_file(smooth SSE2 SSE4_1 AVX2)
ocv_add_dispatched_file(sumpixels SSE2 AVX2 AVX512_SKX)
ocv_add_dispatched_file(warp_kernels SSE2 SSE4_1 AVX2 NEON NEON_FP16 RVV LASX)
ocv_define_module(imgproc opencv_core WRAP java objc python js)

ocv_module_include_directories(opencv_imgproc ${ZLIB_INCLUDE_DIRS})

if(HAVE_IPP)
  # OPENCV_IPP_ENABLE_ALL is defined in modules/core/CMakeList.txt
  OCV_OPTION(OPENCV_IPP_GAUSSIAN_BLUR "Enable IPP optimizations for GaussianBlur (+8Mb in binary size)" OPENCV_IPP_ENABLE_ALL)
  if(OPENCV_IPP_GAUSSIAN_BLUR)
    ocv_append_source_file_compile_definitions(${CMAKE_CURRENT_SOURCE_DIR}/src/smooth.dispatch.cpp "ENABLE_IPP_GAUSSIAN_BLUR=1")
  endif()
endif()

set(UNIFONT_MD5 "fb79cf5b4f4c89414f1233f14c2eb273")
set(UNIFONT_NAME "WenQuanYiMicroHei.ttf.gz")
set(UNIFONT_COMMIT "cc7d85179d69a704bee209aa37ce8a657f2f8b34")
set(UNIFONT_URL "https://raw.githubusercontent.com/vpisarev/opencv_3rdparty/${UNIFONT_COMMIT}/")

unset(HAVE_UNIFONT)
unset(HAVE_UNIFONT CACHE)

if (WITH_UNIFONT)
    ocv_download(FILENAME ${UNIFONT_NAME}
                 HASH ${UNIFONT_MD5}
                 URL
                   "${OPENCV_UNIFONT_URL}"
                   "${UNIFONT_URL}"
                 DESTINATION_DIR "${CMAKE_CURRENT_BINARY_DIR}"
                 ID UNIFONT
                 STATUS res
                 RELATIVE_URL)

    if (res)
        message(STATUS "Unicode font has been downloaded successfully.")
        set(HAVE_UNIFONT ON CACHE INTERNAL "")
    else()
        message(STATUS "Unicode font download failed. Turning it off.")
        set(HAVE_UNIFONT OFF CACHE INTERNAL "")
    endif()
else()
    set(HAVE_UNIFONT OFF CACHE INTERNAL "")
endif()

ocv_blob2hdr("${CMAKE_CURRENT_SOURCE_DIR}/fonts/Rubik.ttf.gz" "${CMAKE_CURRENT_BINARY_DIR}/builtin_font_sans.h" OcvBuiltinFontSans)
ocv_blob2hdr("${CMAKE_CURRENT_SOURCE_DIR}/fonts/Rubik-Italic.ttf.gz" "${CMAKE_CURRENT_BINARY_DIR}/builtin_font_italic.h" OcvBuiltinFontItalic)
if (HAVE_UNIFONT)
    ocv_blob2hdr("${CMAKE_CURRENT_BINARY_DIR}/${UNIFONT_NAME}" "${CMAKE_CURRENT_BINARY_DIR}/builtin_font_uni.h" OcvBuiltinFontUni)
endif()
include_directories("${CMAKE_CURRENT_BINARY_DIR}")
ocv_target_link_libraries(${the_module} LINK_PRIVATE ${ZLIB_LIBRARIES})
ocv_install_3rdparty_licenses(fonts fonts/Rubik_OFL.txt)
