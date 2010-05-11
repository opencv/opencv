add_subdirectory(calib3d)
add_subdirectory(core)
add_subdirectory(features2d)

if(MSVC OR MINGW)
   if(NOT CMAKE_CL_64)
      add_subdirectory(ffmpeg)
   endif()
endif()

add_subdirectory(highgui)
add_subdirectory(imgproc)
add_subdirectory(legacy)
add_subdirectory(contrib)
add_subdirectory(ml)
add_subdirectory(objdetect)

if(PYTHONLIBS_FOUND AND BUILD_NEW_PYTHON_SUPPORT)
    add_subdirectory(python)
endif()

add_subdirectory(video)
add_subdirectory(haartraining)
add_subdirectory(traincascade)
