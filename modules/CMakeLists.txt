if(ANDROID AND WITH_ANDROID_CAMERA)
    add_subdirectory(androidcamera)
endif()

add_subdirectory(calib3d)
add_subdirectory(core)
add_subdirectory(features2d)
add_subdirectory(flann)

if(BUILD_TESTS OR BUILD_PERF_TESTS)
  add_subdirectory(ts)
endif()

add_subdirectory(highgui)
add_subdirectory(imgproc)
add_subdirectory(legacy)
add_subdirectory(contrib)
add_subdirectory(ml)
add_subdirectory(objdetect)

if(PYTHONLIBS_FOUND AND BUILD_NEW_PYTHON_SUPPORT AND PYTHON_USE_NUMPY)
  add_subdirectory(python)
endif()

if(BUILD_JAVA_SUPPORT)
  add_subdirectory(java)
endif()

add_subdirectory(video)

if(NOT (ANDROID OR IOS))
  add_subdirectory(gpu)
endif()

if(NOT IOS)
  add_subdirectory(traincascade)
  add_subdirectory(haartraining)
  add_subdirectory(stitching)
endif()
