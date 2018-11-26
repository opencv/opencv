OCV_OPTION(WITH_ADE "Enable ADE framework (required for Graph API module)" ON)

if(NOT WITH_ADE)
  return()
endif()

if (ade_DIR)
  # if ade_DIR is set, use ADE-supplied CMake script
  # to set up variables to the prebuilt ADE
  find_package(ade 0.1.0)
endif()

if(NOT TARGET ade)
  # if ade_DIR is not set, try to use automatically
  # downloaded one (if there any)
  include("${CMAKE_CURRENT_LIST_DIR}/DownloadADE.cmake")
endif()
