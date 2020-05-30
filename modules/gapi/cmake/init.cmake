OCV_OPTION(WITH_ADE "Enable ADE framework (required for Graph API module)" ON)

OCV_OPTION(WITH_FREETYPE "Enable FreeType framework" OFF)
OCV_OPTION(WITH_PLAIDML  "Include PlaidML2 support"  OFF)

if(NOT WITH_ADE)
  return()
endif()

if(ade_DIR)
  # if ade_DIR is set, use ADE-supplied CMake script
  # to set up variables to the prebuilt ADE
  find_package(ade 0.1.0)
endif()

if(NOT TARGET ade)
  # if ade_DIR is not set, try to use automatically
  # downloaded one (if there any)
  include("${CMAKE_CURRENT_LIST_DIR}/DownloadADE.cmake")
endif()

if(WITH_FREETYPE)
  ocv_check_modules(FREETYPE freetype2)
  if (FREETYPE_FOUND)
      set(HAVE_FREETYPE TRUE)
  endif()
endif()

if(WITH_PLAIDML)
  find_package(PlaidML2 CONFIG QUIET)
  if (PLAIDML_FOUND)
      set(HAVE_PLAIDML TRUE)
  endif()
endif()
