if(OPENCV_PYTHON_SKIP_DETECTION)
  return()
endif()

# Prevent multiple lookups if already completed in a previous configuration run
if(NOT OpenCV_Python3_FOUND)

  # Setup version constraints based on input options
  option(OPENCV_PYTHON3_VERSION "Python3 version" "")

  if(OPENCV_PYTHON3_VERSION)
    set(Python3_EXACT_VER "EXACT")
    set(Python3_REQ_VER "${OPENCV_PYTHON3_VERSION}")
  elseif(MIN_VER_PYTHON3)
    set(Python3_EXACT_VER "")
    set(Python3_REQ_VER "${MIN_VER_PYTHON3}")
  else()
    set(Python3_EXACT_VER "")
    set(Python3_REQ_VER "")
  endif()
  if(NOT DEFINED Python3_EXECUTABLE)
    find_program(Python3_PATH "python3")
    if(NOT Python3_PATH)
      find_program(Python3_PATH "python")
    endif()
    if(Python3_PATH)
      set(Python3_EXECUTABLE "${Python3_PATH}")
    endif()
  endif()
  # Call the unified modern package locator.
  # This guarantees the Interpreter, Libraries, and NumPy match perfectly.
  find_package(Python3 ${Python3_REQ_VER} ${Python3_EXACT_VER}
               COMPONENTS Development NumPy)
  set(_version_major_minor "${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}")
  if(NOT ANDROID AND NOT IOS AND NOT XROS)
    if(CMAKE_HOST_UNIX)
      if("${Python3_SITEARCH}" MATCHES "site-packages")
        set(Python3_PACKAGES_PATH "python${_version_major_minor}/site-packages")
      else() #debian based assumed, install to the dist-packages.
        set(Python3_PACKAGES_PATH "python${_version_major_minor}/dist-packages")
      endif()
      set(Python3_PACKAGES_PATH "lib/${Python3_PACKAGES_PATH}")
    elseif(CMAKE_HOST_WIN32)
      get_filename_component(_path "${Python3_EXECUTABLE}" PATH)
      file(TO_CMAKE_PATH "${_path}" _path)
      if(NOT EXISTS "${_path}/Lib/site-packages")
        unset(_path)
        get_filename_component(_path "[HKEY_LOCAL_MACHINE\\SOFTWARE\\Python\\PythonCore\\${_version_major_minor}\\InstallPath]" ABSOLUTE)
        if(NOT _path)
           get_filename_component(_path "[HKEY_CURRENT_USER\\SOFTWARE\\Python\\PythonCore\\${_version_major_minor}\\InstallPath]" ABSOLUTE)
        endif()
        file(TO_CMAKE_PATH "${_path}" _path)
      endif()
      set(Python3_PACKAGES_PATH "${_path}/Lib/site-packages")
      unset(_path)
    endif()
  endif()


endif()

# Handle Limited API logic using the found NumPy version
if(Python3_NumPy_VERSION)
  OCV_OPTION(Python3_LIMITED_API "Build with Python Limited API (not available with numpy >=1.15 <1.17)" NO
             VISIBLE_IF Python3_NumPy_VERSION VERSION_LESS "1.15" OR NOT Python3_NumPy_VERSION VERSION_LESS "1.17")

  if(Python3_LIMITED_API)
    set(_default_ver "0x03060000")
    if(Python3_VERSION VERSION_LESS "3.6")
      set(_default_ver "0x030${Python3_VERSION_MINOR}0000")
    endif()
    set(Python3_LIMITED_API_VERSION ${_default_ver} CACHE STRING "Minimal Python version for Limited API")
  endif()
endif()

# Set global fallbacks for other build modules that request a default python interpreter
if(Python3_Interpreter_FOUND)
    set(PYTHON_DEFAULT_AVAILABLE "TRUE")
    set(PYTHON_DEFAULT_EXECUTABLE "${Python3_EXECUTABLE}")
    set(PYTHON_DEFAULT_VERSION "${Python3_VERSION}")
endif()
