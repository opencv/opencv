cmake_minimum_required(VERSION 3.1)

if(NOT HAVE_VULKAN)
  set(CMAKE_MODULE_PATH "${OpenCV_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
  find_package(Vulkan) # Try CMake-based config files
  if(Vulkan_FOUND)
    set(VULKAN_INCLUDE_DIRS "${Vulkan_INCLUDE_DIRS}" CACHE PATH "Vulkan include directories" FORCE)
    set(VULKAN_LIBRARIES "${Vulkan_LIBRARIES}" CACHE PATH "Vulkan libraries" FORCE)
    set(HAVE_VULKAN 1)
  endif()
endif()

if(HAVE_VULKAN)
  include_directories(${VULKAN_INCLUDE_DIRS})
  list(APPEND OPENCV_LINKER_LIBS ${VULKAN_LIBRARIES})
else()
  ocv_clear_vars(VULKAN_INCLUDE_DIRS VULKAN_LIBRARIES)
endif()
