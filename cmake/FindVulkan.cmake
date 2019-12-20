# Find Vulkan
#
# Vulkan_INCLUDE_DIRS
# Vulkan_LIBRARIES
# Vulkan_FOUND
if (WIN32)
    find_path(Vulkan_INCLUDE_DIRS NAMES vulkan/vulkan.h HINTS
        "$ENV{VULKAN_SDK}/Include"
        "$ENV{VK_SDK_PATH}/Include")
    if (CMAKE_CL_64)
        find_library(Vulkan_LIBRARIES NAMES vulkan-1 HINTS
            "$ENV{VULKAN_SDK}/Bin"
            "$ENV{VK_SDK_PATH}/Bin")
    else()
        find_library(Vulkan_LIBRARIES NAMES vulkan-1 HINTS
            "$ENV{VULKAN_SDK}/Bin32"
            "$ENV{VK_SDK_PATH}/Bin32")
    endif()
else()
    find_path(Vulkan_INCLUDE_DIRS NAMES vulkan/vulkan.h HINTS
        "$ENV{VULKAN_SDK}/include")
    find_library(Vulkan_LIBRARIES NAMES vulkan HINTS
        "$ENV{VULKAN_SDK}/lib")
endif()
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Vulkan DEFAULT_MSG Vulkan_LIBRARIES Vulkan_INCLUDE_DIRS)
mark_as_advanced(Vulkan_INCLUDE_DIRS Vulkan_LIBRARIES)
