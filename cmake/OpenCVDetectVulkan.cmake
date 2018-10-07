set(VULKAN_INCLUDE_DIRS "${OpenCV_SOURCE_DIR}/modules/dnn/src/vkcom/" CACHE PATH "Vulkan include directory")
set(VULKAN_LIBRARIES "")

try_compile(VALID_VULKAN
      "${OpenCV_BINARY_DIR}"
      "${OpenCV_SOURCE_DIR}/cmake/checks/vulkan.cpp"
      CMAKE_FLAGS "-DINCLUDE_DIRECTORIES:STRING=${VULKAN_INCLUDE_DIRS}"
      OUTPUT_VARIABLE TRY_OUT
      )
if(NOT ${VALID_VULKAN})
  message(WARNING "Can't use Vulkan")
  return()
endif()

set(HAVE_VULKAN 1)
add_definitions(-DVK_NO_PROTOTYPES)
include_directories(${VULKAN_INCLUDE_DIRS})
