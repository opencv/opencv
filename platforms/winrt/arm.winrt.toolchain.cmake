set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_SYSTEM_PROCESSOR "arm-v7a")

set(CMAKE_FIND_ROOT_PATH "${CMAKE_SOURCE_DIR}/platforms/winrt")
set(CMAKE_REQUIRED_DEFINITIONS -D_ARM_WINAPI_PARTITION_DESKTOP_SDK_AVAILABLE)
add_definitions(-D_ARM_WINAPI_PARTITION_DESKTOP_SDK_AVAILABLE)

set(CMAKE_CXX_FLAGS           ""                    CACHE STRING "c++ flags")
set(CMAKE_C_FLAGS             ""                    CACHE STRING "c flags")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ZW -EHsc -GS")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -GS")


set(CMAKE_SHARED_LINKER_FLAGS "/r:System.Runtime.WindowsRuntime.dll /r:System.Threading.Tasks.dll"                    CACHE STRING "shared linker flags")
set(CMAKE_MODULE_LINKER_FLAGS "/r:System.Runtime.WindowsRuntime.dll /r:System.Threading.Tasks.dll"                    CACHE STRING "module linker flags")
set(CMAKE_EXE_LINKER_FLAGS    "/r:System.Runtime.WindowsRuntime.dll /r:System.Threading.Tasks.dll"  CACHE STRING "executable linker flags")