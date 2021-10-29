set(WINRT TRUE)

add_definitions(-DWINRT -DNO_GETENV)

# Making definitions available to other configurations and
# to filter dependency restrictions at compile time.
if(WINDOWS_PHONE)
  set(WINRT_PHONE TRUE)
  add_definitions(-DWINRT_PHONE)
elseif(WINDOWS_STORE)
  set(WINRT_STORE TRUE)
  add_definitions(-DWINRT_STORE)
endif()

if(CMAKE_SYSTEM_VERSION MATCHES 10)
  set(WINRT_10 TRUE)
  add_definitions(-DWINRT_10)
  add_definitions(/DWINVER=_WIN32_WINNT_WIN10 /DNTDDI_VERSION=NTDDI_WIN10 /D_WIN32_WINNT=_WIN32_WINNT_WIN10)
elseif(CMAKE_SYSTEM_VERSION MATCHES 8.1)
  set(WINRT_8_1 TRUE)
  add_definitions(-DWINRT_8_1)
  add_definitions(/DWINVER=_WIN32_WINNT_WINBLUE /DNTDDI_VERSION=NTDDI_WINBLUE /D_WIN32_WINNT=_WIN32_WINNT_WINBLUE)
elseif(CMAKE_SYSTEM_VERSION MATCHES 8.0)
  set(WINRT_8_0 TRUE)
  add_definitions(-DWINRT_8_0)
  add_definitions(/DWINVER=_WIN32_WINNT_WIN8 /DNTDDI_VERSION=NTDDI_WIN8 /D_WIN32_WINNT=_WIN32_WINNT_WIN8)
else()
  message(STATUS "Unsupported WINRT version (consider upgrading OpenCV): ${CMAKE_SYSTEM_VERSION}")
endif()

set(OPENCV_DEBUG_POSTFIX "")
