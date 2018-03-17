set(HAVE_MFX 0)

if (UNIX)
    set(root "$ENV{MFX_HOME}")
elseif(WIN32)
    set(root "$ENV{INTELMEDIASDKROOT}")
endif()

# TODO: ICC? MINGW? ARM? IOS?
if(WIN32)
    if(X86_64)
        set(arch "x64")
    else()
        set(arch "win32")
    endif()
elseif(UNIX)
    set(arch "lin_x64")
else()
    # ???
endif()

find_path(MFX_INCLUDE mfxdefs.h PATHS "${root}/include" NO_DEFAULT_PATH)
message(STATUS "MFX_INCLUDE: ${MFX_INCLUDE} (${root}/include)")
find_library(MFX_LIBRARY NAMES mfx PATHS "${root}/lib/${arch}" NO_DEFAULT_PATH)
if(MSVC)
    if(MSVC14)
        find_library(MFX_LIBRARY NAMES libmfx_vs2015.lib PATHS "${root}/lib/${arch}" NO_DEFAULT_PATH)
    else()
        find_library(MFX_LIBRARY NAMES libmfx.lib PATHS "${root}/lib/${arch}" NO_DEFAULT_PATH)
    endif()
endif()

if(NOT MFX_INCLUDE OR NOT MFX_LIBRARY)
    return()
endif()

set(deps)

if (UNIX)
    find_library(MFX_VA_LIBRARY va)
    find_library(MFX_VA_DRM_LIBRARY va-drm)
    if (NOT MFX_VA_LIBRARY OR NOT MFX_VA_DRM_LIBRARY)
        return()
    endif()
    add_library(mfx-va UNKNOWN IMPORTED)
    set_target_properties(mfx-va PROPERTIES IMPORTED_LOCATION "${MFX_VA_LIBRARY}")
    add_library(mfx-va-drm UNKNOWN IMPORTED)
    set_target_properties(mfx-va-drm PROPERTIES IMPORTED_LOCATION "${MFX_VA_DRM_LIBRARY}")
    list(APPEND deps mfx-va mfx-va-drm "-Wl,--exclude-libs=libmfx")
endif()

add_library(mfx UNKNOWN IMPORTED)
set_target_properties(mfx PROPERTIES
  IMPORTED_LOCATION "${MFX_LIBRARY}"
  INTERFACE_INCLUDE_DIRECTORIES "${MFX_INCLUDE}"
  INTERFACE_LINK_LIBRARIES "${deps}"
)

set(HAVE_MFX 1)
