set(root "$ENV{MFX_HOME}")

find_path(MFX_INCLUDE mfxdefs.h PATHS "${root}/include" NO_DEFAULT_PATH)

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

find_library(MFX_LIBRARY mfx PATHS "${root}/lib/${arch}" NO_DEFAULT_PATH)
find_library(MFX_VA_LIBRARY va)
find_library(MFX_VA_DRM_LIBRARY va-drm)

if(MFX_INCLUDE AND MFX_LIBRARY AND MFX_VA_LIBRARY AND MFX_VA_DRM_LIBRARY)
    add_library(mfx-va UNKNOWN IMPORTED)
    set_target_properties(mfx-va PROPERTIES IMPORTED_LOCATION "${MFX_VA_LIBRARY}")

    add_library(mfx-va-drm UNKNOWN IMPORTED)
    set_target_properties(mfx-va-drm PROPERTIES IMPORTED_LOCATION "${MFX_VA_DRM_LIBRARY}")

    add_library(mfx UNKNOWN IMPORTED)
    set_target_properties(mfx PROPERTIES
      IMPORTED_LOCATION "${MFX_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${MFX_INCLUDE}"
      INTERFACE_LINK_LIBRARIES "mfx-va;mfx-va-drm;-Wl,--exclude-libs=libmfx"
    )
    set(HAVE_MFX 1)
else()
    set(HAVE_MFX 0)
endif()
