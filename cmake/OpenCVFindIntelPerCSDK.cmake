# Main variables:
# INTELPERC_LIBRARIES and INTELPERC_INCLUDE to link Intel Perceptial Computing SDK modules
# HAVE_INTELPERC for conditional compilation OpenCV with/without Intel Perceptial Computing SDK

if(X86_64)
    find_path(INTELPERC_INCLUDE_DIR "pxcsession.h" PATHS "$ENV{PCSDK_DIR}include" DOC "Path to Intel Perceptual Computing SDK interface headers")
    find_file(INTELPERC_LIBRARIES "libpxc.lib" PATHS "$ENV{PCSDK_DIR}lib/x64" DOC "Path to Intel Perceptual Computing SDK interface libraries")
else()
    find_path(INTELPERC_INCLUDE_DIR "pxcsession.h" PATHS "$ENV{PCSDK_DIR}include" DOC "Path to Intel Perceptual Computing SDK interface headers")
    find_file(INTELPERC_LIBRARIES "libpxc.lib" PATHS "$ENV{PCSDK_DIR}lib/Win32" DOC "Path to Intel Perceptual Computing SDK interface libraries")
endif()

if(INTELPERC_INCLUDE_DIR AND INTELPERC_LIBRARIES)
    set(HAVE_INTELPERC TRUE)
else()
    set(HAVE_INTELPERC FALSE)
    message(WARNING "Intel Perceptual Computing SDK library directory (set by INTELPERC_LIB_DIR variable) is not found or does not have Intel Perceptual Computing SDK libraries.")
endif() #if(INTELPERC_INCLUDE_DIR AND INTELPERC_LIBRARIES)

mark_as_advanced(FORCE INTELPERC_LIBRARIES INTELPERC_INCLUDE_DIR)