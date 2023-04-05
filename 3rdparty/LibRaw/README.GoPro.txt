GoPro HERO/Fusion .GPR files are DNG files by nature, but with own wavelet-based codec.

GoPro provides GPR SDK for reading these files, available at https://github.com/gopro/gpr

LibRaw is able to use this GPR SDK to read GoPro HERO/Fusion files.

To enable this support:

1. Build GPR SDK (see NOTES section below)
2. Build LibRaw with -DUSE_DNGSDK and -DUSE_GPRSDK compiler flags
   (you'll need to adjust INCLUDE path and linker flags to add GPR SDK files to compile/link).


NOTES:
I. GPR SDK comes with (patched) Adobe DNG SDK source (v1.4 but outdated).
   This DNG SDK is *NOT* compatible with LibRaw since 0.20 due to
   internals change .

II. So, you need to patch latest Adobe DNG SDK v1.4 (dated 2015), this version
   is available from Adobe:
   http://download.adobe.com/pub/adobe/dng/dng_sdk_1_4.zip
   or use Adobe DNG SDK v1.6   
  
  (most likely, this apply for v1.5 too, but not tested/checked):

   a) For Adobe DNG SDK v1.4 you'll need to enable dng_ifd.fCompression value == 9 in 
      dng_ifd::IsValidCFA() call
      Use provided patch: LibRaw/GoPro/dng-sdk-1_4-allow-VC5-validate.diff 
      (it may not apply to any Adobe DNG SDK version, if so apply it by hands).

      This compression type is already handled (passed via validation)
      in Adobe DNG SDK v1.6

   b) Adobe DNG SDK v1.6 defines the ccVc5 constant in dng_tag_values.h
      so GPR SDK's gpr_read_image.cpp will not compile due to constant redefinition
      so use provided patch:   LibRaw/GoPro/dng-sdk-1_6-hide-ccVc5-definitiion.diff
      to use Adobe's definitiion     

   c) Newer (than supplied w/ GPR SDK) Adobe SDK versions changes 
     dng_read_image::ReadTile interface, please apply patches 
     LibRaw/GoPro/gpr_read_image.cpp.diff 
     and  LibRaw/GoPro/gpr_read_image.h.diff to your GPR SDK code

   d) GPR SDK's gpr_sdk/private/gpr.cpp uses own (added) dng_host method 
      GetGPMFPayload so it will not compile with Adobes (not patched) 
      dng_host.h
      LibRaw does not use high-level interface provided by gpr.cpp, so
      possible problem solutions are:
       - either compile GPR SDK without gpr_sdk/private/gpr.cpp file
       - or provide GPR's dng_host.h while building GPR SDK.
     (in our software we use 1st method).       

   See Note VII below for detailed GPR SDK build instructions w/ Cmake

III. LibRaw uses private gpr_read_image() interface
    So you'll need to add PATH_TO/gpr_sdk/gpr_sdk/private to -I compiler flags.

IV.  -DUSE_GPRSDK LibRaw build flag requires -DUSE_DNGSDK. LibRaw will not 
     compile if USE_GPRSDK is set, but USE_DNGSDK is not

V.  LibRaw will use DNG SDK to unpack GoPro files even if imgdata.params.use_dng_sdk is set to 0.

VI. If LibRaw is built with -DUSE_GPRSDK, LibRaw::capabilities will return LIBRAW_CAPS_GPRSDK flag.

VII. GPR SDK build using cmake (contributed by our user, great thanks)

1. replace gopro's toplevel CMakeLists.txt with this one (this builds only a subset of the libraries):
------------------------------------
# minimum required cmake version
cmake_minimum_required( VERSION 3.5 FATAL_ERROR )

set(CMAKE_SUPPRESS_REGENERATION true)
set(CMAKE_C_FLAGS "-std=c99")

# project name
project( gpr )

option(DNGINCLUDEDIR "Adobe DNG toolkit include directory")
INCLUDE_DIRECTORIES( ${DNGINCLUDEDIR} )

# DNG toolkit requires C++11 minimum:
set_property(GLOBAL PROPERTY CXX_STANDARD 17)

# add needed subdirectories
add_subdirectory( "source/lib/common" )
add_subdirectory( "source/lib/vc5_common" )
add_subdirectory( "source/lib/vc5_decoder" )
add_subdirectory( "source/lib/gpr_sdk" )

set_property(TARGET gpr_sdk PROPERTY CXX_STANDARD 17)

IF (WIN32)
TARGET_COMPILE_DEFINITIONS( gpr_sdk PUBLIC -DqWinOS=1 -DqMacOS=0 -DqLinux=0)
ELSEIF (APPLE)
TARGET_COMPILE_DEFINITIONS( gpr_sdk PUBLIC -DqWinOS=0 -DqMacOS=1 -DqLinux=0)
ELSE()
TARGET_COMPILE_DEFINITIONS( gpr_sdk PUBLIC -DqWinOS=0 -DqMacOS=0 -DqLinux=1)
ENDIF()
----------------------------------------

2. apply the two patches of README.GoPro.txt section II b.
the patch of section IIa is not needed with libdng1.5.

3. delete these two files:
/source/lib/gpr_sdk/private/gpr.cpp
/source/lib/gpr_sdk/private/gpr_image_writer.cpp

4. run CMAKE with -DDNGINCLUDEDIR, pointing to the headers from Adobe dng 1.5.

5. build. You get 4 libraries "gpr_sdk", "vc5_common", "vc5_decoder", "common", the rest is ignored.


