if(CMAKE_SIZEOF_VOID_P EQUAL 4)			
    set(BIT_SUFF 32)
else()
    set(BIT_SUFF 64)
endif()

if (APPLE)
    set(PLATFORM_SUFF Darwin)
elseif (UNIX)
    set(PLATFORM_SUFF Linux)
else()
    set(PLATFORM_SUFF Windows)
endif()

set(LIB_FILE NPP_staging_static_${PLATFORM_SUFF}_${BIT_SUFF}_v1)

find_library(NPPST_LIB 
    NAMES "${LIB_FILE}" "lib${LIB_FILE}" 
    PATHS "${CMAKE_SOURCE_DIR}/3rdparty/NPP_staging" 
    DOC "NPP staging library"
    )	

SET(NPPST_INC "${CMAKE_SOURCE_DIR}//3rdparty/NPP_staging")
 