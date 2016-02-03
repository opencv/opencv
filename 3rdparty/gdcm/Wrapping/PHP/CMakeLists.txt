# Required:
# sudo apt-get install php5-dev
find_package(SWIG REQUIRED)
include(${SWIG_USE_FILE})

include_directories(
  "${GDCM_BINARY_DIR}/Source/Common"
  "${GDCM_SOURCE_DIR}/Source/Common"
  "${GDCM_SOURCE_DIR}/Source/DataStructureAndEncodingDefinition"
  "${GDCM_SOURCE_DIR}/Source/InformationObjectDefinition"
  "${GDCM_SOURCE_DIR}/Source/MediaStorageAndFileFormat"
  "${GDCM_SOURCE_DIR}/Source/DataDictionary"
  "${GDCM_SOURCE_DIR}/Source/MessageExchangeDefinition"
)

#find_package(PHP4 REQUIRED)
find_package(PHP5 REQUIRED)

include_directories(
  ${PHP4_INCLUDE_PATH}
  ${PHP5_INCLUDE_PATH}
)

set_source_files_properties(gdcm.i PROPERTIES CPLUSPLUS ON)

# Some old swig 1.3 did not support this option:
#set(CMAKE_SWIG_OUTDIR "${CMAKE_CURRENT_BINARY_DIR}")

# swig expect a name like that: php_gdcm.so
SWIG_ADD_MODULE(php_gdcm php gdcm.i)
SWIG_LINK_LIBRARIES(php_gdcm gdcmMSFF gdcmMEXD)
# warning, by default there are undefined symbols in the generated php/gdcm module
# one would need to actually link to : /usr/lib/apache2/modules/libphp5.so
# I do not believe this is correct, instead make sure that
# CMAKE_MODULE_LINKER_FLAGS = -Wl,--no-undefined
# will make gdcm fails to compile:
# with something like undefined reference to `zend_error'
set_property(TARGET ${SWIG_MODULE_php_gdcm_REAL_NAME} PROPERTY NO_SONAME 1)

set_target_properties(${SWIG_MODULE_php_gdcm_REAL_NAME} PROPERTIES LINK_INTERFACE_LIBRARIES "")
# Remove 'lib' prefix :no-op on windows and valid for UNIX based syste
set_target_properties(${SWIG_MODULE_php_gdcm_REAL_NAME} PROPERTIES PREFIX "")

# For some reason the php glue module name is odd using swig 1.3.40, it is php_gdcm on windows
# while simply gdcm on other unix system:
# ...
#    if (!dl('php_gdcm.dll')) return;
#  } else {
#    // PHP_SHLIB_SUFFIX gives 'dylib' on MacOS X but modules are 'so'.
#    if (PHP_SHLIB_SUFFIX === 'dylib') {
#      if (!dl('gdcm.so')) return;
#    } else {
#      if (!dl('gdcm.'.PHP_SHLIB_SUFFIX)) return;
# ...
if(UNIX)
  set_target_properties(${SWIG_MODULE_php_gdcm_REAL_NAME} PROPERTIES OUTPUT_NAME "gdcm")
endif()

# See gdcm bug #3175803
# This is fixed in SWIG 2.0.2:
# http://sourceforge.net/tracker/?func=detail&atid=101645&aid=3166423&group_id=1645
if(${SWIG_VERSION} LESS 2.0.2)
  add_custom_command(TARGET ${SWIG_MODULE_php_gdcm_REAL_NAME}
    PRE_BUILD
    COMMAND sed -i -e 's/zend_error_noreturn/zend_error/g' "${swig_generated_file_fullname}"
    COMMENT "Patching zend_error_noreturn into zend_error"
  )
endif()

# Let's copy gdcm.php into the bin dir:
add_custom_command(
  OUTPUT ${LIBRARY_OUTPUT_PATH}/gdcm.php
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/gdcm.php ${LIBRARY_OUTPUT_PATH}/gdcm.php
  DEPENDS "${swig_generated_file_fullname}"
  COMMENT "copying gdcm.php"
)
add_custom_target(GDCMPHP ALL
  DEPENDS ${LIBRARY_OUTPUT_PATH}/gdcm.php
  COMMENT "building gdcm.php"
)

# Module are always place in the library destination
# but for poor win32 user I decided to place them
# right next to the other dlls
if(NOT GDCM_INSTALL_NO_LIBRARIES)
  install_swig_module(php_gdcm PHP)
  # because gdcm.php is constructed with custom commands, it need the install(FILES signature:
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/gdcm.php
    DESTINATION ${GDCM_INSTALL_LIB_DIR} COMPONENT PHPModule
  )
endif()
