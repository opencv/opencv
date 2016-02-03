cmake_minimum_required(VERSION 2.8.9)
# simply copy this file anywhere on your system and execute like this:
# ctest -S mymachine_openjpeg.cmake -V
# This will retrieve/compile/run tests/upload to cdash OpenJPEG
# results will be available at: http://my.cdash.org/index.php?project=OPENJPEG

# Begin User inputs:
set( CTEST_SITE             "mymachine" ) # generally the output of hostname
set( CTEST_DASHBOARD_ROOT   "/tmp" ) # writable path
set( CTEST_CMAKE_GENERATOR  "Unix Makefiles" ) # What is your compilation apps ?
set( CTEST_BUILD_CONFIGURATION  Debug) # What type of build do you want ?
set( ENV{CFLAGS} "-Wall" ) # just for fun...

# For testing we need to define the path to J2KP4files
#  wget http://www.crc.ricoh.com/~gormish/jpeg2000conformance/j2kp4files_v1_5.zip
#  unzip j2kp4files_v1_5.zip
set( CACHE_CONTENTS "
BUILD_TESTING:BOOL=TRUE
JPEG2000_CONFORMANCE_DATA_ROOT:PATH=${CTEST_SOURCE_DIRECTORY}/J2KP4files" )
# End User inputs:

# You do not need to change anything after that:
# 1. openjpeg specific:
set( CTEST_PROJECT_NAME         "OPENJPEG" )
set( CTEST_SOURCE_NAME          OpenJPEG)
set( CTEST_BUILD_NAME           "${CMAKE_SYSTEM}-${CTEST_CMAKE_GENERATOR}-${CTEST_BUILD_CONFIGURATION}")
set( CTEST_BINARY_NAME          "${CTEST_SOURCE_NAME}-${CTEST_BUILD_NAME}")

# 2. cdash/openjpeg specific:
# svn checkout http://openjpeg.googlecode.com/svn/trunk/ openjpeg-read-only
set( CTEST_SVN_URL          "http://openjpeg.googlecode.com/svn/")
set( CTEST_UPDATE_COMMAND   "svn")
#set( CTEST_CHECKOUT_COMMAND "${CTEST_UPDATE_COMMAND} co ${CTEST_SVN_URL}/trunk ${CTEST_SOURCE_NAME}")
set( CTEST_CHECKOUT_COMMAND "${CTEST_UPDATE_COMMAND} co ${CTEST_SVN_URL}/branches/v2 ${CTEST_SOURCE_NAME}")

# 3. cmake specific:
set( CTEST_SOURCE_DIRECTORY "${CTEST_DASHBOARD_ROOT}/${CTEST_SOURCE_NAME}")
set( CTEST_BINARY_DIRECTORY "${CTEST_DASHBOARD_ROOT}/${CTEST_BINARY_NAME}")
set( CTEST_NOTES_FILES      "${CTEST_SCRIPT_DIRECTORY}/${CTEST_SCRIPT_NAME}")

ctest_empty_binary_directory( "${CTEST_BINARY_DIRECTORY}" )
file(WRITE "${CTEST_BINARY_DIRECTORY}/CMakeCache.txt" "${CACHE_CONTENTS}")

# Perform the Nightly build
ctest_start(Nightly)
ctest_update(SOURCE "${CTEST_SOURCE_DIRECTORY}")
ctest_configure(BUILD "${CTEST_BINARY_DIRECTORY}")
ctest_build(BUILD "${CTEST_BINARY_DIRECTORY}")
ctest_test(BUILD "${CTEST_BINARY_DIRECTORY}")
ctest_submit()
