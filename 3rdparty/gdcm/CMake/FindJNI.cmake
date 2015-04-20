#  Copyright (c) 2006-2011 Mathieu Malaterre <mathieu.malaterre@gmail.com>
#
#  Redistribution and use is allowed according to the terms of the New
#  BSD license.
#  For details see the accompanying COPYING-CMAKE-SCRIPTS file.

# Make sure to not use FindJNI anymore and prefer FindJavaProperties
find_package(JavaProperties REQUIRED)
find_path(JAVA_INCLUDE_PATH jni.h
  ${JavaProp_JAVA_HOME}/../include
)

string(TOLOWER ${JavaProp_OS_NAME} include_os_name) # Linux -> linux
set(JAVA_JNI_MD_INCLUDE_DIRECTORIES
  ${JAVA_INCLUDE_PATH}/${include_os_name}
  ${JAVA_INCLUDE_PATH}/win32 # win32
  ${JAVA_INCLUDE_PATH}/linux # kFreeBSD
  ${JAVA_INCLUDE_PATH}/solaris # SunOS
  )
find_path(JAVA_INCLUDE_PATH2 jni_md.h
  ${JAVA_JNI_MD_INCLUDE_DIRECTORIES}
  )

find_path(JAVA_AWT_INCLUDE_PATH jawt.h
  ${JAVA_INCLUDE_PATH}
)

set(JAVA_AWT_LIBRARY_DIRECTORIES
  ${JavaProp_SUN_BOOT_LIBRARY_PATH} # works for linux
  ${JavaProp_JAVA_HOME}/../lib # works for win32
  )

foreach(dir ${JAVA_AWT_LIBRARY_DIRECTORIES})
  set(JAVA_JVM_LIBRARY_DIRECTORIES
    ${JAVA_JVM_LIBRARY_DIRECTORIES}
    "${dir}"
    "${dir}/client"
    "${dir}/server"
    )
endforeach()

find_library(JAVA_AWT_LIBRARY NAMES jawt
  PATHS ${JAVA_AWT_LIBRARY_DIRECTORIES}
  )

find_library(JAVA_JVM_LIBRARY NAMES jvm JavaVM
  PATHS ${JAVA_JVM_LIBRARY_DIRECTORIES}
  )

# on linux I get this annoying error:
# Exception in thread "main" java.lang.UnsatisfiedLinkError: libvtkgdcmJava.so:
# libmawt.so: cannot open shared object file: No such file or directory

# let's find this lib here then
if(UNIX)
  find_library(JAVA_MAWT_LIBRARY NAMES mawt
    # there is one also in headless but it does not work...
    PATHS ${JavaProp_SUN_BOOT_LIBRARY_PATH}/xawt
    )
endif()

FIND_PACKAGE_HANDLE_STANDARD_ARGS(JNI  DEFAULT_MSG  JAVA_AWT_LIBRARY JAVA_JVM_LIBRARY
                                                    JAVA_INCLUDE_PATH  JAVA_INCLUDE_PATH2 JAVA_AWT_INCLUDE_PATH)

mark_as_advanced(
  JAVA_AWT_LIBRARY
  JAVA_MAWT_LIBRARY
  JAVA_JVM_LIBRARY
  JAVA_AWT_INCLUDE_PATH
  JAVA_INCLUDE_PATH
  JAVA_INCLUDE_PATH2
)

set(JNI_LIBRARIES
  ${JAVA_AWT_LIBRARY}
  ${JAVA_JVM_LIBRARY}
)

set(JNI_INCLUDE_DIRS
  ${JAVA_INCLUDE_PATH}
  ${JAVA_INCLUDE_PATH2}
  ${JAVA_AWT_INCLUDE_PATH}
)
