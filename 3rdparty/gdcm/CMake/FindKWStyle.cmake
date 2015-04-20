#
# this module looks for KWStyle
# http://public.kitware.com/KWStyle
#
#
#  Copyright (c) 2009-2011 Mathieu Malaterre <mathieu.malaterre@gmail.com>
#
#  Redistribution and use is allowed according to the terms of the New
#  BSD license.
#  For details see the accompanying COPYING-CMAKE-SCRIPTS file.
#

find_program(KWSTYLE_EXECUTABLE
  NAMES KWStyle
  PATHS
  "[HKEY_LOCAL_MACHINE\\SOFTWARE\\Kitware Inc.\\KWStyle 1.0.0]/bin"
  )

#  option(KWSTYLE_USE_VIM_FORMAT "Set KWStyle to generate errors with a VIM-compatible format." OFF)
#  option(KWSTYLE_USE_MSVC_FORMAT "Set KWStyle to generate errors with a VisualStudio-compatible format." OFF)
#  mark_as_advanced(KWSTYLE_USE_VIM_FORMAT)
#  mark_as_advanced(KWSTYLE_USE_MSVC_FORMAT)
#
#  if(KWSTYLE_USE_VIM_FORMAT)
#    set(KWSTYLE_ARGUMENTS -vim ${KWSTYLE_ARGUMENTS})
#  endif()
#
#  if(KWSTYLE_USE_MSVC_FORMAT)
#    set(KWSTYLE_ARGUMENTS -msvc ${KWSTYLE_ARGUMENTS})
#  endif()


mark_as_advanced(
  KWSTYLE_EXECUTABLE
  )
