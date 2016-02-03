# A C# Module for cmake
#
# TODO:
# Should I inspect the ENV{CSC} var first ?
#
#
#  Copyright (c) 2006-2011 Mathieu Malaterre <mathieu.malaterre@gmail.com>
#
#  Redistribution and use is allowed according to the terms of the New
#  BSD license.
#  For details see the accompanying COPYING-CMAKE-SCRIPTS file.
#

if(WIN32)
  find_package(DotNETFrameworkSDK)
else()
  #TODO handle CSharp_FIND_QUIETLY
  #TODO handle CSharp_FIND_REQUIRED
  find_package(MONO)
endif()

# http://public.kitware.com/Bug/view.php?id=7757
get_filename_component(current_list_path ${CMAKE_CURRENT_LIST_FILE} PATH)
set(CSharp_USE_FILE ${current_list_path}/UseCSharp.cmake)
