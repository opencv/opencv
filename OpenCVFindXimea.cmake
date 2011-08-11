# - Find XIMEA
# This module finds if XIMEA Software package is installed
# and determines where the binaries and header files are.
# This code sets the following variables:
#  
#  XIMEA_FOUND          - True if XIMEA API found
#  XIMEA_PATH:          - Path to the XIMEA API folder
#  XIMEA_LIBRARY_DIRS   - XIMEA libraries folder
#
# Created: 5 Aug 2011 by Marian Zajko (marian.zajko@ximea.com)
#

set(XIMEA_FOUND)
set(XIMEA_PATH)
set(XIMEA_LIBRARY_DIR)

# Try to find the XIMEA API path in registry.
GET_FILENAME_COMPONENT(XIMEA_PATH "[HKEY_CURRENT_USER\\Software\\XIMEA\\CamSupport\\API;Path]" ABSOLUTE)                                                

if(XIMEA_PATH)
  set(XIMEA_FOUND 1)
  
  # set LIB folders
  set(XIMEA_LIBRARY_DIR "${XIMEA_PATH}\\x86")
  
else()
  set(XIMEA_FOUND 0)
endif()                                            

mark_as_advanced(FORCE XIMEA_FOUND)
mark_as_advanced(FORCE XIMEA_PATH)
mark_as_advanced(FORCE XIMEA_LIBRARY_DIR)
  
   
   