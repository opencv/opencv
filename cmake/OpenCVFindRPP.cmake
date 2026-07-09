# ----------------------------------------------------------------------------
# Detect AMD ROCm Performance Primitives (RPP)
# ----------------------------------------------------------------------------

find_package(rpp QUIET)

if(rpp_FOUND)
  set(HAVE_RPP TRUE)
  message(STATUS "RPP found: ${rpp_VERSION}")
  message(STATUS "    includes: ${rpp_INCLUDE_DIR}")
  message(STATUS "    libs: ${rpp_LIBRARIES}")
else()
  set(HAVE_RPP FALSE)
  message(STATUS "RPP: Not found")
endif()
