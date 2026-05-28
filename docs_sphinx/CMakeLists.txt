# Sphinx wrapper for the OpenCV doc/ tree.
# Adds a `sphinx` custom target so `cmake --build <build> --target sphinx`
# works the same way `--target doxygen` does.
#
# Gated on BUILD_DOCS so it follows the existing toggle.

if(NOT BUILD_DOCS)
  return()
endif()

# Prefer the sphinx-build that lives in the conda env where myst-parser /
# pydata-sphinx-theme / breathe / exhale are installed. Override on the
# CMake command line:  -DSPHINX_BUILD=/path/to/sphinx-build
find_program(SPHINX_BUILD
  NAMES sphinx-build
  DOC "Path to sphinx-build tool")

if(NOT SPHINX_BUILD)
  message(STATUS "docs_sphinx: sphinx-build not found; `sphinx` target disabled")
  return()
endif()

set(_SPHINX_SOURCE   "${CMAKE_SOURCE_DIR}/doc")
set(_SPHINX_CONFDIR  "${CMAKE_CURRENT_SOURCE_DIR}")
set(_SPHINX_OUTDIR   "${CMAKE_CURRENT_BINARY_DIR}/html")

add_custom_target(sphinx
  COMMAND ${CMAKE_COMMAND} -E make_directory ${_SPHINX_OUTDIR}
  COMMAND ${SPHINX_BUILD} -c ${_SPHINX_CONFDIR} ${_SPHINX_SOURCE} ${_SPHINX_OUTDIR}
  WORKING_DIRECTORY ${_SPHINX_CONFDIR}
  COMMENT "Building Sphinx HTML site -> ${_SPHINX_OUTDIR}"
  VERBATIM)

# Clean rebuild: `cmake --build <build> --target sphinx-clean`
add_custom_target(sphinx-clean
  COMMAND ${CMAKE_COMMAND} -E rm -rf ${_SPHINX_OUTDIR}
  COMMENT "Removing ${_SPHINX_OUTDIR}"
  VERBATIM)

message(STATUS "docs_sphinx: sphinx target enabled (sphinx-build: ${SPHINX_BUILD})")
