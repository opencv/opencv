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

set(_SPHINX_CONFDIR    "${CMAKE_CURRENT_SOURCE_DIR}")
set(_SPHINX_OUTDIR     "${CMAKE_CURRENT_BINARY_DIR}/html")
set(_SPHINX_INPUT_ROOT "${CMAKE_CURRENT_BINARY_DIR}/docs_sphinx_input")

# Stage merged input tree: symlink main + contrib tutorial subtrees into one srcdir.
set(_SPHINX_INPUT_TUTORIALS "${_SPHINX_INPUT_ROOT}/tutorials")
set(_SPHINX_INPUT_CONTRIB   "${_SPHINX_INPUT_ROOT}/tutorials_contrib")
file(REMOVE_RECURSE "${_SPHINX_INPUT_ROOT}")
file(MAKE_DIRECTORY "${_SPHINX_INPUT_TUTORIALS}")

# Main tree: symlink the master file + each main module subtree.
file(CREATE_LINK
  "${CMAKE_SOURCE_DIR}/doc/tutorials/tutorials.markdown"
  "${_SPHINX_INPUT_TUTORIALS}/tutorials.markdown"
  SYMBOLIC COPY_ON_ERROR)
file(GLOB _main_tutorial_children
     RELATIVE "${CMAKE_SOURCE_DIR}/doc/tutorials"
     "${CMAKE_SOURCE_DIR}/doc/tutorials/*")
foreach(_d ${_main_tutorial_children})
  if(IS_DIRECTORY "${CMAKE_SOURCE_DIR}/doc/tutorials/${_d}")
    file(CREATE_LINK
      "${CMAKE_SOURCE_DIR}/doc/tutorials/${_d}"
      "${_SPHINX_INPUT_TUTORIALS}/${_d}"
      SYMBOLIC COPY_ON_ERROR)
  endif()
endforeach()

# js_tutorials, py_tutorials, images/ sit directly under opencv/doc/.
foreach(_root js_tutorials py_tutorials images)
  if(EXISTS "${CMAKE_SOURCE_DIR}/doc/${_root}")
    file(CREATE_LINK
      "${CMAKE_SOURCE_DIR}/doc/${_root}"
      "${_SPHINX_INPUT_ROOT}/${_root}"
      SYMBOLIC COPY_ON_ERROR)
  endif()
endforeach()

# Contrib tree: stage when OPENCV_EXTRA_MODULES_PATH is set; unselected modules dropped by conf.py.
if(OPENCV_EXTRA_MODULES_PATH AND EXISTS "${OPENCV_EXTRA_MODULES_PATH}")
  file(MAKE_DIRECTORY "${_SPHINX_INPUT_CONTRIB}")
  set(_contrib_root_md "${_SPHINX_INPUT_CONTRIB}/contrib_root.markdown")
  file(WRITE "${_contrib_root_md}"
    "Tutorials for contrib modules {#tutorial_contrib_root}\n"
    "=============================\n\n")
  file(GLOB _contrib_subdirs RELATIVE "${OPENCV_EXTRA_MODULES_PATH}"
       "${OPENCV_EXTRA_MODULES_PATH}/*")
  foreach(_m ${_contrib_subdirs})
    set(_tut_dir "${OPENCV_EXTRA_MODULES_PATH}/${_m}/tutorials")
    if(IS_DIRECTORY "${_tut_dir}")
      file(CREATE_LINK "${_tut_dir}" "${_SPHINX_INPUT_CONTRIB}/${_m}"
           SYMBOLIC COPY_ON_ERROR)
      file(GLOB _tocs RELATIVE "${_tut_dir}" "${_tut_dir}/*.markdown")
      foreach(_t ${_tocs})
        file(STRINGS "${_tut_dir}/${_t}" _id LIMIT_COUNT 1 REGEX ".*\\{#[^}]+\\}")
        string(REGEX REPLACE ".*\\{#([^}]+)\\}.*" "\\1" _id "${_id}")
        if(_id)
          file(APPEND "${_contrib_root_md}" "- @subpage ${_id}\n")
        endif()
      endforeach()
    endif()
  endforeach()
endif()

# Forward staging paths via env. OPENCV_CONTRIB_MODULES is not forwarded; set it in shell to override.
set(_SPHINX_ENV
  "OPENCV_SPHINX_INPUT_ROOT=${_SPHINX_INPUT_ROOT}"
  "OPENCV_CONTRIB_ROOT=${OPENCV_EXTRA_MODULES_PATH}"
)

add_custom_target(sphinx
  COMMAND ${CMAKE_COMMAND} -E make_directory ${_SPHINX_OUTDIR}
  COMMAND ${CMAKE_COMMAND} -E env ${_SPHINX_ENV}
          ${SPHINX_BUILD} -c ${_SPHINX_CONFDIR}
                          ${_SPHINX_INPUT_ROOT}
                          ${_SPHINX_OUTDIR}
  WORKING_DIRECTORY ${_SPHINX_CONFDIR}
  COMMENT "Building Sphinx HTML site -> ${_SPHINX_OUTDIR}"
  VERBATIM)

# Clean rebuild: `cmake --build <build> --target sphinx-clean`
add_custom_target(sphinx-clean
  COMMAND ${CMAKE_COMMAND} -E rm -rf ${_SPHINX_OUTDIR}
  COMMENT "Removing ${_SPHINX_OUTDIR}"
  VERBATIM)

message(STATUS "docs_sphinx: sphinx target enabled (sphinx-build: ${SPHINX_BUILD})")
