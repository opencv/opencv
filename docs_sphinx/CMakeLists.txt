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
  HINTS /home/abhishek/miniconda3/envs/opencvpy310/bin
  DOC "Path to sphinx-build")

if(NOT SPHINX_BUILD)
  message(STATUS "docs_sphinx: sphinx-build not found; `sphinx` target disabled")
  return()
endif()

set(_SPHINX_CONFDIR    "${CMAKE_CURRENT_SOURCE_DIR}")
set(_SPHINX_OUTDIR     "${CMAKE_CURRENT_BINARY_DIR}/html")
set(_SPHINX_INPUT_ROOT "${CMAKE_CURRENT_BINARY_DIR}/docs_sphinx_input")

# ---------------------------------------------------------------------------
# Stage a merged input tree at ${CMAKE_BINARY_DIR}/docs_sphinx_input/.
# Sphinx requires a single srcdir, but the wrapper needs to see both
# opencv/doc/tutorials/ and (optionally) opencv_contrib/modules/<m>/tutorials/.
# We symlink each main module subtree under tutorials/ and each enabled contrib
# module subtree under tutorials_contrib/, then auto-generate the contrib root
# page — the same pattern opencv/doc/CMakeLists.txt already uses for Doxygen.
# Nothing under opencv/doc/ or opencv_contrib/modules/ is modified.
# ---------------------------------------------------------------------------
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

# Contrib tree: only when OPENCV_EXTRA_MODULES_PATH is set, matching the legacy
# Doxygen contrib build's activation gate exactly. Stage every contrib module
# that has a tutorials/ dir — the canonical "which modules to compile" list
# lives in conf.py's CONTRIB_MODULES, exactly parallel to DOC_MODULES for main.
# Staged-but-unselected modules have their @subpage entries silently dropped
# by the toctree machinery (conf.py only populates _ANCHOR_TO_DOC for enabled
# modules), so listing them all in contrib_root.markdown is safe and means
# adding a new contrib module is a one-line conf.py edit (no reconfigure).
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
      # Match every top-level *.markdown in <m>/tutorials/, the same shape the
      # legacy Doxygen build uses (opencv/doc/CMakeLists.txt) — many contrib
      # modules use names like alphamat_tutorial.markdown or fuzzy.markdown
      # instead of the table_of_content_<m>.markdown convention.
      file(GLOB _tocs RELATIVE "${_tut_dir}" "${_tut_dir}/*.markdown")
      foreach(_t ${_tocs})
        file(STRINGS "${_tut_dir}/${_t}" _id LIMIT_COUNT 1 REGEX ".*\\{#[^}]+\\}")
        string(REGEX REPLACE ".*\\{#([^}]+)\\}.*" "\\1" _id "${_id}")
        if(_id)
          file(APPEND "${_contrib_root_md}" "- ${_m}. @subpage ${_id}\n")
        endif()
      endforeach()
    endif()
  endforeach()
endif()

# Forward staging info to conf.py via env. OPENCV_CONTRIB_MODULES is NOT
# forwarded — conf.py's default list wins unless the caller sets it in their
# shell env (OPENCV_CONTRIB_MODULES=ml cmake --build ...) to override per-build.
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
