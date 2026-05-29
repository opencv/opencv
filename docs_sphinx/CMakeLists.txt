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

# Top-level standalone pages staged at srcdir root (siblings of tutorials/).
# Add further `file(CREATE_LINK …)` lines here if more top-level pages get
# onboarded.  intro.markdown lives inside modules/core/doc/ (it's the core
# module's @page; the legacy Doxygen build pulls it in via the same module
# scan it uses for bib files) — symlink it under a flat name at srcdir root.
file(CREATE_LINK
  "${CMAKE_SOURCE_DIR}/doc/faq.markdown"
  "${_SPHINX_INPUT_ROOT}/faq.markdown"
  SYMBOLIC COPY_ON_ERROR)
file(CREATE_LINK
  "${CMAKE_SOURCE_DIR}/modules/core/doc/intro.markdown"
  "${_SPHINX_INPUT_ROOT}/intro.markdown"
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

# js_tutorials, py_tutorials, and images/ sit directly under opencv/doc/ —
# symlink them into the staged input tree.
foreach(_root js_tutorials py_tutorials images)
  if(EXISTS "${CMAKE_SOURCE_DIR}/doc/${_root}")
    file(CREATE_LINK
      "${CMAKE_SOURCE_DIR}/doc/${_root}"
      "${_SPHINX_INPUT_ROOT}/${_root}"
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

# ---------------------------------------------------------------------------
# API reference: Doxygen XML generation (Phase A of API onboarding).
# Strategy: write a build-tree Doxyfile-xml that @INCLUDEs the legacy
# <build>/doc/Doxyfile (produced by opencv/doc/CMakeLists.txt's
# configure_file(Doxyfile.in -> Doxyfile) at line 227) and overrides ONLY
# the format flags. Doxyfile.in is never modified — its GENERATE_XML=NO at
# line 288 stays NO. add_subdirectory(doc) runs at opencv/CMakeLists.txt
# line 1042, before add_subdirectory(docs_sphinx) at line 1043, so the
# legacy Doxyfile already exists by the time this branch is evaluated.
# ---------------------------------------------------------------------------
set(_LEGACY_DOXYFILE "${CMAKE_BINARY_DIR}/doc/Doxyfile")
set(_DOXYGEN_XML_DIR "${CMAKE_BINARY_DIR}/doc/doxygen/xml")

# Modules to process with Doxygen — must match OPENCV_API_MODULES used at
# sphinx-build time. Default "core" matches conf.py's OPENCV_API_MODULES default.
# Override: cmake -DSPHINX_API_MODULES="core;photo;imgproc;objdetect;3d"
if(NOT DEFINED SPHINX_API_MODULES)
  set(SPHINX_API_MODULES "core")
endif()
set(_doxy_input "")
foreach(_m ${SPHINX_API_MODULES})
  foreach(_base "${CMAKE_SOURCE_DIR}/modules" "${OPENCV_EXTRA_MODULES_PATH}")
    if(EXISTS "${_base}/${_m}/include")
      string(APPEND _doxy_input " ${_base}/${_m}/include")
      break()
    endif()
  endforeach()
endforeach()

if(EXISTS "${_LEGACY_DOXYFILE}")
  set(_SPHINX_DOXYFILE "${CMAKE_CURRENT_BINARY_DIR}/Doxyfile-xml")
  file(WRITE "${_SPHINX_DOXYFILE}"
    "@INCLUDE = ${_LEGACY_DOXYFILE}\n"
    "GENERATE_HTML = NO\n"
    "GENERATE_LATEX = NO\n"
    "GENERATE_XML = YES\n"
    "XML_OUTPUT = xml\n"
    "XML_PROGRAMLISTING = NO\n"
    "CREATE_SUBDIRS = NO\n"
    "EXCLUDE_PATTERNS += cap_ios.h\n"
    "CLASS_GRAPH = NO\n"
    "COLLABORATION_GRAPH = NO\n"
    "GROUP_GRAPHS = NO\n"
    "INCLUDE_GRAPH = NO\n"
    "INCLUDED_BY_GRAPH = NO\n"
    "DIRECTORY_GRAPH = NO\n"
    "INPUT =${_doxy_input}\n"
    "RECURSIVE = YES\n"
    # CV_CREATE_MEMBER_CHECK expands to complex template code that crashes Doxygen; disable expansion entirely.
    "MACRO_EXPANSION = NO\n"
  )
  find_program(DOXYGEN_EXE NAMES doxygen)
  if(DOXYGEN_EXE)
    # Stamp-based: Doxygen only re-runs when Doxyfile changes, not on every build.
    set(_SPHINX_XML_STAMP "${_DOXYGEN_XML_DIR}/.sphinx_xml.stamp")
    add_custom_command(
      OUTPUT "${_SPHINX_XML_STAMP}"
      COMMAND ${CMAKE_COMMAND} -E make_directory ${_DOXYGEN_XML_DIR}
      COMMAND ${DOXYGEN_EXE} ${_SPHINX_DOXYFILE}
      COMMAND ${CMAKE_COMMAND} -E touch "${_SPHINX_XML_STAMP}"
      DEPENDS "${_SPHINX_DOXYFILE}"
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/doc
      COMMENT "Doxygen XML (for Sphinx breathe) -> ${_DOXYGEN_XML_DIR}"
      VERBATIM)
    add_custom_target(sphinx-xml DEPENDS "${_SPHINX_XML_STAMP}")
    message(STATUS "docs_sphinx: sphinx-xml target enabled (doxygen: ${DOXYGEN_EXE})")
  else()
    message(STATUS "docs_sphinx: doxygen not found; `sphinx-xml` disabled")
  endif()

  # API stubs are generated by conf.py at sphinx-build time, not here. We need
  # the Doxygen XML to walk the @defgroup hierarchy via <innergroup> elements;
  # that hierarchy isn't available at CMake configure time (sphinx-xml runs
  # later). Doing it in Python at sphinx-build time also keeps the CMake side
  # minimal — Doxyfile-xml + the sphinx-xml target are all the build system
  # needs to know about. See conf.py's `_generate_api_stubs()`.
else()
  message(STATUS "docs_sphinx: legacy Doxyfile not present at ${_LEGACY_DOXYFILE}; "
                 "`sphinx-xml` disabled (re-run cmake after doc/ subdir configures)")
endif()

# Forward staging info to conf.py via env. OPENCV_CONTRIB_MODULES is NOT
# forwarded — conf.py's default list wins unless the caller sets it in their
# shell env (OPENCV_CONTRIB_MODULES=sfm cmake --build ...) to override per-build.
set(_SPHINX_ENV
  "OPENCV_SPHINX_INPUT_ROOT=${_SPHINX_INPUT_ROOT}"
  "OPENCV_CONTRIB_ROOT=${OPENCV_EXTRA_MODULES_PATH}"
  "OPENCV_DOXYGEN_XML_DIR=${_DOXYGEN_XML_DIR}"
)
if(OPENCV_PYTHON_SIGNATURES_FILE)
  list(APPEND _SPHINX_ENV "OPENCV_PYTHON_SIGNATURES_FILE=${OPENCV_PYTHON_SIGNATURES_FILE}")
endif()

# Parallel knob. Sphinx supports `-j auto` (=nproc), `-j N`, or a literal
# integer; override per-build with -DSPHINX_JOBS=8. `auto` is the sweet spot
# on developer machines — using nproc workers takes the API-reference build
# from minutes to seconds. conf.py declares `parallel_read_safe`/
# `parallel_write_safe`, and breathe ≥ 4.30 is parallel-safe, so this is
# unconditionally enabled. `--keep-going` keeps the build progressing past
# non-fatal warnings (we still see them in the log).
if(NOT SPHINX_JOBS)
  set(SPHINX_JOBS "auto")
endif()
set(_SPHINX_WARNINGS "${CMAKE_CURRENT_BINARY_DIR}/sphinx-warnings.log")
add_custom_target(sphinx
  COMMAND ${CMAKE_COMMAND} -E make_directory ${_SPHINX_OUTDIR}
  COMMAND ${CMAKE_COMMAND} -E env ${_SPHINX_ENV}
          ${SPHINX_BUILD} -j ${SPHINX_JOBS} --keep-going
                          -c ${_SPHINX_CONFDIR}
                          ${_SPHINX_INPUT_ROOT}
                          ${_SPHINX_OUTDIR}
  WORKING_DIRECTORY ${_SPHINX_CONFDIR}
  COMMENT "Building Sphinx HTML site -> ${_SPHINX_OUTDIR} (jobs=${SPHINX_JOBS})"
  VERBATIM)

# Ensure breathe sees current XML on every `sphinx` build.
if(TARGET sphinx-xml)
  add_dependencies(sphinx sphinx-xml)
endif()
# Auto-generate Python binding signatures (needed for Python names in enum tables).
if(TARGET gen_opencv_python_source)
  add_dependencies(sphinx gen_opencv_python_source)
endif()

# Clean rebuild: `cmake --build <build> --target sphinx-clean`
add_custom_target(sphinx-clean
  COMMAND ${CMAKE_COMMAND} -E rm -rf ${_SPHINX_OUTDIR}
  COMMAND ${CMAKE_COMMAND} -E rm -rf ${CMAKE_CURRENT_BINARY_DIR}/.doctrees
  COMMENT "Removing ${_SPHINX_OUTDIR} and .doctrees env cache"
  VERBATIM)

message(STATUS "docs_sphinx: sphinx target enabled (sphinx-build: ${SPHINX_BUILD})")
