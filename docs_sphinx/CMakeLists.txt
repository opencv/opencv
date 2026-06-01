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

find_program(DOT_EXECUTABLE NAMES dot
  DOC "Path to graphviz `dot` (required for API collaboration diagrams)")
if(NOT DOT_EXECUTABLE)
  message(WARNING
    "docs_sphinx: graphviz `dot` not found — API class-page collaboration "
    "diagrams will be skipped. Install graphviz (e.g. `sudo apt-get install "
    "graphviz`) and re-run cmake so the Doxygen HTML build enables HAVE_DOT.")
else()
  message(STATUS
    "docs_sphinx: graphviz dot found (${DOT_EXECUTABLE}) — "
    "collaboration diagrams enabled")
endif()

set(_SPHINX_CONFDIR    "${CMAKE_CURRENT_SOURCE_DIR}")
set(_SPHINX_OUTDIR     "${CMAKE_CURRENT_BINARY_DIR}/html")
set(_SPHINX_INPUT_ROOT "${CMAKE_CURRENT_BINARY_DIR}/docs_sphinx_input")

set(_SPHINX_INPUT_TUTORIALS "${_SPHINX_INPUT_ROOT}/tutorials")
set(_SPHINX_INPUT_CONTRIB   "${_SPHINX_INPUT_ROOT}/tutorials_contrib")
file(REMOVE_RECURSE "${_SPHINX_INPUT_ROOT}")
file(MAKE_DIRECTORY "${_SPHINX_INPUT_TUTORIALS}")

# Main tree: symlink the master file + each main module subtree.
file(CREATE_LINK
  "${CMAKE_SOURCE_DIR}/doc/tutorials/tutorials.markdown"
  "${_SPHINX_INPUT_TUTORIALS}/tutorials.markdown"
  SYMBOLIC COPY_ON_ERROR)

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

set(_LEGACY_DOXYFILE "${CMAKE_BINARY_DIR}/doc/Doxyfile")
set(_DOXYGEN_XML_DIR "${CMAKE_BINARY_DIR}/doc/doxygen/xml")

if(NOT DEFINED SPHINX_API_MODULES)
  set(SPHINX_API_MODULES "")
  file(GLOB _umbrella_headers "${CMAKE_SOURCE_DIR}/modules/*/include/opencv2/*.hpp")
  # Also scan opencv_contrib modules when present.
  if(OPENCV_EXTRA_MODULES_PATH AND EXISTS "${OPENCV_EXTRA_MODULES_PATH}")
    file(GLOB _contrib_umbrellas "${OPENCV_EXTRA_MODULES_PATH}/*/include/opencv2/*.hpp")
    list(APPEND _umbrella_headers ${_contrib_umbrellas})
  endif()
  foreach(_hdr ${_umbrella_headers})
    get_filename_component(_stem "${_hdr}" NAME_WE)
    # Only the umbrella header modules/<m>/include/opencv2/<m>.hpp.
    if(_hdr MATCHES "/modules/${_stem}/include/opencv2/${_stem}\\.hpp$")
      file(READ "${_hdr}" _hdr_contents)
      if(_hdr_contents MATCHES "@defgroup")
        list(APPEND SPHINX_API_MODULES "${_stem}")
      endif()
    endif()
  endforeach()
  list(SORT SPHINX_API_MODULES)
  message(STATUS "docs_sphinx: discovered API modules: ${SPHINX_API_MODULES}")
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
    "EXCLUDE_SYMBOLS = cv::DataType<*> int void CV__* T __CV* cv::gapi::detail*\n"
    "MACRO_EXPANSION = YES\n"
    "EXPAND_ONLY_PREDEF = YES\n"
    "PREDEFINED += __device__= __host__= __forceinline__= __global__= __constant__= __shared__= __restrict__=\n"
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
else()
  message(STATUS "docs_sphinx: legacy Doxyfile not present at ${_LEGACY_DOXYFILE}; "
                 "`sphinx-xml` disabled (re-run cmake after doc/ subdir configures)")
endif()

set(_SPHINX_ENV
  "OPENCV_SPHINX_INPUT_ROOT=${_SPHINX_INPUT_ROOT}"
  "OPENCV_CONTRIB_ROOT=${OPENCV_EXTRA_MODULES_PATH}"
  "OPENCV_DOXYGEN_XML_DIR=${_DOXYGEN_XML_DIR}"
)
if(OPENCV_PYTHON_SIGNATURES_FILE)
  list(APPEND _SPHINX_ENV "OPENCV_PYTHON_SIGNATURES_FILE=${OPENCV_PYTHON_SIGNATURES_FILE}")
endif()

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

option(SPHINX_BUILD_DIAGRAMS
  "Rebuild the Doxygen HTML (collaboration diagrams) as part of the sphinx target" ON)
if(SPHINX_BUILD_DIAGRAMS AND TARGET doxygen)
  add_dependencies(sphinx doxygen)
endif()

# Clean rebuild: `cmake --build <build> --target sphinx-clean`
add_custom_target(sphinx-clean
  COMMAND ${CMAKE_COMMAND} -E rm -rf ${_SPHINX_OUTDIR}
  COMMAND ${CMAKE_COMMAND} -E rm -rf ${CMAKE_CURRENT_BINARY_DIR}/.doctrees
  COMMENT "Removing ${_SPHINX_OUTDIR} and .doctrees env cache"
  VERBATIM)

message(STATUS "docs_sphinx: sphinx target enabled (sphinx-build: ${SPHINX_BUILD})")
