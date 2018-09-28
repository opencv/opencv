if(NOT BUILD_DOCS)
  return()
endif()

# Dependencies scheme (* - optional):
#
# javadoc* -> doxygen_javadoc* -> doxygen_cpp ---------> doxygen -> opencv_docs
#    \                               \                     /        /
#     \                               -> doxygen_python* ->        /
#      ---------------------------------------------------------->

find_package(Doxygen)
if(DOXYGEN_FOUND)
  add_custom_target(doxygen)

  # not documented modules list
  set(blacklist "${DOXYGEN_BLACKLIST}")
  list(APPEND blacklist "ts" "java_bindings_generator" "java" "python_bindings_generator" "python2" "python3" "js" "world")
  unset(CMAKE_DOXYGEN_TUTORIAL_CONTRIB_ROOT)
  unset(CMAKE_DOXYGEN_TUTORIAL_JS_ROOT)

  set(OPENCV_MATHJAX_RELPATH "https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0" CACHE STRING "URI to a MathJax installation")

  # gathering headers
  set(paths_include)
  set(paths_doc)
  set(paths_bib)
  set(paths_sample)
  set(paths_tutorial)
  set(paths_hal_interface)
  set(refs_main)
  set(refs_extra)
  set(deps)
  foreach(m ${OPENCV_MODULES_MAIN} ${OPENCV_MODULES_EXTRA})
    list(FIND blacklist ${m} _pos)
    if(${_pos} EQUAL -1)
      # include folder
      set(header_dir "${OPENCV_MODULE_opencv_${m}_LOCATION}/include")
      if(EXISTS "${header_dir}")
        list(APPEND paths_include "${header_dir}")
        list(APPEND deps ${header_dir})
      endif()
      # doc folder
      set(docs_dir "${OPENCV_MODULE_opencv_${m}_LOCATION}/doc")
      if(EXISTS "${docs_dir}")
        list(APPEND paths_doc "${docs_dir}")
        list(APPEND deps ${docs_dir})
      endif()
      # sample folder
      set(sample_dir "${OPENCV_MODULE_opencv_${m}_LOCATION}/samples")
      if(EXISTS "${sample_dir}")
        list(APPEND paths_sample "${sample_dir}")
        list(APPEND deps ${sample_dir})
      endif()
      # tutorial folder
      set(tutorial_dir "${OPENCV_MODULE_opencv_${m}_LOCATION}/tutorials")
      if(EXISTS "${tutorial_dir}")
        list(APPEND paths_tutorial "${tutorial_dir}")
        list(APPEND deps ${tutorial_dir})

        # tutorial reference entry
        file(GLOB tutorials RELATIVE "${OPENCV_MODULE_opencv_${m}_LOCATION}" "${tutorial_dir}/*.markdown")
        foreach (t ${tutorials})
          if (NOT DEFINED CMAKE_DOXYGEN_TUTORIAL_CONTRIB_ROOT)
            set(CMAKE_DOXYGEN_TUTORIAL_CONTRIB_ROOT "- @ref tutorial_contrib_root")
            set(tutorial_contrib_root "${CMAKE_CURRENT_BINARY_DIR}/contrib_tutorials.markdown")
            file(WRITE "${tutorial_contrib_root}"
              "Tutorials for contrib modules {#tutorial_contrib_root}\n"
              "=============================\n")
          endif()
          file(STRINGS "${OPENCV_MODULE_opencv_${m}_LOCATION}/${t}" tutorial_id LIMIT_COUNT 1 REGEX ".*{#[^}]+}")
          string(REGEX REPLACE ".*{#([^}]+)}" "\\1" tutorial_id "${tutorial_id}")
          file(APPEND "${tutorial_contrib_root}" "- ${m}. @subpage ${tutorial_id}\n")
        endforeach()
      endif()
      # HAL replacement file
      set(replacement_header "${OPENCV_MODULE_opencv_${m}_LOCATION}/src/hal_replacement.hpp")
      if(EXISTS "${replacement_header}")
        list(APPEND paths_hal_interface "${replacement_header}")
      endif()

      # BiBTeX file
      set(bib_file "${docs_dir}/${m}.bib")
      if(EXISTS "${bib_file}")
        set(paths_bib "${paths_bib} ${bib_file}")
        list(APPEND deps ${bib_file})
      endif()
      # Reference entry
      set(one_ref "\t- ${m}. @ref ${m}\n")
      list(FIND OPENCV_MODULES_EXTRA ${m} _pos)
      if(${_pos} EQUAL -1)
        set(refs_main "${refs_main}${one_ref}")
      else()
        set(refs_extra "${refs_extra}${one_ref}")
      endif()
    endif()
  endforeach()

  # fix references
  # set(ref_header "Module name | Folder\n----------- | ------")
  # if(refs_main)
  #    set(refs_main "### Main modules\n${ref_header}\n${refs_main}")
  # endif()
  # if(refs_extra)
  #   set(refs_extra "### Extra modules\n${ref_header}\n${refs_extra}")
  # endif()
  if(refs_main)
    set(refs_main "- Main modules:\n${refs_main}")
  endif()
  if(refs_extra)
    set(refs_extra "- Extra modules:\n${refs_extra}")
  endif()

  # additional config
  set(doxyfile "${CMAKE_CURRENT_BINARY_DIR}/Doxyfile")
  set(rootfile "${CMAKE_CURRENT_BINARY_DIR}/root.markdown")
  set(bibfile "${CMAKE_CURRENT_SOURCE_DIR}/opencv.bib")
  set(faqfile "${CMAKE_CURRENT_SOURCE_DIR}/faq.markdown")
  set(tutorial_path "${CMAKE_CURRENT_SOURCE_DIR}/tutorials")
  set(tutorial_py_path "${CMAKE_CURRENT_SOURCE_DIR}/py_tutorials")
  set(CMAKE_DOXYGEN_TUTORIAL_JS_ROOT "- @ref tutorial_js_root")
  set(tutorial_js_path "${CMAKE_CURRENT_SOURCE_DIR}/js_tutorials")
  set(example_path "${CMAKE_SOURCE_DIR}/samples")

  # set export variables
  string(REPLACE ";" " \\\n" CMAKE_DOXYGEN_INPUT_LIST "${rootfile} ; ${faqfile} ; ${paths_include} ; ${paths_hal_interface} ; ${paths_doc} ; ${tutorial_path} ; ${tutorial_py_path} ; ${tutorial_js_path} ; ${paths_tutorial} ; ${tutorial_contrib_root}")
  string(REPLACE ";" " \\\n" CMAKE_DOXYGEN_IMAGE_PATH "${paths_doc} ; ${tutorial_path} ; ${tutorial_py_path} ; ${tutorial_js_path} ; ${paths_tutorial}")
  # TODO: remove paths_doc from EXAMPLE_PATH after face module tutorials/samples moved to separate folders
  string(REPLACE ";" " \\\n" CMAKE_DOXYGEN_EXAMPLE_PATH  "${example_path} ; ${paths_doc} ; ${paths_sample}")
  set(CMAKE_DOXYGEN_LAYOUT "${CMAKE_CURRENT_BINARY_DIR}/DoxygenLayout.xml")
  set(CMAKE_DOXYGEN_OUTPUT_PATH "doxygen")
  set(CMAKE_DOXYGEN_MAIN_REFERENCE "${refs_main}")
  set(CMAKE_DOXYGEN_EXTRA_REFERENCE "${refs_extra}")
  set(CMAKE_EXTRA_BIB_FILES "${bibfile} ${paths_bib}")
  if (CMAKE_DOXYGEN_GENERATE_QHP)
    set(CMAKE_DOXYGEN_GENERATE_QHP "YES")
  else()
    set(CMAKE_DOXYGEN_GENERATE_QHP "NO")
  endif()

  list(APPEND CMAKE_DOXYGEN_HTML_FILES "${CMAKE_CURRENT_SOURCE_DIR}/opencv.ico")
  list(APPEND CMAKE_DOXYGEN_HTML_FILES "${CMAKE_CURRENT_SOURCE_DIR}/pattern.png")
  list(APPEND CMAKE_DOXYGEN_HTML_FILES "${CMAKE_CURRENT_SOURCE_DIR}/acircles_pattern.png")
  list(APPEND CMAKE_DOXYGEN_HTML_FILES "${CMAKE_CURRENT_SOURCE_DIR}/bodybg.png")
  # list(APPEND CMAKE_DOXYGEN_HTML_FILES "${CMAKE_CURRENT_SOURCE_DIR}/mymath.sty")
  list(APPEND CMAKE_DOXYGEN_HTML_FILES "${CMAKE_CURRENT_SOURCE_DIR}/tutorial-utils.js")
  string(REPLACE ";" " \\\n" CMAKE_DOXYGEN_HTML_FILES "${CMAKE_DOXYGEN_HTML_FILES}")

  # writing file
  configure_file(DoxygenLayout.xml DoxygenLayout.xml @ONLY)
  configure_file(Doxyfile.in ${doxyfile} @ONLY)
  configure_file(root.markdown.in ${rootfile} @ONLY)

  # js tutorial assets
  set(opencv_tutorial_html_dir "${CMAKE_CURRENT_BINARY_DIR}/doxygen/html")
  set(js_tutorials_assets_dir "${CMAKE_CURRENT_SOURCE_DIR}/js_tutorials/js_assets")
  set(js_tutorials_assets_deps "")

  # make sure the build directory exists
  file(MAKE_DIRECTORY "${opencv_tutorial_html_dir}")

  # gather and copy specific files for js tutorials
  file(GLOB_RECURSE js_assets "${js_tutorials_assets_dir}/*")
  ocv_list_filterout(js_assets "\\\\.eslintrc.json")
  list(APPEND js_assets "${OpenCV_SOURCE_DIR}/samples/cpp/tutorial_code/calib3d/real_time_pose_estimation/Data/box.mp4")

  if(BUILD_opencv_js)
    set(ocv_js_dir "${CMAKE_BINARY_DIR}/bin")
    set(ocv_js "opencv.js")
    list(APPEND js_assets "${ocv_js_dir}/${ocv_js}")
  elseif(DEFINED OPENCV_JS_LOCATION)
    list(APPEND js_assets "${OPENCV_JS_LOCATION}")
  endif()

  # copy haar cascade files
  set(haar_cascade_files "")
  set(data_harrcascades_path "${OpenCV_SOURCE_DIR}/data/haarcascades/")
  list(APPEND js_tutorials_assets_deps "${data_harrcascades_path}/haarcascade_frontalface_default.xml" "${data_harrcascades_path}/haarcascade_eye.xml")
  list(APPEND js_assets "${data_harrcascades_path}/haarcascade_frontalface_default.xml" "${data_harrcascades_path}/haarcascade_eye.xml")

  foreach(f ${js_assets})
    get_filename_component(fname "${f}" NAME)
    add_custom_command(OUTPUT "${opencv_tutorial_html_dir}/${fname}"
                       COMMAND ${CMAKE_COMMAND} -E copy_if_different "${f}" "${opencv_tutorial_html_dir}/${fname}"
                       DEPENDS "${f}"
                       COMMENT "Copying ${fname}"
    )
    list(APPEND js_tutorials_assets_deps "${f}" "${opencv_tutorial_html_dir}/${fname}")
  endforeach()

  add_custom_target(
    doxygen_cpp
    COMMAND ${DOXYGEN_EXECUTABLE} ${doxyfile}
    DEPENDS ${doxyfile} ${rootfile} ${bibfile} ${deps} ${js_tutorials_assets_deps}
    COMMENT "Generate Doxygen documentation"
  )

  if(NOT DEFINED HAVE_PYTHON_BS4 AND PYTHON_DEFAULT_EXECUTABLE)
    # Documentation post-processing tool requires BuautifulSoup Python package
    execute_process(COMMAND "${PYTHON_DEFAULT_EXECUTABLE}" -c "import bs4; from bs4 import BeautifulSoup; print(bs4.__version__)"
                    RESULT_VARIABLE _result
                    OUTPUT_VARIABLE _bs4_version
                    OUTPUT_STRIP_TRAILING_WHITESPACE)

    if(NOT _result EQUAL 0)
      set(HAVE_PYTHON_BS4 0 CACHE INTERNAL "")
    else()
      message(STATUS "Python BeautifulSoup (bs4) version: ${_bs4_version}")
      set(HAVE_PYTHON_BS4 1 CACHE INTERNAL "")
    endif()
  endif()

  if(PYTHON_DEFAULT_EXECUTABLE
      AND HAVE_PYTHON_BS4
      AND OPENCV_PYTHON_SIGNATURES_FILE
      AND TARGET gen_opencv_python_source)
    add_custom_target(doxygen_python
      COMMAND ${PYTHON_DEFAULT_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/tools/add_signatures.py" "${CMAKE_CURRENT_BINARY_DIR}/doxygen/html/" "${OPENCV_PYTHON_SIGNATURES_FILE}" "python"
      DEPENDS doxygen_cpp gen_opencv_python_source
      COMMENT "Inject Python signatures into documentation"
    )
  endif()

  add_dependencies(doxygen doxygen_cpp)

  if(TARGET doxygen_python)
    add_dependencies(doxygen doxygen_python)
  endif()

  if(TARGET doxygen_javadoc)
    add_dependencies(doxygen_cpp doxygen_javadoc)
  endif()

  add_dependencies(opencv_docs doxygen)

  install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/doxygen/html
    DESTINATION "${OPENCV_DOC_INSTALL_PATH}"
    COMPONENT "docs" OPTIONAL
    ${compatible_MESSAGE_NEVER}
  )
endif()
