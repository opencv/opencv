set(the_description "Vision-Language Model based OCR / document understanding engines")

find_package(CURL QUIET)

ocv_add_module(vlm opencv_core opencv_imgproc opencv_imgcodecs opencv_dnn WRAP python)
ocv_glob_module_sources()
ocv_module_include_directories()
ocv_create_module()

if(CURL_FOUND)
  ocv_target_compile_definitions(${the_module} PRIVATE "HAVE_CURL=1")
  ocv_target_include_directories(${the_module} PRIVATE ${CURL_INCLUDE_DIRS})
  ocv_target_link_libraries(${the_module} PRIVATE ${CURL_LIBRARIES})
else()
  message(STATUS "vlm: libcurl not found -- cloud VLM providers (OpenAI/Anthropic/Gemini/Grok) will be unavailable at runtime")
endif()

ocv_add_accuracy_tests()
ocv_add_perf_tests()
ocv_add_samples()
