diff --git a/cmake/developer_package/add_ie_target.cmake b/cmake/developer_package/add_ie_target.cmake
index d49f16a4d..2726ca787 100644
--- a/cmake/developer_package/add_ie_target.cmake
+++ b/cmake/developer_package/add_ie_target.cmake
@@ -92,7 +92,7 @@ function(addIeTarget)
     if (ARG_TYPE STREQUAL EXECUTABLE)
         add_executable(${ARG_NAME} ${all_sources})
     elseif(ARG_TYPE STREQUAL STATIC OR ARG_TYPE STREQUAL SHARED)
-        add_library(${ARG_NAME} ${ARG_TYPE} ${all_sources})
+        add_library(${ARG_NAME} ${ARG_TYPE} EXCLUDE_FROM_ALL ${all_sources})
     else()
         message(SEND_ERROR "Invalid target type ${ARG_TYPE} specified for target name ${ARG_NAME}")
     endif()
diff --git a/inference-engine/CMakeLists.txt b/inference-engine/CMakeLists.txt
index 1ac7fd8bf..df7091e51 100644
--- a/inference-engine/CMakeLists.txt
+++ b/inference-engine/CMakeLists.txt
@@ -39,7 +39,7 @@ if(ENABLE_TESTS)
     add_subdirectory(tests)
 endif()
 
-add_subdirectory(tools)
+#add_subdirectory(tools)
 
 function(ie_build_samples)
     # samples should be build with the same flags as from OpenVINO package,
@@ -58,7 +58,7 @@ endfunction()
 
 # gflags and format_reader targets are kept inside of samples directory and
 # they must be built even if samples build is disabled (required for tests and tools).
-ie_build_samples()
+#ie_build_samples()
 
 if(ENABLE_PYTHON)
     add_subdirectory(ie_bridges/python)
@@ -142,7 +142,7 @@ endif()
 # Developer package
 #
 
-openvino_developer_export_targets(COMPONENT openvino_common TARGETS format_reader gflags ie_samples_utils)
+#openvino_developer_export_targets(COMPONENT openvino_common TARGETS format_reader gflags ie_samples_utils)
 
 # for Template plugin
 if(NGRAPH_INTERPRETER_ENABLE)
@@ -166,7 +166,7 @@ function(ie_generate_dev_package_config)
                 @ONLY)
 endfunction()
 
-ie_generate_dev_package_config()
+#ie_generate_dev_package_config()
 
 #
 # Coverage
diff --git a/inference-engine/src/inference_engine/CMakeLists.txt b/inference-engine/src/inference_engine/CMakeLists.txt
index e8ed1a5c4..1fc9fc3ff 100644
--- a/inference-engine/src/inference_engine/CMakeLists.txt
+++ b/inference-engine/src/inference_engine/CMakeLists.txt
@@ -110,7 +110,7 @@ add_cpplint_target(${TARGET_NAME}_plugin_api_cpplint FOR_SOURCES ${plugin_api_sr
 
 # Create object library
 
-add_library(${TARGET_NAME}_obj OBJECT
+add_library(${TARGET_NAME}_obj OBJECT EXCLUDE_FROM_ALL
             ${LIBRARY_SRC}
             ${LIBRARY_HEADERS}
             ${PUBLIC_HEADERS})
@@ -181,7 +181,7 @@ ie_add_api_validator_post_build_step(TARGET ${TARGET_NAME})
 
 # Static library used for unit tests which are always built
 
-add_library(${TARGET_NAME}_s STATIC
+add_library(${TARGET_NAME}_s STATIC EXCLUDE_FROM_ALL
             $<TARGET_OBJECTS:${TARGET_NAME}_legacy_obj>
             $<TARGET_OBJECTS:${TARGET_NAME}_obj>
             ${IE_STATIC_DEPENDENT_FILES})
diff --git a/inference-engine/src/legacy_api/CMakeLists.txt b/inference-engine/src/legacy_api/CMakeLists.txt
index 8eae82bd2..e0e6745b1 100644
--- a/inference-engine/src/legacy_api/CMakeLists.txt
+++ b/inference-engine/src/legacy_api/CMakeLists.txt
@@ -26,7 +26,7 @@ endif()
 
 file(TOUCH ${CMAKE_CURRENT_BINARY_DIR}/dummy.cpp)
 
-add_library(${TARGET_NAME}_obj OBJECT
+add_library(${TARGET_NAME}_obj OBJECT EXCLUDE_FROM_ALL
             ${LIBRARY_SRC}
             ${PUBLIC_HEADERS})
 
diff --git a/inference-engine/src/mkldnn_plugin/CMakeLists.txt b/inference-engine/src/mkldnn_plugin/CMakeLists.txt
index fe57b29dd..07831e2fb 100644
--- a/inference-engine/src/mkldnn_plugin/CMakeLists.txt
+++ b/inference-engine/src/mkldnn_plugin/CMakeLists.txt
@@ -67,7 +67,7 @@ ie_add_api_validator_post_build_step(TARGET ${TARGET_NAME})
 
 #  add test object library
 
-add_library(${TARGET_NAME}_obj OBJECT ${SOURCES} ${HEADERS})
+add_library(${TARGET_NAME}_obj OBJECT EXCLUDE_FROM_ALL ${SOURCES} ${HEADERS})
 target_link_libraries(${TARGET_NAME}_obj PUBLIC mkldnn)
 
 target_include_directories(${TARGET_NAME}_obj PRIVATE $<TARGET_PROPERTY:inference_engine_preproc_s,INTERFACE_INCLUDE_DIRECTORIES>
diff --git a/inference-engine/src/preprocessing/CMakeLists.txt b/inference-engine/src/preprocessing/CMakeLists.txt
index f9548339d..ef962145a 100644
--- a/inference-engine/src/preprocessing/CMakeLists.txt
+++ b/inference-engine/src/preprocessing/CMakeLists.txt
@@ -101,7 +101,7 @@ endif()
 
 # Create object library
 
-add_library(${TARGET_NAME}_obj OBJECT
+add_library(${TARGET_NAME}_obj OBJECT EXCLUDE_FROM_ALL
             ${LIBRARY_SRC}
             ${LIBRARY_HEADERS})
 
@@ -153,7 +153,7 @@ ie_add_api_validator_post_build_step(TARGET ${TARGET_NAME})
 
 # Static library used for unit tests which are always built
 
-add_library(${TARGET_NAME}_s STATIC
+add_library(${TARGET_NAME}_s STATIC EXCLUDE_FROM_ALL
             $<TARGET_OBJECTS:${TARGET_NAME}_obj>)
 
 set_ie_threading_interface_for(${TARGET_NAME}_s)
diff --git a/inference-engine/src/vpu/common/CMakeLists.txt b/inference-engine/src/vpu/common/CMakeLists.txt
index 249e47c28..4ddf63049 100644
--- a/inference-engine/src/vpu/common/CMakeLists.txt
+++ b/inference-engine/src/vpu/common/CMakeLists.txt
@@ -5,7 +5,7 @@
 file(GLOB_RECURSE SOURCES *.cpp *.hpp *.h)
 
 function(add_common_target TARGET_NAME STATIC_IE)
-    add_library(${TARGET_NAME} STATIC ${SOURCES})
+    add_library(${TARGET_NAME} STATIC EXCLUDE_FROM_ALL ${SOURCES})
 
     ie_faster_build(${TARGET_NAME}
         UNITY
@@ -60,7 +60,7 @@ add_common_target("vpu_common_lib" FALSE)
 
 # Unit tests support for graph transformer
 if(WIN32)
-    add_common_target("vpu_common_lib_test_static" TRUE)
+    #add_common_target("vpu_common_lib_test_static" TRUE)
 else()
     add_library("vpu_common_lib_test_static" ALIAS "vpu_common_lib")
 endif()
diff --git a/inference-engine/src/vpu/graph_transformer/CMakeLists.txt b/inference-engine/src/vpu/graph_transformer/CMakeLists.txt
index bc73ab5b1..b4c1547fc 100644
--- a/inference-engine/src/vpu/graph_transformer/CMakeLists.txt
+++ b/inference-engine/src/vpu/graph_transformer/CMakeLists.txt
@@ -5,7 +5,7 @@
 file(GLOB_RECURSE SOURCES *.cpp *.hpp *.h *.inc)
 
 function(add_graph_transformer_target TARGET_NAME STATIC_IE)
-    add_library(${TARGET_NAME} STATIC ${SOURCES})
+    add_library(${TARGET_NAME} STATIC EXCLUDE_FROM_ALL ${SOURCES})
 
     set_ie_threading_interface_for(${TARGET_NAME})
 
@@ -70,7 +70,7 @@ add_graph_transformer_target("vpu_graph_transformer" FALSE)
 
 # Unit tests support for graph transformer
 if(WIN32)
-    add_graph_transformer_target("vpu_graph_transformer_test_static" TRUE)
+    #add_graph_transformer_target("vpu_graph_transformer_test_static" TRUE)
 else()
     add_library("vpu_graph_transformer_test_static" ALIAS "vpu_graph_transformer")
 endif()
diff --git a/inference-engine/thirdparty/pugixml/CMakeLists.txt b/inference-engine/thirdparty/pugixml/CMakeLists.txt
index 8bcb2801a..f7e031c01 100644
--- a/inference-engine/thirdparty/pugixml/CMakeLists.txt
+++ b/inference-engine/thirdparty/pugixml/CMakeLists.txt
@@ -41,7 +41,7 @@ if(BUILD_SHARED_LIBS)
 else()
 	add_library(pugixml STATIC ${SOURCES})
 	if (MSVC)
-		add_library(pugixml_mt STATIC ${SOURCES})
+                #add_library(pugixml_mt STATIC ${SOURCES})
 		#if (WIN32)
 		#	set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
 		#	set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
diff --git a/ngraph/core/builder/CMakeLists.txt b/ngraph/core/builder/CMakeLists.txt
index ff5c381e7..2797ec9ab 100644
--- a/ngraph/core/builder/CMakeLists.txt
+++ b/ngraph/core/builder/CMakeLists.txt
@@ -16,7 +16,7 @@ source_group("src" FILES ${LIBRARY_SRC})
 source_group("include" FILES ${PUBLIC_HEADERS})
 
 # Create shared library
-add_library(${TARGET_NAME} STATIC ${LIBRARY_SRC} ${PUBLIC_HEADERS})
+add_library(${TARGET_NAME} STATIC EXCLUDE_FROM_ALL ${LIBRARY_SRC} ${PUBLIC_HEADERS})
 
 if(COMMAND ie_faster_build)
     ie_faster_build(${TARGET_NAME}
diff --git a/ngraph/core/reference/CMakeLists.txt b/ngraph/core/reference/CMakeLists.txt
index ef4a764ab..f6d3172e2 100644
--- a/ngraph/core/reference/CMakeLists.txt
+++ b/ngraph/core/reference/CMakeLists.txt
@@ -16,7 +16,7 @@ source_group("src" FILES ${LIBRARY_SRC})
 source_group("include" FILES ${PUBLIC_HEADERS})
 
 # Create shared library
-add_library(${TARGET_NAME} STATIC ${LIBRARY_SRC} ${PUBLIC_HEADERS})
+add_library(${TARGET_NAME} STATIC EXCLUDE_FROM_ALL ${LIBRARY_SRC} ${PUBLIC_HEADERS})
 
 if(COMMAND ie_faster_build)
     ie_faster_build(${TARGET_NAME}
diff --git a/openvino/itt/CMakeLists.txt b/openvino/itt/CMakeLists.txt
index e9f880b8c..c63f4df63 100644
--- a/openvino/itt/CMakeLists.txt
+++ b/openvino/itt/CMakeLists.txt
@@ -6,7 +6,7 @@ set(TARGET_NAME itt)
 
 file(GLOB_RECURSE SOURCES "src/*.cpp" "src/*.hpp")
 
-add_library(${TARGET_NAME} STATIC ${SOURCES})
+add_library(${TARGET_NAME} STATIC EXCLUDE_FROM_ALL ${SOURCES})
 
 add_library(openvino::itt ALIAS ${TARGET_NAME})
 
