diff --git a/cmake/developer_package/add_ie_target.cmake b/cmake/developer_package/add_ie_target.cmake
index b081a6945..5468f09f0 100644
--- a/cmake/developer_package/add_ie_target.cmake
+++ b/cmake/developer_package/add_ie_target.cmake
@@ -91,7 +91,7 @@ function(addIeTarget)
     if (ARG_TYPE STREQUAL EXECUTABLE)
         add_executable(${ARG_NAME} ${all_sources})
     elseif(ARG_TYPE STREQUAL STATIC OR ARG_TYPE STREQUAL SHARED)
-        add_library(${ARG_NAME} ${ARG_TYPE} ${all_sources})
+        add_library(${ARG_NAME} ${ARG_TYPE} EXCLUDE_FROM_ALL ${all_sources})
     else()
         message(SEND_ERROR "Invalid target type ${ARG_TYPE} specified for target name ${ARG_NAME}")
     endif()
diff --git a/inference-engine/CMakeLists.txt b/inference-engine/CMakeLists.txt
index 95c657222..3ab53f854 100644
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
 
 if (ENABLE_PYTHON)
     add_subdirectory(ie_bridges/python)
@@ -138,7 +138,7 @@ endif()
 # Developer package
 #
 
-openvino_developer_export_targets(COMPONENT openvino_common TARGETS format_reader)
+#openvino_developer_export_targets(COMPONENT openvino_common TARGETS format_reader)
 openvino_developer_export_targets(COMPONENT ngraph TARGETS ${NGRAPH_LIBRARIES})
 
 # for Template plugin
@@ -146,7 +146,7 @@ if(NGRAPH_INTERPRETER_ENABLE)
     openvino_developer_export_targets(COMPONENT ngraph TARGETS ngraph_backend interpreter_backend)
 endif()
 
-ie_developer_export()
+#ie_developer_export()
 
 configure_file(
     "${IE_MAIN_SOURCE_DIR}/cmake/templates/InferenceEngineDeveloperPackageConfig.cmake.in"
diff --git a/inference-engine/src/inference_engine/CMakeLists.txt b/inference-engine/src/inference_engine/CMakeLists.txt
index 1ea322763..b5c25837d 100644
--- a/inference-engine/src/inference_engine/CMakeLists.txt
+++ b/inference-engine/src/inference_engine/CMakeLists.txt
@@ -95,7 +95,7 @@ add_cpplint_target(${TARGET_NAME}_plugin_api_cpplint FOR_SOURCES ${plugin_api_sr
 
 # Create object library
 
-add_library(${TARGET_NAME}_obj OBJECT
+add_library(${TARGET_NAME}_obj OBJECT EXCLUDE_FROM_ALL
             ${LIBRARY_SRC}
             ${LIBRARY_HEADERS}
             ${PUBLIC_HEADERS})
@@ -156,7 +156,7 @@ ie_add_api_validator_post_build_step(TARGET ${TARGET_NAME})
 
 # Static library used for unit tests which are always built
 
-add_library(${TARGET_NAME}_s STATIC
+add_library(${TARGET_NAME}_s STATIC EXCLUDE_FROM_ALL
             $<TARGET_OBJECTS:${TARGET_NAME}_legacy_obj>
             $<TARGET_OBJECTS:${TARGET_NAME}_obj>
             ${IE_STATIC_DEPENDENT_FILES})
diff --git a/inference-engine/src/legacy_api/CMakeLists.txt b/inference-engine/src/legacy_api/CMakeLists.txt
index 66498fdbd..4a6c7f619 100644
--- a/inference-engine/src/legacy_api/CMakeLists.txt
+++ b/inference-engine/src/legacy_api/CMakeLists.txt
@@ -26,7 +26,7 @@ endif()
 
 file(TOUCH ${CMAKE_CURRENT_BINARY_DIR}/dummy.cpp)
 
-add_library(${TARGET_NAME}_obj OBJECT
+add_library(${TARGET_NAME}_obj OBJECT EXCLUDE_FROM_ALL
             ${LIBRARY_SRC}
             ${PUBLIC_HEADERS})
 
diff --git a/inference-engine/src/mkldnn_plugin/CMakeLists.txt b/inference-engine/src/mkldnn_plugin/CMakeLists.txt
index 73c7ba9f9..e8cf8d9f9 100644
--- a/inference-engine/src/mkldnn_plugin/CMakeLists.txt
+++ b/inference-engine/src/mkldnn_plugin/CMakeLists.txt
@@ -78,7 +78,7 @@ ie_add_api_validator_post_build_step(TARGET ${TARGET_NAME})
 
 #  add test object library
 
-add_library(${TARGET_NAME}_obj OBJECT ${SOURCES} ${HEADERS})
+add_library(${TARGET_NAME}_obj OBJECT EXCLUDE_FROM_ALL ${SOURCES} ${HEADERS})
 target_link_libraries(${TARGET_NAME}_obj PUBLIC mkldnn)
 
 target_include_directories(${TARGET_NAME}_obj PRIVATE $<TARGET_PROPERTY:inference_engine_preproc_s,INTERFACE_INCLUDE_DIRECTORIES>
diff --git a/inference-engine/src/preprocessing/CMakeLists.txt b/inference-engine/src/preprocessing/CMakeLists.txt
index 973fafcbf..d886d6aa4 100644
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
index 5c31c9a7a..adb170a5f 100644
--- a/inference-engine/src/vpu/common/CMakeLists.txt
+++ b/inference-engine/src/vpu/common/CMakeLists.txt
@@ -5,7 +5,7 @@
 file(GLOB_RECURSE SOURCES *.cpp *.hpp *.h)
 
 function(add_common_target TARGET_NAME STATIC_IE)
-    add_library(${TARGET_NAME} STATIC ${SOURCES})
+    add_library(${TARGET_NAME} STATIC EXCLUDE_FROM_ALL ${SOURCES})
 
     ie_faster_build(${TARGET_NAME}
         UNITY
@@ -62,7 +62,7 @@ add_common_target("vpu_common_lib" FALSE)
 
 # Unit tests support for graph transformer
 if(WIN32)
-    add_common_target("vpu_common_lib_test_static" TRUE)
+    #add_common_target("vpu_common_lib_test_static" TRUE)
 else()
     add_library("vpu_common_lib_test_static" ALIAS "vpu_common_lib")
 endif()
diff --git a/inference-engine/src/vpu/graph_transformer/CMakeLists.txt b/inference-engine/src/vpu/graph_transformer/CMakeLists.txt
index 97bd4caa9..0f49ed144 100644
--- a/inference-engine/src/vpu/graph_transformer/CMakeLists.txt
+++ b/inference-engine/src/vpu/graph_transformer/CMakeLists.txt
@@ -5,7 +5,7 @@
 file(GLOB_RECURSE SOURCES *.cpp *.hpp *.h *.inc)
 
 function(add_graph_transformer_target TARGET_NAME STATIC_IE)
-    add_library(${TARGET_NAME} STATIC ${SOURCES})
+    add_library(${TARGET_NAME} STATIC EXCLUDE_FROM_ALL ${SOURCES})
 
     set_ie_threading_interface_for(${TARGET_NAME})
 
@@ -63,7 +63,7 @@ add_graph_transformer_target("vpu_graph_transformer" FALSE)
 
 # Unit tests support for graph transformer
 if(WIN32)
-    add_graph_transformer_target("vpu_graph_transformer_test_static" TRUE)
+    #add_graph_transformer_target("vpu_graph_transformer_test_static" TRUE)
 else()
     add_library("vpu_graph_transformer_test_static" ALIAS "vpu_graph_transformer")
 endif()
diff --git a/inference-engine/thirdparty/pugixml/CMakeLists.txt b/inference-engine/thirdparty/pugixml/CMakeLists.txt
index 8bcb2801a..5a17fa3f7 100644
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
index 13b31ee17..be613b65f 100644
--- a/ngraph/core/builder/CMakeLists.txt
+++ b/ngraph/core/builder/CMakeLists.txt
@@ -28,7 +28,7 @@ source_group("src" FILES ${LIBRARY_SRC})
 source_group("include" FILES ${PUBLIC_HEADERS})
 
 # Create shared library
-add_library(${TARGET_NAME} STATIC ${LIBRARY_SRC} ${PUBLIC_HEADERS})
+add_library(${TARGET_NAME} STATIC EXCLUDE_FROM_ALL ${LIBRARY_SRC} ${PUBLIC_HEADERS})
 
 if(COMMAND ie_faster_build)
     ie_faster_build(${TARGET_NAME}
diff --git a/ngraph/core/reference/CMakeLists.txt b/ngraph/core/reference/CMakeLists.txt
index 62749a650..dc857f853 100644
--- a/ngraph/core/reference/CMakeLists.txt
+++ b/ngraph/core/reference/CMakeLists.txt
@@ -28,7 +28,7 @@ source_group("src" FILES ${LIBRARY_SRC})
 source_group("include" FILES ${PUBLIC_HEADERS})
 
 # Create shared library
-add_library(${TARGET_NAME} STATIC ${LIBRARY_SRC} ${PUBLIC_HEADERS})
+add_library(${TARGET_NAME} STATIC EXCLUDE_FROM_ALL ${LIBRARY_SRC} ${PUBLIC_HEADERS})
 
 if(COMMAND ie_faster_build)
     ie_faster_build(${TARGET_NAME}
diff --git a/openvino/itt/CMakeLists.txt b/openvino/itt/CMakeLists.txt
index 648ac0a05..4291740d7 100644
--- a/openvino/itt/CMakeLists.txt
+++ b/openvino/itt/CMakeLists.txt
@@ -18,7 +18,7 @@ set(TARGET_NAME itt)
 
 file(GLOB_RECURSE SOURCES "src/*.cpp" "src/*.hpp")
 
-add_library(${TARGET_NAME} STATIC ${SOURCES})
+add_library(${TARGET_NAME} STATIC EXCLUDE_FROM_ALL ${SOURCES})
 
 add_library(openvino::itt ALIAS ${TARGET_NAME})
 
