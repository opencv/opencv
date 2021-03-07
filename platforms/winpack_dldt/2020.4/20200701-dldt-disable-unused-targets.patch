diff --git a/inference-engine/CMakeLists.txt b/inference-engine/CMakeLists.txt
index 39ff413b..df4e89c7 100644
--- a/inference-engine/CMakeLists.txt
+++ b/inference-engine/CMakeLists.txt
@@ -66,7 +66,7 @@ if(ENABLE_TESTS)
     add_subdirectory(tests)
 endif()
 
-add_subdirectory(tools)
+#add_subdirectory(tools)
 
 function(ie_build_samples)
     # samples should be build with the same flags as from OpenVINO package,
@@ -85,7 +85,7 @@ endfunction()
 
 # gflags and format_reader targets are kept inside of samples directory and
 # they must be built even if samples build is disabled (required for tests and tools).
-ie_build_samples()
+#ie_build_samples()
 
 file(GLOB_RECURSE SAMPLES_SOURCES samples/*.cpp samples/*.hpp samples/*.h)
 add_cpplint_target(sample_cpplint
@@ -174,10 +174,10 @@ endif()
 # Developer package
 #
 
-ie_developer_export_targets(format_reader)
+#ie_developer_export_targets(format_reader)
 ie_developer_export_targets(${NGRAPH_LIBRARIES})
 
-ie_developer_export()
+#ie_developer_export()
 
 configure_file(
     "${IE_MAIN_SOURCE_DIR}/cmake/developer_package_config.cmake.in"
diff --git a/inference-engine/src/inference_engine/CMakeLists.txt b/inference-engine/src/inference_engine/CMakeLists.txt
index 4ae0d560..e37acbe0 100644
--- a/inference-engine/src/inference_engine/CMakeLists.txt
+++ b/inference-engine/src/inference_engine/CMakeLists.txt
@@ -99,7 +99,7 @@ add_cpplint_target(${TARGET_NAME}_plugin_api_cpplint FOR_SOURCES ${plugin_api_sr
 
 # Create common base object library
 
-add_library(${TARGET_NAME}_common_obj OBJECT
+add_library(${TARGET_NAME}_common_obj OBJECT EXCLUDE_FROM_ALL
             ${IE_BASE_SOURCE_FILES})
 
 target_compile_definitions(${TARGET_NAME}_common_obj PRIVATE IMPLEMENT_INFERENCE_ENGINE_API)
@@ -112,7 +112,7 @@ target_include_directories(${TARGET_NAME}_common_obj SYSTEM PRIVATE
 
 # Create object library
 
-add_library(${TARGET_NAME}_obj OBJECT
+add_library(${TARGET_NAME}_obj OBJECT EXCLUDE_FROM_ALL
             ${LIBRARY_SRC}
             ${LIBRARY_HEADERS}
             ${PUBLIC_HEADERS})
@@ -177,7 +177,7 @@ ie_register_plugins(MAIN_TARGET ${TARGET_NAME}
 
 # Static library used for unit tests which are always built
 
-add_library(${TARGET_NAME}_s STATIC
+add_library(${TARGET_NAME}_s STATIC EXCLUDE_FROM_ALL
             $<TARGET_OBJECTS:${TARGET_NAME}_obj>
             $<TARGET_OBJECTS:${TARGET_NAME}_common_obj>
             $<TARGET_OBJECTS:${TARGET_NAME}_legacy_obj>
diff --git a/inference-engine/src/legacy_api/CMakeLists.txt b/inference-engine/src/legacy_api/CMakeLists.txt
index 85524310..ed27e058 100644
--- a/inference-engine/src/legacy_api/CMakeLists.txt
+++ b/inference-engine/src/legacy_api/CMakeLists.txt
@@ -21,7 +21,7 @@ source_group("include" FILES ${PUBLIC_HEADERS})
 
 # Create object library
 
-add_library(${TARGET_NAME}_obj OBJECT
+add_library(${TARGET_NAME}_obj OBJECT EXCLUDE_FROM_ALL
             ${LIBRARY_SRC}
             ${PUBLIC_HEADERS})
 
diff --git a/inference-engine/src/mkldnn_plugin/CMakeLists.txt b/inference-engine/src/mkldnn_plugin/CMakeLists.txt
index 297783da..06da35c3 100644
--- a/inference-engine/src/mkldnn_plugin/CMakeLists.txt
+++ b/inference-engine/src/mkldnn_plugin/CMakeLists.txt
@@ -192,7 +192,7 @@ cross_compiled_file(${TARGET_NAME}
 
 #  add test object library
 
-add_library(${TARGET_NAME}_obj OBJECT ${SOURCES} ${HEADERS})
+add_library(${TARGET_NAME}_obj OBJECT EXCLUDE_FROM_ALL ${SOURCES} ${HEADERS})
 
 target_include_directories(${TARGET_NAME}_obj PRIVATE $<TARGET_PROPERTY:inference_engine_preproc_s,INTERFACE_INCLUDE_DIRECTORIES>
                                                       $<TARGET_PROPERTY:inference_engine_lp_transformations,INTERFACE_INCLUDE_DIRECTORIES>
diff --git a/inference-engine/src/preprocessing/CMakeLists.txt b/inference-engine/src/preprocessing/CMakeLists.txt
index adc52f06..6b7d0ffe 100644
--- a/inference-engine/src/preprocessing/CMakeLists.txt
+++ b/inference-engine/src/preprocessing/CMakeLists.txt
@@ -124,7 +124,7 @@ endif()
 
 # Create object library
 
-add_library(${TARGET_NAME}_obj OBJECT
+add_library(${TARGET_NAME}_obj OBJECT EXCLUDE_FROM_ALL
             ${LIBRARY_SRC}
             ${LIBRARY_HEADERS})
 
@@ -183,7 +183,7 @@ add_cpplint_target(${TARGET_NAME}_cpplint FOR_TARGETS ${TARGET_NAME}
 
 # Static library used for unit tests which are always built
 
-add_library(${TARGET_NAME}_s STATIC
+add_library(${TARGET_NAME}_s STATIC EXCLUDE_FROM_ALL
             $<TARGET_OBJECTS:${TARGET_NAME}_obj>)
 
 set_ie_threading_interface_for(${TARGET_NAME}_s)
diff --git a/inference-engine/src/vpu/common/CMakeLists.txt b/inference-engine/src/vpu/common/CMakeLists.txt
index 43e9308f..2e40dd31 100644
--- a/inference-engine/src/vpu/common/CMakeLists.txt
+++ b/inference-engine/src/vpu/common/CMakeLists.txt
@@ -55,7 +55,7 @@ add_common_target("vpu_common_lib" FALSE)
 
 # Unit tests support for graph transformer
 if(WIN32)
-    add_common_target("vpu_common_lib_test_static" TRUE)
+    #add_common_target("vpu_common_lib_test_static" TRUE)
 else()
     add_library("vpu_common_lib_test_static" ALIAS "vpu_common_lib")
 endif()
diff --git a/inference-engine/src/vpu/graph_transformer/CMakeLists.txt b/inference-engine/src/vpu/graph_transformer/CMakeLists.txt
index 982d3c7f..15fcf3e8 100644
--- a/inference-engine/src/vpu/graph_transformer/CMakeLists.txt
+++ b/inference-engine/src/vpu/graph_transformer/CMakeLists.txt
@@ -64,7 +64,7 @@ add_graph_transformer_target("vpu_graph_transformer" FALSE)
 
 # Unit tests support for graph transformer
 if(WIN32)
-    add_graph_transformer_target("vpu_graph_transformer_test_static" TRUE)
+    #add_graph_transformer_target("vpu_graph_transformer_test_static" TRUE)
 else()
     add_library("vpu_graph_transformer_test_static" ALIAS "vpu_graph_transformer")
 endif()
diff --git a/inference-engine/thirdparty/CMakeLists.txt b/inference-engine/thirdparty/CMakeLists.txt
index f94453e0..c80e75c5 100644
--- a/inference-engine/thirdparty/CMakeLists.txt
+++ b/inference-engine/thirdparty/CMakeLists.txt
@@ -43,13 +43,13 @@ function(build_with_lto)
     endfunction()
 
     ie_build_pugixml()
-    add_subdirectory(stb_lib)
+    #add_subdirectory(stb_lib)
     add_subdirectory(ade)
     add_subdirectory(fluid/modules/gapi)
 
     target_include_directories(pugixml INTERFACE "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/pugixml/src>")
 
-    set_target_properties(pugixml ade fluid stb_image
+    set_target_properties(pugixml ade fluid
                           PROPERTIES FOLDER thirdparty)
 
     # developer package
diff --git a/inference-engine/thirdparty/pugixml/CMakeLists.txt b/inference-engine/thirdparty/pugixml/CMakeLists.txt
index 8bcb2801..380fb468 100644
--- a/inference-engine/thirdparty/pugixml/CMakeLists.txt
+++ b/inference-engine/thirdparty/pugixml/CMakeLists.txt
@@ -41,7 +41,7 @@ if(BUILD_SHARED_LIBS)
 else()
 	add_library(pugixml STATIC ${SOURCES})
 	if (MSVC)
-		add_library(pugixml_mt STATIC ${SOURCES})
+               #add_library(pugixml_mt STATIC ${SOURCES})
 		#if (WIN32)
 		#	set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
 		#	set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
