diff --git a/inference-engine/CMakeLists.txt b/inference-engine/CMakeLists.txt
index 7f45ab02..a7bac7e9 100644
--- a/inference-engine/CMakeLists.txt
+++ b/inference-engine/CMakeLists.txt
@@ -70,7 +70,7 @@ if(ENABLE_TESTS)
     add_subdirectory(tests)
 endif()
 
-add_subdirectory(tools)
+#add_subdirectory(tools)
 
 function(ie_build_samples)
     # samples should be build with the same flags as from OpenVINO package,
@@ -89,7 +89,7 @@ endfunction()
 
 # gflags and format_reader targets are kept inside of samples directory and
 # they must be built even if samples build is disabled (required for tests and tools).
-ie_build_samples()
+#ie_build_samples()
 
 file(GLOB_RECURSE SAMPLES_SOURCES samples/*.cpp samples/*.hpp samples/*.h)
 add_cpplint_target(sample_cpplint
@@ -180,7 +180,7 @@ endif()
 # Developer package
 #
 
-ie_developer_export_targets(format_reader)
+#ie_developer_export_targets(format_reader)
 ie_developer_export_targets(${NGRAPH_LIBRARIES})
 
 # for Template plugin
@@ -188,7 +188,7 @@ if(NGRAPH_INTERPRETER_ENABLE)
     ie_developer_export_targets(ngraph_backend interpreter_backend)
 endif()
 
-ie_developer_export()
+#ie_developer_export()
 
 configure_file(
     "${IE_MAIN_SOURCE_DIR}/cmake/developer_package_config.cmake.in"
diff --git a/inference-engine/src/inference_engine/CMakeLists.txt b/inference-engine/src/inference_engine/CMakeLists.txt
index 9ab88898..8badb591 100644
--- a/inference-engine/src/inference_engine/CMakeLists.txt
+++ b/inference-engine/src/inference_engine/CMakeLists.txt
@@ -118,7 +118,7 @@ add_cpplint_target(${TARGET_NAME}_plugin_api_cpplint FOR_SOURCES ${plugin_api_sr
 
 # Create common base object library
 
-add_library(${TARGET_NAME}_common_obj OBJECT
+add_library(${TARGET_NAME}_common_obj OBJECT EXCLUDE_FROM_ALL
             ${IE_BASE_SOURCE_FILES})
 
 target_compile_definitions(${TARGET_NAME}_common_obj PRIVATE IMPLEMENT_INFERENCE_ENGINE_API)
@@ -132,7 +132,7 @@ target_include_directories(${TARGET_NAME}_common_obj SYSTEM PRIVATE
 
 # Create object library
 
-add_library(${TARGET_NAME}_obj OBJECT
+add_library(${TARGET_NAME}_obj OBJECT EXCLUDE_FROM_ALL
             ${LIBRARY_SRC}
             ${LIBRARY_HEADERS}
             ${PUBLIC_HEADERS})
@@ -183,7 +183,7 @@ ie_register_plugins(MAIN_TARGET ${TARGET_NAME}
 
 # Static library used for unit tests which are always built
 
-add_library(${TARGET_NAME}_s STATIC
+add_library(${TARGET_NAME}_s STATIC EXCLUDE_FROM_ALL
             $<TARGET_OBJECTS:${TARGET_NAME}_obj>
             $<TARGET_OBJECTS:${TARGET_NAME}_common_obj>
             $<TARGET_OBJECTS:${TARGET_NAME}_legacy_obj>
diff --git a/inference-engine/src/legacy_api/CMakeLists.txt b/inference-engine/src/legacy_api/CMakeLists.txt
index ed87a073..b30e6671 100644
--- a/inference-engine/src/legacy_api/CMakeLists.txt
+++ b/inference-engine/src/legacy_api/CMakeLists.txt
@@ -26,7 +26,7 @@ endif()
 
 # Create object library
 
-add_library(${TARGET_NAME}_obj OBJECT
+add_library(${TARGET_NAME}_obj OBJECT EXCLUDE_FROM_ALL
             ${LIBRARY_SRC}
             ${PUBLIC_HEADERS})
 
diff --git a/inference-engine/src/mkldnn_plugin/CMakeLists.txt b/inference-engine/src/mkldnn_plugin/CMakeLists.txt
index 166818cd..6c1e8e36 100644
--- a/inference-engine/src/mkldnn_plugin/CMakeLists.txt
+++ b/inference-engine/src/mkldnn_plugin/CMakeLists.txt
@@ -193,7 +193,7 @@ cross_compiled_file(${TARGET_NAME}
 
 #  add test object library
 
-add_library(${TARGET_NAME}_obj OBJECT ${SOURCES} ${HEADERS})
+add_library(${TARGET_NAME}_obj OBJECT EXCLUDE_FROM_ALL ${SOURCES} ${HEADERS})
 
 target_include_directories(${TARGET_NAME}_obj PRIVATE $<TARGET_PROPERTY:inference_engine_preproc_s,INTERFACE_INCLUDE_DIRECTORIES>
                                                       $<TARGET_PROPERTY:inference_engine_lp_transformations,INTERFACE_INCLUDE_DIRECTORIES>
diff --git a/inference-engine/src/preprocessing/CMakeLists.txt b/inference-engine/src/preprocessing/CMakeLists.txt
index f4fed72a..9cedd6b5 100644
--- a/inference-engine/src/preprocessing/CMakeLists.txt
+++ b/inference-engine/src/preprocessing/CMakeLists.txt
@@ -124,7 +124,7 @@ endif()
 
 # Create object library
 
-add_library(${TARGET_NAME}_obj OBJECT
+add_library(${TARGET_NAME}_obj OBJECT EXCLUDE_FROM_ALL
             ${LIBRARY_SRC}
             ${LIBRARY_HEADERS})
 
@@ -175,7 +175,7 @@ add_cpplint_target(${TARGET_NAME}_cpplint FOR_TARGETS ${TARGET_NAME}
 
 # Static library used for unit tests which are always built
 
-add_library(${TARGET_NAME}_s STATIC
+add_library(${TARGET_NAME}_s STATIC EXCLUDE_FROM_ALL
             $<TARGET_OBJECTS:${TARGET_NAME}_obj>)
 
 set_ie_threading_interface_for(${TARGET_NAME}_s)
diff --git a/inference-engine/src/vpu/common/CMakeLists.txt b/inference-engine/src/vpu/common/CMakeLists.txt
index b291d5b4..74ab8287 100644
--- a/inference-engine/src/vpu/common/CMakeLists.txt
+++ b/inference-engine/src/vpu/common/CMakeLists.txt
@@ -57,7 +57,7 @@ add_common_target("vpu_common_lib" FALSE)
 
 # Unit tests support for graph transformer
 if(WIN32)
-    add_common_target("vpu_common_lib_test_static" TRUE)
+    #add_common_target("vpu_common_lib_test_static" TRUE)
 else()
     add_library("vpu_common_lib_test_static" ALIAS "vpu_common_lib")
 endif()
diff --git a/inference-engine/src/vpu/graph_transformer/CMakeLists.txt b/inference-engine/src/vpu/graph_transformer/CMakeLists.txt
index a4543745..807b8e36 100644
--- a/inference-engine/src/vpu/graph_transformer/CMakeLists.txt
+++ b/inference-engine/src/vpu/graph_transformer/CMakeLists.txt
@@ -65,7 +65,7 @@ add_graph_transformer_target("vpu_graph_transformer" FALSE)
 
 # Unit tests support for graph transformer
 if(WIN32)
-    add_graph_transformer_target("vpu_graph_transformer_test_static" TRUE)
+    #add_graph_transformer_target("vpu_graph_transformer_test_static" TRUE)
 else()
     add_library("vpu_graph_transformer_test_static" ALIAS "vpu_graph_transformer")
 endif()
diff --git a/inference-engine/thirdparty/CMakeLists.txt b/inference-engine/thirdparty/CMakeLists.txt
index a2550bfa..10ce316f 100644
--- a/inference-engine/thirdparty/CMakeLists.txt
+++ b/inference-engine/thirdparty/CMakeLists.txt
@@ -56,13 +56,13 @@ function(build_with_lto)
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
