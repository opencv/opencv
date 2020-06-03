diff --git a/inference-engine/CMakeLists.txt b/inference-engine/CMakeLists.txt
index e7ea6547..7333d19c 100644
--- a/inference-engine/CMakeLists.txt
+++ b/inference-engine/CMakeLists.txt
@@ -72,11 +72,11 @@ if(ENABLE_TESTS)
     add_subdirectory(tests)
 endif()
 
-add_subdirectory(tools)
+#add_subdirectory(tools)
 
 # gflags and format_reader targets are kept inside of samples directory and
 # they must be built even if samples build is disabled (required for tests and tools).
-add_subdirectory(samples)
+#add_subdirectory(samples)
 
 file(GLOB_RECURSE SAMPLES_SOURCES samples/*.cpp samples/*.hpp samples/*.h)
 add_cpplint_target(sample_cpplint
@@ -154,10 +154,10 @@ endif()
 # Developer package
 #
 
-ie_developer_export_targets(format_reader)
+#ie_developer_export_targets(format_reader)
 ie_developer_export_targets(${NGRAPH_LIBRARIES})
 
-ie_developer_export()
+#ie_developer_export()
 
 configure_file(
     "${IE_MAIN_SOURCE_DIR}/cmake/developer_package_config.cmake.in"
diff --git a/inference-engine/src/legacy_api/CMakeLists.txt b/inference-engine/src/legacy_api/CMakeLists.txt
index a03a5f23..63d4f687 100644
--- a/inference-engine/src/legacy_api/CMakeLists.txt
+++ b/inference-engine/src/legacy_api/CMakeLists.txt
@@ -22,7 +22,7 @@ source_group("include" FILES ${PUBLIC_HEADERS})
 
 # Create object library
 
-add_library(${TARGET_NAME}_obj OBJECT
+add_library(${TARGET_NAME}_obj OBJECT EXCLUDE_FROM_ALL
             ${LIBRARY_SRC}
             ${NN_BUILDER_LIBRARY_SRC}
             ${PUBLIC_HEADERS})
diff --git a/inference-engine/src/inference_engine/CMakeLists.txt b/inference-engine/src/inference_engine/CMakeLists.txt
index 2071c126..015d8ff8 100644
--- a/inference-engine/src/inference_engine/CMakeLists.txt
+++ b/inference-engine/src/inference_engine/CMakeLists.txt
@@ -98,7 +98,7 @@ add_clang_format_target(${TARGET_NAME}_plugin_api_clang_format FOR_SOURCES ${plu
 
 # Create common base object library
 
-add_library(${TARGET_NAME}_common_obj OBJECT
+add_library(${TARGET_NAME}_common_obj OBJECT EXCLUDE_FROM_ALL
             ${IE_BASE_SOURCE_FILES})
 
 target_compile_definitions(${TARGET_NAME}_common_obj PRIVATE IMPLEMENT_INFERENCE_ENGINE_API)
@@ -110,7 +110,7 @@ target_include_directories(${TARGET_NAME}_common_obj SYSTEM PRIVATE
 
 # Create object library
 
-add_library(${TARGET_NAME}_obj OBJECT
+add_library(${TARGET_NAME}_obj OBJECT EXCLUDE_FROM_ALL
             ${LIBRARY_SRC}
             ${LIBRARY_HEADERS}
             ${PUBLIC_HEADERS})
@@ -200,7 +200,7 @@ add_clang_format_target(${TARGET_NAME}_nn_builder_clang_format FOR_TARGETS ${TAR
 
 # Static library used for unit tests which are always built
 
-add_library(${TARGET_NAME}_s STATIC
+add_library(${TARGET_NAME}_s STATIC EXCLUDE_FROM_ALL
             $<TARGET_OBJECTS:${TARGET_NAME}_obj>
             $<TARGET_OBJECTS:${TARGET_NAME}_common_obj>
             $<TARGET_OBJECTS:${TARGET_NAME}_legacy_obj>
diff --git a/inference-engine/src/mkldnn_plugin/CMakeLists.txt b/inference-engine/src/mkldnn_plugin/CMakeLists.txt
index 52183e86..4fd6d7d4 100644
--- a/inference-engine/src/mkldnn_plugin/CMakeLists.txt
+++ b/inference-engine/src/mkldnn_plugin/CMakeLists.txt
@@ -163,9 +163,9 @@ add_library(mkldnn_plugin_layers_no_opt OBJECT ${CROSS_COMPILED_SOURCES})
 set_ie_threading_interface_for(mkldnn_plugin_layers_no_opt)
 target_compile_definitions(mkldnn_plugin_layers_no_opt PRIVATE "IMPLEMENT_INFERENCE_ENGINE_PLUGIN")
 
-add_library(mkldnn_plugin_layers_no_opt_s OBJECT ${CROSS_COMPILED_SOURCES})
-set_ie_threading_interface_for(mkldnn_plugin_layers_no_opt_s)
-target_compile_definitions(mkldnn_plugin_layers_no_opt_s PRIVATE "USE_STATIC_IE;IMPLEMENT_INFERENCE_ENGINE_PLUGIN")
+#add_library(mkldnn_plugin_layers_no_opt_s OBJECT ${CROSS_COMPILED_SOURCES})
+#set_ie_threading_interface_for(mkldnn_plugin_layers_no_opt_s)
+#target_compile_definitions(mkldnn_plugin_layers_no_opt_s PRIVATE "USE_STATIC_IE;IMPLEMENT_INFERENCE_ENGINE_PLUGIN")
 
 set(object_libraries mkldnn_plugin_layers_no_opt)
 set(mkldnn_plugin_object_libraries mkldnn_plugin_layers_no_opt_s)
@@ -190,7 +190,7 @@ if (ENABLE_SSE42)
     endfunction()
 
     mkldnn_create_sse42_layers(mkldnn_plugin_layers_sse42)
-    mkldnn_create_sse42_layers(mkldnn_plugin_layers_sse42_s)
+    #mkldnn_create_sse42_layers(mkldnn_plugin_layers_sse42_s)
 
     list(APPEND object_libraries mkldnn_plugin_layers_sse42)
     list(APPEND mkldnn_plugin_object_libraries mkldnn_plugin_layers_sse42_s)
@@ -216,7 +216,7 @@ if (ENABLE_AVX2)
     endfunction()
 
     mkldnn_create_avx2_layers(mkldnn_plugin_layers_avx2)
-    mkldnn_create_avx2_layers(mkldnn_plugin_layers_avx2_s)
+    #mkldnn_create_avx2_layers(mkldnn_plugin_layers_avx2_s)
 
     list(APPEND object_libraries mkldnn_plugin_layers_avx2)
     list(APPEND mkldnn_plugin_object_libraries mkldnn_plugin_layers_avx2_s)
@@ -242,7 +242,7 @@ if (ENABLE_AVX512F)
     endfunction()
 
     mkldnn_create_avx512f_layers(mkldnn_plugin_layers_avx512)
-    mkldnn_create_avx512f_layers(mkldnn_plugin_layers_avx512_s)
+    #mkldnn_create_avx512f_layers(mkldnn_plugin_layers_avx512_s)
 
     list(APPEND object_libraries mkldnn_plugin_layers_avx512)
     list(APPEND mkldnn_plugin_object_libraries mkldnn_plugin_layers_avx512_s)
@@ -264,7 +264,7 @@ target_link_libraries(${TARGET_NAME} PRIVATE inference_engine inference_engine_l
 
 #  add test object library
 
-add_library(${TARGET_NAME}_obj OBJECT ${SOURCES} ${HEADERS})
+add_library(${TARGET_NAME}_obj OBJECT EXCLUDE_FROM_ALL ${SOURCES} ${HEADERS})
 
 target_include_directories(${TARGET_NAME}_obj PRIVATE $<TARGET_PROPERTY:inference_engine_preproc_s,INTERFACE_INCLUDE_DIRECTORIES>
                                                       $<TARGET_PROPERTY:inference_engine_lp_transformations,INTERFACE_INCLUDE_DIRECTORIES>
diff --git a/inference-engine/src/preprocessing/CMakeLists.txt b/inference-engine/src/preprocessing/CMakeLists.txt
index c3ad1e58..b5913840 100644
--- a/inference-engine/src/preprocessing/CMakeLists.txt
+++ b/inference-engine/src/preprocessing/CMakeLists.txt
@@ -124,7 +124,7 @@ endif()
 
 # Create object library
 
-add_library(${TARGET_NAME}_obj OBJECT
+add_library(${TARGET_NAME}_obj OBJECT EXCLUDE_FROM_ALL
             ${LIBRARY_SRC}
             ${LIBRARY_HEADERS})
 
@@ -167,7 +167,7 @@ endif()
 
 # Static library used for unit tests which are always built
 
-add_library(${TARGET_NAME}_s STATIC
+add_library(${TARGET_NAME}_s STATIC EXCLUDE_FROM_ALL
             $<TARGET_OBJECTS:${TARGET_NAME}_obj>)
 
 set_ie_threading_interface_for(${TARGET_NAME}_s)
diff --git a/inference-engine/src/vpu/common/CMakeLists.txt b/inference-engine/src/vpu/common/CMakeLists.txt
index 65215299..03ba4a4c 100644
--- a/inference-engine/src/vpu/common/CMakeLists.txt
+++ b/inference-engine/src/vpu/common/CMakeLists.txt
@@ -53,7 +53,7 @@ add_common_target("vpu_common_lib" FALSE)
 
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
index ebf32c71..ad8cb435 100644
--- a/inference-engine/thirdparty/CMakeLists.txt
+++ b/inference-engine/thirdparty/CMakeLists.txt
@@ -36,7 +36,7 @@ function(build_with_lto)
     endif()
 
     add_subdirectory(pugixml)
-    add_subdirectory(stb_lib)
+    #add_subdirectory(stb_lib)
     add_subdirectory(ade)
     add_subdirectory(fluid/modules/gapi)
 
diff --git a/inference-engine/thirdparty/pugixml/CMakeLists.txt b/inference-engine/thirdparty/pugixml/CMakeLists.txt
index 8bcb2801..f7e031c0 100644
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
