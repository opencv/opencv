diff --git a/inference-engine/CMakeLists.txt b/inference-engine/CMakeLists.txt
index d5feedb..1b7aa7e 100644
--- a/inference-engine/CMakeLists.txt
+++ b/inference-engine/CMakeLists.txt
@@ -59,11 +59,11 @@ if(ENABLE_TESTS)
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
@@ -134,7 +134,7 @@ install(DIRECTORY ${ie_python_api_SOURCE_DIR}/sample/
 add_custom_target(ie_dev_targets ALL DEPENDS inference_engine HeteroPlugin)
 
 # Developer package
-ie_developer_export_targets(format_reader)
+#ie_developer_export_targets(format_reader)
 
 if (ENABLE_NGRAPH)
     ie_developer_export_targets(${NGRAPH_LIBRARIES})
diff --git a/inference-engine/src/inference_engine/CMakeLists.txt b/inference-engine/src/inference_engine/CMakeLists.txt
index 54e264c..c0b7495 100644
--- a/inference-engine/src/inference_engine/CMakeLists.txt
+++ b/inference-engine/src/inference_engine/CMakeLists.txt
@@ -228,7 +228,7 @@ target_include_directories(${TARGET_NAME}_nn_builder PRIVATE "${CMAKE_CURRENT_SO
 
 # Static library used for unit tests which are always built
 
-add_library(${TARGET_NAME}_s STATIC
+add_library(${TARGET_NAME}_s STATIC EXCLUDE_FROM_ALL
             $<TARGET_OBJECTS:${TARGET_NAME}_obj>
             ${NN_BUILDER_LIBRARY_SRC})
 
diff --git a/inference-engine/src/mkldnn_plugin/CMakeLists.txt b/inference-engine/src/mkldnn_plugin/CMakeLists.txt
index cd727fd..2f09b44 100644
--- a/inference-engine/src/mkldnn_plugin/CMakeLists.txt
+++ b/inference-engine/src/mkldnn_plugin/CMakeLists.txt
@@ -184,9 +184,9 @@ endif()
 add_library(mkldnn_plugin_layers_no_opt OBJECT ${CROSS_COMPILED_SOURCES})
 set_ie_threading_interface_for(mkldnn_plugin_layers_no_opt)
 
-add_library(mkldnn_plugin_layers_no_opt_s OBJECT ${CROSS_COMPILED_SOURCES})
-set_ie_threading_interface_for(mkldnn_plugin_layers_no_opt_s)
-target_compile_definitions(mkldnn_plugin_layers_no_opt_s PRIVATE USE_STATIC_IE)
+#add_library(mkldnn_plugin_layers_no_opt_s OBJECT ${CROSS_COMPILED_SOURCES})
+#set_ie_threading_interface_for(mkldnn_plugin_layers_no_opt_s)
+#target_compile_definitions(mkldnn_plugin_layers_no_opt_s PRIVATE USE_STATIC_IE)
 
 set(object_libraries mkldnn_plugin_layers_no_opt)
 set(mkldnn_plugin_object_libraries mkldnn_plugin_layers_no_opt_s)
@@ -220,7 +220,7 @@ if (ENABLE_SSE42)
     endfunction()
 
     mkldnn_create_sse42_layers(mkldnn_plugin_layers_sse42)
-    mkldnn_create_sse42_layers(mkldnn_plugin_layers_sse42_s)
+    #mkldnn_create_sse42_layers(mkldnn_plugin_layers_sse42_s)
 
     list(APPEND object_libraries mkldnn_plugin_layers_sse42)
     list(APPEND mkldnn_plugin_object_libraries mkldnn_plugin_layers_sse42_s)
@@ -259,7 +259,7 @@ if (ENABLE_AVX2)
     endfunction()
 
     mkldnn_create_avx2_layers(mkldnn_plugin_layers_avx2)
-    mkldnn_create_avx2_layers(mkldnn_plugin_layers_avx2_s)
+    #mkldnn_create_avx2_layers(mkldnn_plugin_layers_avx2_s)
 
     list(APPEND object_libraries mkldnn_plugin_layers_avx2)
     list(APPEND mkldnn_plugin_object_libraries mkldnn_plugin_layers_avx2_s)
@@ -297,7 +297,7 @@ if (ENABLE_AVX512F)
     endfunction()
 
     mkldnn_create_avx512f_layers(mkldnn_plugin_layers_avx512)
-    mkldnn_create_avx512f_layers(mkldnn_plugin_layers_avx512_s)
+    #mkldnn_create_avx512f_layers(mkldnn_plugin_layers_avx512_s)
 
     list(APPEND object_libraries mkldnn_plugin_layers_avx512)
     list(APPEND mkldnn_plugin_object_libraries mkldnn_plugin_layers_avx512_s)
@@ -317,7 +317,7 @@ target_link_libraries(${TARGET_NAME} PRIVATE inference_engine ${INTEL_ITT_LIBS}
 
 #  add test object library
 
-add_library(${TARGET_NAME}_obj OBJECT ${SOURCES} ${HEADERS})
+add_library(${TARGET_NAME}_obj OBJECT EXCLUDE_FROM_ALL ${SOURCES} ${HEADERS})
 
 target_include_directories(${TARGET_NAME}_obj PRIVATE $<TARGET_PROPERTY:inference_engine_preproc_s,INTERFACE_INCLUDE_DIRECTORIES>)
 
diff --git a/inference-engine/src/preprocessing/CMakeLists.txt b/inference-engine/src/preprocessing/CMakeLists.txt
index 41f14a9..0e1b4f6 100644
--- a/inference-engine/src/preprocessing/CMakeLists.txt
+++ b/inference-engine/src/preprocessing/CMakeLists.txt
@@ -81,7 +81,7 @@ endif()
 
 # Static library used for unit tests which are always built
 
-add_library(${TARGET_NAME}_s STATIC
+add_library(${TARGET_NAME}_s STATIC EXCLUDE_FROM_ALL
             $<TARGET_OBJECTS:${TARGET_NAME}_obj>)
 
 set_ie_threading_interface_for(${TARGET_NAME}_s)
diff --git a/inference-engine/src/vpu/common/CMakeLists.txt b/inference-engine/src/vpu/common/CMakeLists.txt
index 8995390..8413faf 100644
--- a/inference-engine/src/vpu/common/CMakeLists.txt
+++ b/inference-engine/src/vpu/common/CMakeLists.txt
@@ -49,7 +49,7 @@ add_common_target("vpu_common_lib" FALSE)
 
 # Unit tests support for graph transformer
 if(WIN32)
-    add_common_target("vpu_common_lib_test_static" TRUE)
+    #add_common_target("vpu_common_lib_test_static" TRUE)
 else()
     add_library("vpu_common_lib_test_static" ALIAS "vpu_common_lib")
 endif()
diff --git a/inference-engine/src/vpu/graph_transformer/CMakeLists.txt b/inference-engine/src/vpu/graph_transformer/CMakeLists.txt
index e77296e..333f560 100644
--- a/inference-engine/src/vpu/graph_transformer/CMakeLists.txt
+++ b/inference-engine/src/vpu/graph_transformer/CMakeLists.txt
@@ -60,7 +60,7 @@ add_graph_transformer_target("vpu_graph_transformer" FALSE)
 
 # Unit tests support for graph transformer
 if(WIN32)
-    add_graph_transformer_target("vpu_graph_transformer_test_static" TRUE)
+    #add_graph_transformer_target("vpu_graph_transformer_test_static" TRUE)
 else()
     add_library("vpu_graph_transformer_test_static" ALIAS "vpu_graph_transformer")
 endif()
diff --git a/inference-engine/thirdparty/CMakeLists.txt b/inference-engine/thirdparty/CMakeLists.txt
index ec22761..8bb3325 100644
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
index 8bcb280..5a17fa3 100644
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
