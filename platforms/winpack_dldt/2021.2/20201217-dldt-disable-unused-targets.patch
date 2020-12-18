diff --git a/inference-engine/CMakeLists.txt b/inference-engine/CMakeLists.txt
index a3e4f74c..190305a6 100644
--- a/inference-engine/CMakeLists.txt
+++ b/inference-engine/CMakeLists.txt
@@ -69,7 +69,7 @@ if(ENABLE_TESTS)
     add_subdirectory(tests)
 endif()
 
-add_subdirectory(tools)
+#add_subdirectory(tools)
 
 function(ie_build_samples)
     # samples should be build with the same flags as from OpenVINO package,
@@ -88,7 +88,7 @@ endfunction()
 
 # gflags and format_reader targets are kept inside of samples directory and
 # they must be built even if samples build is disabled (required for tests and tools).
-ie_build_samples()
+#ie_build_samples()
 
 file(GLOB_RECURSE SAMPLES_SOURCES samples/*.cpp samples/*.hpp samples/*.h)
 add_cpplint_target(sample_cpplint
@@ -179,7 +179,7 @@ endif()
 # Developer package
 #
 
-ie_developer_export_targets(format_reader)
+#ie_developer_export_targets(format_reader)
 ie_developer_export_targets(${NGRAPH_LIBRARIES})
 
 # for Template plugin
@@ -187,7 +187,7 @@ if(NGRAPH_INTERPRETER_ENABLE)
     ie_developer_export_targets(ngraph_backend interpreter_backend)
 endif()
 
-ie_developer_export()
+#ie_developer_export()
 
 configure_file(
     "${IE_MAIN_SOURCE_DIR}/cmake/developer_package_config.cmake.in"
diff --git a/inference-engine/cmake/add_ie_target.cmake b/inference-engine/cmake/add_ie_target.cmake
index 35b96542..48dacfb3 100644
--- a/inference-engine/cmake/add_ie_target.cmake
+++ b/inference-engine/cmake/add_ie_target.cmake
@@ -91,7 +91,7 @@ function(addIeTarget)
     if (ARG_TYPE STREQUAL EXECUTABLE)
         add_executable(${ARG_NAME} ${all_sources})
     elseif(ARG_TYPE STREQUAL STATIC OR ARG_TYPE STREQUAL SHARED)
-        add_library(${ARG_NAME} ${ARG_TYPE} ${all_sources})
+        add_library(${ARG_NAME} ${ARG_TYPE} EXCLUDE_FROM_ALL ${all_sources})
     else()
         message(SEND_ERROR "Invalid target type ${ARG_TYPE} specified for target name ${ARG_NAME}")
     endif()
diff --git a/inference-engine/src/inference_engine/CMakeLists.txt b/inference-engine/src/inference_engine/CMakeLists.txt
index f012a038..5204fb6a 100644
--- a/inference-engine/src/inference_engine/CMakeLists.txt
+++ b/inference-engine/src/inference_engine/CMakeLists.txt
@@ -99,7 +99,7 @@ add_cpplint_target(${TARGET_NAME}_plugin_api_cpplint FOR_SOURCES ${plugin_api_sr
 
 # Create object library
 
-add_library(${TARGET_NAME}_obj OBJECT
+add_library(${TARGET_NAME}_obj OBJECT EXCLUDE_FROM_ALL
             ${LIBRARY_SRC}
             ${LIBRARY_HEADERS}
             ${PUBLIC_HEADERS})
@@ -162,7 +162,7 @@ ie_add_api_validator_post_build_step(TARGET ${TARGET_NAME})
 
 # Static library used for unit tests which are always built
 
-add_library(${TARGET_NAME}_s STATIC
+add_library(${TARGET_NAME}_s STATIC EXCLUDE_FROM_ALL
             $<TARGET_OBJECTS:${TARGET_NAME}_legacy_obj>
             $<TARGET_OBJECTS:${TARGET_NAME}_obj>
             ${IE_STATIC_DEPENDENT_FILES})
diff --git a/inference-engine/src/legacy_api/CMakeLists.txt b/inference-engine/src/legacy_api/CMakeLists.txt
index fab2f68d..864953a1 100644
--- a/inference-engine/src/legacy_api/CMakeLists.txt
+++ b/inference-engine/src/legacy_api/CMakeLists.txt
@@ -22,7 +22,7 @@ endif()
 
 # Create object library
 
-add_library(${TARGET_NAME}_obj OBJECT
+add_library(${TARGET_NAME}_obj OBJECT EXCLUDE_FROM_ALL
             ${LIBRARY_SRC}
             ${PUBLIC_HEADERS})
 
diff --git a/inference-engine/src/mkldnn_plugin/CMakeLists.txt b/inference-engine/src/mkldnn_plugin/CMakeLists.txt
index f52926d6..dd039e29 100644
--- a/inference-engine/src/mkldnn_plugin/CMakeLists.txt
+++ b/inference-engine/src/mkldnn_plugin/CMakeLists.txt
@@ -194,7 +194,7 @@ ie_add_api_validator_post_build_step(TARGET ${TARGET_NAME})
 
 #  add test object library
 
-add_library(${TARGET_NAME}_obj OBJECT ${SOURCES} ${HEADERS})
+add_library(${TARGET_NAME}_obj OBJECT EXCLUDE_FROM_ALL ${SOURCES} ${HEADERS})
 
 target_include_directories(${TARGET_NAME}_obj PRIVATE $<TARGET_PROPERTY:inference_engine_preproc_s,INTERFACE_INCLUDE_DIRECTORIES>
                                                       $<TARGET_PROPERTY:inference_engine_legacy,INTERFACE_INCLUDE_DIRECTORIES>
diff --git a/inference-engine/src/preprocessing/CMakeLists.txt b/inference-engine/src/preprocessing/CMakeLists.txt
index d47dfb35..a9046654 100644
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
index bd97c2c6..d89cdaa5 100644
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
index 797ef975..0cc5a65a 100644
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
diff --git a/inference-engine/thirdparty/CMakeLists.txt b/inference-engine/thirdparty/CMakeLists.txt
index fa2a4d02..c2ca41cd 100644
--- a/inference-engine/thirdparty/CMakeLists.txt
+++ b/inference-engine/thirdparty/CMakeLists.txt
@@ -61,11 +61,11 @@ else()
     target_include_directories(pugixml INTERFACE "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/pugixml/src>")
 endif()
 
-add_subdirectory(stb_lib)
+#add_subdirectory(stb_lib)
 add_subdirectory(ade)
 add_subdirectory(fluid/modules/gapi)
 
-set_target_properties(ade fluid stb_image PROPERTIES FOLDER thirdparty)
+set_target_properties(ade fluid PROPERTIES FOLDER thirdparty)
 
 # developer package
 
diff --git a/inference-engine/thirdparty/mkldnn.cmake b/inference-engine/thirdparty/mkldnn.cmake
index 0c2e936e..f36e7beb 100644
--- a/inference-engine/thirdparty/mkldnn.cmake
+++ b/inference-engine/thirdparty/mkldnn.cmake
@@ -117,7 +117,7 @@ if(WIN32)
     endif()
 endif()
 
-add_library(${TARGET} STATIC ${HDR} ${SRC})
+add_library(${TARGET} STATIC EXCLUDE_FROM_ALL ${HDR} ${SRC})
 set_ie_threading_interface_for(${TARGET})
 
 if(GEMM STREQUAL "OPENBLAS")
diff --git a/inference-engine/thirdparty/pugixml/CMakeLists.txt b/inference-engine/thirdparty/pugixml/CMakeLists.txt
index 8bcb2801..380fb468 100644
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
index 4c5a4766..6f5f2535 100644
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
index 2fa49195..ce68fdc8 100644
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
index 766521a1..04240a89 100644
--- a/openvino/itt/CMakeLists.txt
+++ b/openvino/itt/CMakeLists.txt
@@ -56,7 +56,7 @@ if(ENABLE_PROFILING_ITT)
     endif()
 endif()
 
-add_library(${TARGET_NAME} STATIC ${SOURCES})
+add_library(${TARGET_NAME} STATIC EXCLUDE_FROM_ALL ${SOURCES})
 
 add_library(openvino::itt ALIAS ${TARGET_NAME})
 
