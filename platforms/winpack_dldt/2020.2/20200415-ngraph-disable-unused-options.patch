diff --git a/CMakeLists.txt b/CMakeLists.txt
index edf8233f..addac6cd 100644
--- a/CMakeLists.txt
+++ b/CMakeLists.txt
@@ -78,8 +78,7 @@ function(build_ngraph)
     if (NOT ANDROID)
         ngraph_set(NGRAPH_UNIT_TEST_ENABLE TRUE)
         ngraph_set(NGRAPH_UNIT_TEST_OPENVINO_ENABLE TRUE)
-        # ngraph_set(NGRAPH_ONNX_IMPORT_ENABLE TRUE)
-        set(NGRAPH_ONNX_IMPORT_ENABLE TRUE CACHE BOOL "" FORCE)
+        ngraph_set(NGRAPH_ONNX_IMPORT_ENABLE TRUE)
     else()
         ngraph_set(NGRAPH_UNIT_TEST_ENABLE FALSE)
         ngraph_set(NGRAPH_TEST_UTIL_ENABLE FALSE)
