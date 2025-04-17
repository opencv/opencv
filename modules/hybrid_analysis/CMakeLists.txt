set(NAME "opencv_hybrid_analysis")
set(DESCRIPTION "Hybrid Vision-Text Context Analysis")
ocv_add_module(${NAME} PRIVATE_REQUIRTS opencv_dnn)

ocv_add_header_files(${NAME}
    HEADERS
    include/opencv2/hybrid_analysis.hpp
)

ocv_add_source_files(${NAME}
    SOURCES
    src/vit_onnx_fusion.cpp
    src/text_encoder.cpp
)

ocv_add_tests(${NAME}
    FILES
    test/test_hybrid.cpp
)
