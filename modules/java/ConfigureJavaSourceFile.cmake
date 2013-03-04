# ConfigureJavaSourceFile.cmake:
# Needs to be called with -D INPUT_FILE=input_file -D OUTPUT_FILE=output_file -D native_module_output_name=opencv_javaNNN
configure_file(${INPUT_FILE} ${OUTPUT_FILE} @ONLY)
