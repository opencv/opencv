set(the_description "Image Processing")
ocv_add_dispatched_file(accum SSE4_1 AVX AVX2)
ocv_define_module(imgproc opencv_core WRAP java python js)
