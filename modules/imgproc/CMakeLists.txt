set(the_description "Image Processing")
ocv_add_dispatched_file(accum SSE2 AVX NEON)
ocv_define_module(imgproc opencv_core WRAP java python js)
