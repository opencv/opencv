%feature("director") Mat;
%feature("director") glcamera;
%feature("director") image_pool;
%typemap("javapackage") Mat, Mat *, Mat & "com.opencv.jni";
%typemap("javapackage") glcamera, glcamera *, glcamera & "com.opencv.jni";
%typemap("javapackage") image_pool, image_pool *, image_pool & "com.opencv.jni";