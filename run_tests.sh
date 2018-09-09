#!/bin/sh
export OPENCV_TEST_DATA_PATH=/Users/mkv/Documents/marat/opencv_extra/testdata/
export OPENCV_OPENCL_DEVICE='Apple:CPU:0'
export OPENCV_DNN_OPENCL_ALLOW_ALL_DEVICES=True
chmod 777 ./build/bin/opencv_test_dnn
./build/bin/opencv_test_dnn > ./last_test.txt