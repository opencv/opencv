#!/bin/bash
cd build
./bin/opencv_test_imgproc --gtest_filter=*_GaussianBlur*:*_Filter2D*
./bin/opencv_perf_imgproc --gtest_filter=*_Filter2D*:*_GaussianBlur*
