#!/bin/bash
./bin/opencv_test_imgproc --gtest_filter=Imgproc_GaussianBlur*:Imgproc_Filter2D*
./bin/opencv_perf_imgproc --gtest_filter=Imgproc_GaussianBlur*:Imgproc_Filter2D*
