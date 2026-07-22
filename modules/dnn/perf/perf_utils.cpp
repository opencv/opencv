// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2017, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "perf_precomp.hpp"

namespace opencv_test {

using Utils_blobFromImage = TestBaseWithParam<std::vector<int>>;
PERF_TEST_P_(Utils_blobFromImage, HWC_TO_NCHW) {
    std::vector<int> input_shape = GetParam();

    Mat input(input_shape, CV_32FC3);
    randu(input, -10.0f, 10.f);

    TEST_CYCLE() {
        Mat blob = blobFromImage(input);
    }

    SANITY_CHECK_NOTHING();
}

INSTANTIATE_TEST_CASE_P(/**/, Utils_blobFromImage,
    Values(std::vector<int>{  32,   32},
           std::vector<int>{  64,   64},
           std::vector<int>{ 128,  128},
           std::vector<int>{ 256,  256},
           std::vector<int>{ 512,  512},
           std::vector<int>{1024, 1024},
           std::vector<int>{2048, 2048})
);

using Utils_blobFromImages = TestBaseWithParam<std::vector<int>>;
PERF_TEST_P_(Utils_blobFromImages, HWC_TO_NCHW) {
    std::vector<int> input_shape = GetParam();

    int batch = input_shape.front();
    std::vector<int> input_shape_no_batch(input_shape.begin()+1, input_shape.end());

    if (input_shape_no_batch[0]*input_shape_no_batch[1] >= 2048*2048)
    {
        applyTestTag( CV_TEST_TAG_MEMORY_2GB);
    }

    std::vector<Mat> inputs;
    for (int i = 0; i < batch; i++) {
        Mat input(input_shape_no_batch, CV_32FC3);
        randu(input, -10.0f, 10.f);
        inputs.push_back(input);
    }

    TEST_CYCLE() {
        Mat blobs = blobFromImages(inputs);
    }

    SANITY_CHECK_NOTHING();
}

INSTANTIATE_TEST_CASE_P(/**/, Utils_blobFromImages,
    Values(std::vector<int>{16,   32,   32},
           std::vector<int>{16,   64,   64},
           std::vector<int>{16,  128,  128},
           std::vector<int>{16,  256,  256},
           std::vector<int>{16,  512,  512},
           std::vector<int>{16, 1024, 1024},
           std::vector<int>{16, 2048, 2048})
);

// NCHW, 8U->32F, C3, mean+scale+swapRB at 640x640
using Utils_blobFromImage_8U_NCHW = TestBaseWithParam<std::vector<int>>;
PERF_TEST_P_(Utils_blobFromImage_8U_NCHW, MeanScale_SwapRB) {
    std::vector<int> input_shape = GetParam();

    Mat input(input_shape, CV_8UC3);
    randu(input, 0, 255);

    TEST_CYCLE() {
        Mat blob = blobFromImage(input, 1.0/255.0, Size(), Scalar(104, 117, 123), true, false, CV_32F);
    }

    SANITY_CHECK_NOTHING();
}

INSTANTIATE_TEST_CASE_P(/**/, Utils_blobFromImage_8U_NCHW,
    Values(std::vector<int>{ 640,  640})
);

// NHWC, 8U->32F, C3
using Utils_blobFromImage_8U_NHWC = TestBaseWithParam<std::vector<int>>;
PERF_TEST_P_(Utils_blobFromImage_8U_NHWC, SwapRB) {
    std::vector<int> input_shape = GetParam();

    Mat input(input_shape, CV_8UC3);
    randu(input, 0, 255);

    Image2BlobParams params;
    params.scalefactor = Scalar::all(1.0);
    params.swapRB = true;
    params.datalayout = DNN_LAYOUT_NHWC;

    TEST_CYCLE() {
        Mat blob = blobFromImageWithParams(input, params);
    }

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(Utils_blobFromImage_8U_NHWC, MeanScale_SwapRB) {
    std::vector<int> input_shape = GetParam();

    Mat input(input_shape, CV_8UC3);
    randu(input, 0, 255);

    Image2BlobParams params;
    params.scalefactor = Scalar::all(1.0/255.0);
    params.mean = Scalar(104, 117, 123);
    params.swapRB = true;
    params.datalayout = DNN_LAYOUT_NHWC;

    TEST_CYCLE() {
        Mat blob = blobFromImageWithParams(input, params);
    }

    SANITY_CHECK_NOTHING();
}

INSTANTIATE_TEST_CASE_P(/**/, Utils_blobFromImage_8U_NHWC,
    Values(std::vector<int>{ 224,  224},
           std::vector<int>{ 640,  640})
);

// NHWC, 32F->32F, C3
using Utils_blobFromImage_32F_NHWC = TestBaseWithParam<std::vector<int>>;
PERF_TEST_P_(Utils_blobFromImage_32F_NHWC, SwapRB) {
    std::vector<int> input_shape = GetParam();

    Mat input(input_shape, CV_32FC3);
    randu(input, 0.0f, 1.0f);

    Image2BlobParams params;
    params.scalefactor = Scalar::all(1.0);
    params.swapRB = true;
    params.datalayout = DNN_LAYOUT_NHWC;

    TEST_CYCLE() {
        Mat blob = blobFromImageWithParams(input, params);
    }

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(Utils_blobFromImage_32F_NHWC, MeanScale_SwapRB) {
    std::vector<int> input_shape = GetParam();

    Mat input(input_shape, CV_32FC3);
    randu(input, 0.0f, 1.0f);

    Image2BlobParams params;
    params.scalefactor = Scalar::all(1.0/0.226);
    params.mean = Scalar(0.485, 0.456, 0.406);
    params.swapRB = true;
    params.datalayout = DNN_LAYOUT_NHWC;

    TEST_CYCLE() {
        Mat blob = blobFromImageWithParams(input, params);
    }

    SANITY_CHECK_NOTHING();
}

INSTANTIATE_TEST_CASE_P(/**/, Utils_blobFromImage_32F_NHWC,
    Values(std::vector<int>{ 224,  224},
           std::vector<int>{ 640,  640})
);

// Resize+crop, 8U->32F, C3, mean+scale+swapRB to 640x640
using Utils_blobFromImage_8U_Resize = TestBaseWithParam<std::vector<int>>;
PERF_TEST_P_(Utils_blobFromImage_8U_Resize, NHWC_Crop_MeanScale_SwapRB) {
    std::vector<int> input_shape = GetParam();

    Mat input(input_shape, CV_8UC3);
    randu(input, 0, 255);

    Image2BlobParams params;
    params.scalefactor = Scalar::all(1.0/255.0);
    params.size = Size(640, 640);
    params.mean = Scalar(104, 117, 123);
    params.swapRB = true;
    params.datalayout = DNN_LAYOUT_NHWC;
    params.paddingmode = DNN_PMODE_CROP_CENTER;

    TEST_CYCLE() {
        Mat blob = blobFromImageWithParams(input, params);
    }

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(Utils_blobFromImage_8U_Resize, NCHW_Crop_MeanScale_SwapRB) {
    std::vector<int> input_shape = GetParam();

    Mat input(input_shape, CV_8UC3);
    randu(input, 0, 255);

    TEST_CYCLE() {
        Mat blob = blobFromImage(input, 1.0/255.0, Size(640, 640), Scalar(104, 117, 123), true, true, CV_32F);
    }

    SANITY_CHECK_NOTHING();
}

INSTANTIATE_TEST_CASE_P(/**/, Utils_blobFromImage_8U_Resize,
    Values(std::vector<int>{  720, 1280},
           std::vector<int>{ 1080, 1920},
           std::vector<int>{ 2160, 3840})
);

// Resize+crop, NCHW, 32F->32F, C3, mean+scale+swapRB to 300x300
using Utils_blobFromImage_32F_NCHW_Resize = TestBaseWithParam<std::vector<int>>;
PERF_TEST_P_(Utils_blobFromImage_32F_NCHW_Resize, Crop_MeanScale_SwapRB_To300) {
    std::vector<int> input_shape = GetParam();

    Mat input(input_shape, CV_32FC3);
    randu(input, 0.0f, 1.0f);

    TEST_CYCLE() {
        Mat blob = blobFromImage(input, 1.0/0.226, Size(300, 300), Scalar(0.485, 0.456, 0.406), true, true, CV_32F);
    }

    SANITY_CHECK_NOTHING();
}

INSTANTIATE_TEST_CASE_P(/**/, Utils_blobFromImage_32F_NCHW_Resize,
    Values(std::vector<int>{  720, 1280},
           std::vector<int>{ 1080, 1920})
);

// Resize+crop, NCHW, 8U->8U, C3
using Utils_blobFromImage_8U_to_8U_Crop = TestBaseWithParam<std::vector<int>>;
PERF_TEST_P_(Utils_blobFromImage_8U_to_8U_Crop, NCHW_SwapRB) {
    std::vector<int> input_shape = GetParam();

    Mat input(input_shape, CV_8UC3);
    randu(input, 0, 255);

    TEST_CYCLE() {
        Mat blob = blobFromImage(input, 1.0, Size(640, 640), Scalar(), true, true, CV_8U);
    }

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(Utils_blobFromImage_8U_to_8U_Crop, NCHW) {
    std::vector<int> input_shape = GetParam();

    Mat input(input_shape, CV_8UC3);
    randu(input, 0, 255);

    TEST_CYCLE() {
        Mat blob = blobFromImage(input, 1.0, Size(640, 640), Scalar(), false, true, CV_8U);
    }

    SANITY_CHECK_NOTHING();
}

INSTANTIATE_TEST_CASE_P(/**/, Utils_blobFromImage_8U_to_8U_Crop,
    Values(std::vector<int>{ 1080, 1920},
           std::vector<int>{ 2160, 3840})
);

// Resize, NCHW, 8U->8U, C3
using Utils_blobFromImage_8U_to_8U_Resize = TestBaseWithParam<std::vector<int>>;
PERF_TEST_P_(Utils_blobFromImage_8U_to_8U_Resize, NCHW_SwapRB) {
    std::vector<int> input_shape = GetParam();

    Mat input(input_shape, CV_8UC3);
    randu(input, 0, 255);

    TEST_CYCLE() {
        Mat blob = blobFromImage(input, 1.0, Size(640, 640), Scalar(), true, false, CV_8U);
    }

    SANITY_CHECK_NOTHING();
}

INSTANTIATE_TEST_CASE_P(/**/, Utils_blobFromImage_8U_to_8U_Resize,
    Values(std::vector<int>{ 1080, 1920},
           std::vector<int>{ 2160, 3840})
);

// Resize+crop, NCHW, 32F->32F, C1, mean
using Utils_blobFromImage_32F_NCHW_C1 = TestBaseWithParam<std::vector<int>>;
PERF_TEST_P_(Utils_blobFromImage_32F_NCHW_C1, Crop_MeanScale_To224) {
    std::vector<int> input_shape = GetParam();

    Mat input(input_shape, CV_32FC1);
    randu(input, 0.0f, 1.0f);

    TEST_CYCLE() {
        Mat blob = blobFromImage(input, 1.0/0.226, Size(224, 224), Scalar(0.5), false, true, CV_32F);
    }

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(Utils_blobFromImage_32F_NCHW_C1, Crop_MeanScale_To640) {
    std::vector<int> input_shape = GetParam();

    Mat input(input_shape, CV_32FC1);
    randu(input, 0.0f, 1.0f);

    TEST_CYCLE() {
        Mat blob = blobFromImage(input, 1.0/0.226, Size(640, 640), Scalar(0.5), false, true, CV_32F);
    }

    SANITY_CHECK_NOTHING();
}

INSTANTIATE_TEST_CASE_P(/**/, Utils_blobFromImage_32F_NCHW_C1,
    Values(std::vector<int>{ 1080, 1920},
           std::vector<int>{ 2160, 3840})
);

// Batch=8, NHWC, 8U->32F, C3, mean+scale+swapRB
using Utils_blobFromImages_NoResize = TestBaseWithParam<std::vector<int>>;
PERF_TEST_P_(Utils_blobFromImages_NoResize, NHWC_MeanScale_SwapRB) {
    std::vector<int> input_shape = GetParam();
    int batch = input_shape.front();
    std::vector<int> input_shape_no_batch(input_shape.begin()+1, input_shape.end());

    std::vector<Mat> inputs;
    for (int i = 0; i < batch; i++) {
        Mat input(input_shape_no_batch, CV_8UC3);
        randu(input, 0, 255);
        inputs.push_back(input);
    }

    Image2BlobParams params;
    params.scalefactor = Scalar::all(1.0/255.0);
    params.mean = Scalar(104, 117, 123);
    params.swapRB = true;
    params.datalayout = DNN_LAYOUT_NHWC;

    TEST_CYCLE() {
        Mat blob = blobFromImagesWithParams(inputs, params);
    }

    SANITY_CHECK_NOTHING();
}

INSTANTIATE_TEST_CASE_P(/**/, Utils_blobFromImages_NoResize,
    Values(std::vector<int>{8,  640,  640})
);

// Batch=8, resize+crop to 640x640, 8U->32F, C3, mean+scale+swapRB
using Utils_blobFromImages_Resize = TestBaseWithParam<std::vector<int>>;
PERF_TEST_P_(Utils_blobFromImages_Resize, NHWC_Crop_MeanScale_SwapRB) {
    std::vector<int> input_shape = GetParam();
    int batch = input_shape.front();
    std::vector<int> input_shape_no_batch(input_shape.begin()+1, input_shape.end());

    std::vector<Mat> inputs;
    for (int i = 0; i < batch; i++) {
        Mat input(input_shape_no_batch, CV_8UC3);
        randu(input, 0, 255);
        inputs.push_back(input);
    }

    Image2BlobParams params;
    params.scalefactor = Scalar::all(1.0/255.0);
    params.size = Size(640, 640);
    params.mean = Scalar(104, 117, 123);
    params.swapRB = true;
    params.datalayout = DNN_LAYOUT_NHWC;
    params.paddingmode = DNN_PMODE_CROP_CENTER;

    TEST_CYCLE() {
        Mat blob = blobFromImagesWithParams(inputs, params);
    }

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(Utils_blobFromImages_Resize, NCHW_Crop_MeanScale_SwapRB) {
    std::vector<int> input_shape = GetParam();
    int batch = input_shape.front();
    std::vector<int> input_shape_no_batch(input_shape.begin()+1, input_shape.end());

    std::vector<Mat> inputs;
    for (int i = 0; i < batch; i++) {
        Mat input(input_shape_no_batch, CV_8UC3);
        randu(input, 0, 255);
        inputs.push_back(input);
    }

    TEST_CYCLE() {
        Mat blob = blobFromImages(inputs, 1.0/255.0, Size(640, 640), Scalar(104, 117, 123), true, true, CV_32F);
    }

    SANITY_CHECK_NOTHING();
}

INSTANTIATE_TEST_CASE_P(/**/, Utils_blobFromImages_Resize,
    Values(std::vector<int>{8,  720, 1280},
           std::vector<int>{8, 1080, 1920})
);

}
