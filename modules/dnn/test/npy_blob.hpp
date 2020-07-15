// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2017, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef __OPENCV_DNN_TEST_NPY_BLOB_HPP__
#define __OPENCV_DNN_TEST_NPY_BLOB_HPP__

namespace cv
{

// Parse serialized NumPy array by np.save(...)
// Based on specification of .npy data format.
Mat blobFromNPY(const std::string& path);

}

#endif
