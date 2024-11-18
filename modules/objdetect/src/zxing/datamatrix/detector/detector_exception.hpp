// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Modified from ZXing. Copyright ZXing authors.
// Licensed under the Apache License, Version 2.0 (the "License").

/*
 * DetectorException.hpp
 *
 *  Created on: Aug 26, 2011
 *      Author: luiz
 */

#ifndef __ZXING_DATAMATRIX_DETECTOR_DETECTOR_EXCEPTION_HPP__
#define __ZXING_DATAMATRIX_DETECTOR_DETECTOR_EXCEPTION_HPP__

#include "../../exception.hpp"

namespace zxing {
namespace datamatrix {

class DetectorException : public Exception {
public:
    DetectorException(const char *msg);
    virtual ~DetectorException() throw();
};
}  // namespace datamatrix
}  // namespace zxing
#endif // __ZXING_DATAMATRIX_DETECTOR_DETECTOR_EXCEPTION_HPP__
