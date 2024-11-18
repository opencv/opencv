// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Modified from ZXing. Copyright ZXing authors.
// Licensed under the Apache License, Version 2.0 (the "License").

#ifndef __ZXING_COMMON_KMEANS_HPP__
#define __ZXING_COMMON_KMEANS_HPP__

#include<vector>

namespace zxing {

typedef unsigned int uint;

struct Cluster
{
    std::vector<double> centroid;
    std::vector<uint> samples;
};


double cal_distance(std::vector<double> a, std::vector<double> b);
std::vector<Cluster> k_means(std::vector<std::vector<double> > trainX, uint k, uint maxepoches, uint minchanged);

}  // namespace zxing
#endif // __ZXING_COMMON_KMEANS_HPP__