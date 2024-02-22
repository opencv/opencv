/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/
#ifndef PNP_USING_EIGEN_LIBRARY_BARRIERMETHODSETTINGS_H
#define PNP_USING_EIGEN_LIBRARY_BARRIERMETHODSETTINGS_H

#include <memory>

namespace NPnP
{
    class BarrierMethodSettings
    {
    public:
        double epsilon;
        int binary_search_depth;
        bool verbose;
        int max_inner_iterations;
        double miu;
        double valid_result_threshold;

        BarrierMethodSettings(double a_epsilon, int a_binary_search_depth, bool a_verbose,
                              int a_max_inner_iterations, double a_miu,
                              double a_valid_result_threshold)
            : epsilon(a_epsilon), binary_search_depth(a_binary_search_depth),
              verbose(a_verbose), max_inner_iterations(a_max_inner_iterations), miu(a_miu),
              valid_result_threshold(a_valid_result_threshold) {}

        static std::shared_ptr<BarrierMethodSettings>
        init(double epsilon = 4E-8, int binary_search_depth = 20, bool verbose = true,
             int max_inner_iterations = 20, double miu = 50,
             double valid_result_threshold = 0.001)
        {
            return std::make_shared<BarrierMethodSettings>(
                epsilon, binary_search_depth, verbose, max_inner_iterations, miu,
                valid_result_threshold);
        }
    };
} // namespace NPnP

#endif // PNP_USING_EIGEN_LIBRARY_BARRIERMETHODSETTINGS_H
