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
#include "NPnpInput.h"
#include "../Utils_Npnp/Parsing.h"
#include <iostream>
namespace NPnP
{
    std::shared_ptr<PnpInput> PnpInput::parse_input(int argc, char **argv)
    {
        if (argc != 5)
        {
            std::cout << "Expected 4 command line arguements:" << std::endl;
            std::cout << "\t1) points.csv path" << std::endl;
            std::cout << "\t2) lines.csv path" << std::endl;
            std::cout << "\t3) weights.csv path" << std::endl;
            std::cout << "\t4) indices.csv path" << std::endl;
            exit(1);
        }
        auto points_path = argv[1];
        auto lines_path = argv[2];
        auto weights_path = argv[3];
        auto indices_path = argv[4];

        auto points = parse_csv_vector_3d(points_path);
        auto lines = parse_csv_vector_3d(lines_path);
        auto weights = parse_csv_array<double>(weights_path);
        auto indices = parse_csv_array<int>(indices_path);

        if (points.size() != lines.size())
        {
            std::cout << "points amount = " << points.size()
                      << " whereas lines amount = " << lines.size() << std::endl;
            exit(1);
        }

        if (indices.size() != weights.size())
        {
            std::cout << "weights amount = " << weights.size()
                      << " whereas indices amount = " << indices.size() << std::endl;
            exit(1);
        }

        return PnpInput::init(points, lines, weights, indices);
    }

    PnpInput::PnpInput(std::vector<Eigen::Vector3d> a_points,
                       std::vector<Eigen::Vector3d> a_lines,
                       std::vector<double> a_weights, std::vector<int> a_indices,
                       int a_indices_amount, int a_points_amount)
        : points(std::move(a_points)), lines(std::move(a_lines)),
          weights(std::move(a_weights)), indices(std::move(a_indices)),
          indices_amount(a_indices_amount), points_amount(a_points_amount) {}

    std::shared_ptr<PnpInput> PnpInput::init(std::vector<Eigen::Vector3d> a_points,
                                             std::vector<Eigen::Vector3d> a_lines,
                                             std::vector<double> a_weights,
                                             std::vector<int> a_indices)
    {
        return std::make_shared<PnpInput>(std::move(a_points), std::move(a_lines),
                                          std::move(a_weights), std::move(a_indices),
                                          a_indices.size(), a_points.size());
    }
} // namespace NPnP
