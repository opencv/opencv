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
// Copyright (C) 2008, Willow Garage Inc., all rights reserved.
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
//   * The name of Intel Corporation may not be used to endorse or promote products
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
#include "precomp.hpp"

namespace cv
{

    inline Moments operator& ( const Moments & lhs, const Matx22d & rhs )
    {
        return Moments (
            lhs.m00,
            rhs ( 0, 0 ) * lhs.m10 + rhs ( 0, 1 ) * lhs.m01,
            rhs ( 1, 0 ) * lhs.m10 + rhs ( 1, 1 ) * lhs.m01,
            rhs ( 0, 0 ) * rhs ( 0, 0 ) * lhs.m20 + rhs ( 0, 1 ) * rhs ( 0, 1 ) * lhs.m02 + 2 * rhs ( 0, 0 ) * rhs ( 0, 1 ) * lhs.m11,
            rhs ( 0, 0 ) * rhs ( 1, 0 ) * lhs.m20 + rhs ( 0, 1 ) * rhs ( 1, 1 ) * lhs.m02 + ( rhs ( 0, 0 ) * rhs ( 1, 1 ) + rhs ( 0, 1 ) * rhs ( 1, 0 ) ) * lhs.m11,
            rhs ( 1, 0 ) * rhs ( 1, 0 ) * lhs.m20 + rhs ( 1, 1 ) * rhs ( 1, 1 ) * lhs.m02 + 2 * rhs ( 1, 0 ) * rhs ( 1, 1 ) * lhs.m11,
            rhs ( 0, 0 ) * rhs ( 0, 0 ) * rhs ( 0, 0 ) * lhs.m30 + 3 * rhs ( 0, 0 ) * rhs ( 0, 0 ) * rhs ( 0, 1 ) * lhs.m21 + 3 * rhs ( 0, 0 ) * rhs ( 0, 1 ) * rhs ( 0, 1 ) * lhs.m12 + rhs ( 0, 1 ) * rhs ( 0, 1 ) * rhs ( 0, 1 ) * lhs.m03,
            rhs ( 0, 0 ) * rhs ( 0, 0 ) * rhs ( 1, 0 ) * lhs.m30 + ( rhs ( 0, 0 ) * rhs ( 0, 0 ) * rhs ( 1, 1 ) + 2 * rhs ( 0, 0 ) * rhs ( 0, 1 ) * rhs ( 1, 0 ) ) * lhs.m21 + ( 2 * rhs ( 0, 0 ) * rhs ( 0, 1 ) * rhs ( 1, 1 ) + rhs ( 0, 1 ) * rhs ( 0, 1 ) * rhs ( 1, 0 ) ) * lhs.m12 + rhs ( 0, 1 ) * rhs ( 0, 1 ) * rhs ( 1, 1 ) * lhs.m03,
            rhs ( 0, 0 ) * rhs ( 1, 0 ) * rhs ( 1, 0 ) * lhs.m30 + ( rhs ( 1, 0 ) * rhs ( 1, 0 ) * rhs ( 0, 1 ) + 2 * rhs ( 0, 0 ) * rhs ( 1, 0 ) * rhs ( 1, 1 ) ) * lhs.m21 + ( 2 * rhs ( 0, 1 ) * rhs ( 1, 0 ) * rhs ( 1, 1 ) + rhs ( 1, 1 ) * rhs ( 1, 1 ) * rhs ( 0, 0 ) ) * lhs.m12 + rhs ( 0, 1 ) * rhs ( 1, 1 ) * rhs ( 1, 1 ) * lhs.m03,
            rhs ( 1, 0 ) * rhs ( 1, 0 ) * rhs ( 1, 0 ) * lhs.m30 + 3 * rhs ( 1, 0 ) * rhs ( 1, 0 ) * rhs ( 1, 1 ) * lhs.m21 + 3 * rhs ( 1, 0 ) * rhs ( 1, 1 ) * rhs ( 1, 1 ) * lhs.m12 + rhs ( 1, 1 ) * rhs ( 1, 1 ) * rhs ( 1, 1 ) * lhs.m03
        );
    }

    inline Matx23d operator| ( const Matx22d & lhs, const Matx21d & rhs )
    {
        return Matx23d ( lhs ( 0, 0 ), lhs ( 0, 1 ), rhs ( 0 ), lhs ( 1, 0 ), lhs ( 1, 1 ), rhs ( 1 ) );
    }

    Matx23d PeiLinNormalization ( InputArray I )
    {
        const Moments  M = moments ( I );
        const double  l1 = ( M.nu20 + M.nu02 + sqrt ( ( M.nu20 - M.nu02 ) * ( M.nu20 - M.nu02 ) + 4 * M.nu11 * M.nu11 ) ) / 2;
        const double  l2 = ( M.nu20 + M.nu02 - sqrt ( ( M.nu20 - M.nu02 ) * ( M.nu20 - M.nu02 ) + 4 * M.nu11 * M.nu11 ) ) / 2;
        const double  ex = ( M.nu11 ) / sqrt ( ( l1 - M.nu20 ) * ( l1 - M.nu20 ) + M.nu11 * M.nu11 );
        const double  ey = ( l1 - M.nu20 ) / sqrt ( ( l1 - M.nu20 ) * ( l1 - M.nu20 ) + M.nu11 * M.nu11 );
        const Matx22d  E = Matx22d ( ex, ey, -ey, ex );
        const Matx22d  W = Matx22d ( sqrt ( sqrt ( l1 * l2 ) ) / sqrt ( l1 ), 0, 0, sqrt ( sqrt ( l1 * l2 ) ) / sqrt ( l2 ) );
        const Matx21d  c = Matx21d ( M.m10 / M.m00, M.m01 / M.m00 );
        const Matx21d  i = Matx21d ( I.size().height / 2, I.size().width / 2 );
        const Moments  N = M & W * E;
        const double  t1 = N.nu12 + N.nu30;
        const double  t2 = N.nu03 + N.nu21;
        const double phi = atan2 ( -t1, t2 );
        const double psi = ( -t1 * sin ( phi ) + t2 * cos ( phi ) >= 0 ) ? phi : ( phi + CV_PI  );
        const Matx22d  A = Matx22d ( cos ( psi ), sin ( psi ), -sin ( psi ), cos ( psi ) );
        return ( A * W * E ) | ( i - A * W * E * c );
    }

}
