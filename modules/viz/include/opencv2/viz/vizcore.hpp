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
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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
// Authors:
//  * Ozan Tonkal, ozantonkal@gmail.com
//  * Anatoly Baksheev, Itseez Inc.  myname.mysurname <> mycompany.com
//
//M*/

#ifndef __OPENCV_VIZCORE_HPP__
#define __OPENCV_VIZCORE_HPP__

#include <opencv2/viz/types.hpp>
#include <opencv2/viz/widgets.hpp>
#include <opencv2/viz/viz3d.hpp>

namespace cv
{
    namespace viz
    {
        //! takes coordiante frame data and builds transfrom to global coordinate frame
        CV_EXPORTS Affine3d makeTransformToGlobal(const Vec3d& axis_x, const Vec3d& axis_y, const Vec3d& axis_z, const Vec3d& origin = Vec3d::all(0));

        //! constructs camera pose from position, focal_point and up_vector (see gluLookAt() for more infromation)
        CV_EXPORTS Affine3d makeCameraPose(const Vec3d& position, const Vec3d& focal_point, const Vec3d& y_dir);

        //! retrieves a window by its name. If no window with such name, then it creates new.
        CV_EXPORTS Viz3d getWindowByName(const String &window_name);

        //! Unregisters all Viz windows from internal database. After it 'getWindowByName()' will create new windows instead getting existing from the database.
        CV_EXPORTS void unregisterAllWindows();

        //! Displays image in specified window
        CV_EXPORTS Viz3d imshow(const String& window_name, InputArray image, const Size& window_size = Size(-1, -1));

        //! checks float value for Nan
        inline bool isNan(float x)
        {
            unsigned int *u = reinterpret_cast<unsigned int *>(&x);
            return ((u[0] & 0x7f800000) == 0x7f800000) && (u[0] & 0x007fffff);
        }

        //! checks double value for Nan
        inline bool isNan(double x)
        {
            unsigned int *u = reinterpret_cast<unsigned int *>(&x);
            return (u[1] & 0x7ff00000) == 0x7ff00000 && (u[0] != 0 || (u[1] & 0x000fffff) != 0);
        }

        //! checks vectors for Nans
        template<typename _Tp, int cn> inline bool isNan(const Vec<_Tp, cn>& v)
        { return isNan(v.val[0]) || isNan(v.val[1]) || isNan(v.val[2]); }

        //! checks point for Nans
        template<typename _Tp> inline bool isNan(const Point3_<_Tp>& p)
        { return isNan(p.x) || isNan(p.y) || isNan(p.z); }


        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// Read/write clouds. Supported formats: ply, xyz, obj and stl (readonly)

        CV_EXPORTS void writeCloud(const String& file, InputArray cloud, InputArray colors = noArray(), InputArray normals = noArray(), bool binary = false);
        CV_EXPORTS Mat  readCloud (const String& file, OutputArray colors = noArray(), OutputArray normals = noArray());

        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// Reads mesh. Only ply format is supported now and no texture load support

        CV_EXPORTS Mesh readMesh(const String& file);

        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// Read/write poses and trajectories

        CV_EXPORTS bool readPose(const String& file, Affine3d& pose, const String& tag = "pose");
        CV_EXPORTS void writePose(const String& file, const Affine3d& pose, const String& tag = "pose");

        //! takes vector<Affine3<T>> with T = float/dobule and writes to a sequence of files with given filename format
        CV_EXPORTS void writeTrajectory(InputArray traj, const String& files_format = "pose%05d.xml", int start = 0, const String& tag = "pose");

        //! takes vector<Affine3<T>> with T = float/dobule and loads poses from sequence of files
        CV_EXPORTS void readTrajectory(OutputArray traj, const String& files_format = "pose%05d.xml", int start = 0, int end = INT_MAX, const String& tag = "pose");


        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// Computing normals for mesh

        CV_EXPORTS void computeNormals(const Mesh& mesh, OutputArray normals);

    } /* namespace viz */
} /* namespace cv */

#endif /* __OPENCV_VIZCORE_HPP__ */
