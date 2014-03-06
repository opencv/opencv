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

#include "precomp.hpp"

cv::Affine3d cv::viz::makeTransformToGlobal(const Vec3d& axis_x, const Vec3d& axis_y, const Vec3d& axis_z, const Vec3d& origin)
{
    Affine3d::Mat3 R(axis_x[0], axis_y[0], axis_z[0],
                     axis_x[1], axis_y[1], axis_z[1],
                     axis_x[2], axis_y[2], axis_z[2]);

    return Affine3d(R, origin);
}

cv::Affine3d cv::viz::makeCameraPose(const Vec3d& position, const Vec3d& focal_point, const Vec3d& y_dir)
{
    // Compute the transformation matrix for drawing the camera frame in a scene
    Vec3d n = normalize(focal_point - position);
    Vec3d u = normalize(y_dir.cross(n));
    Vec3d v = n.cross(u);

    return makeTransformToGlobal(u, v, n, position);
}

///////////////////////////////////////////////////////////////////////////////////////////////
/// VizStorage implementation

cv::viz::VizMap cv::viz::VizStorage::storage;
void cv::viz::VizStorage::unregisterAll() { storage.clear(); }

cv::viz::Viz3d& cv::viz::VizStorage::get(const String &window_name)
{
    String name = generateWindowName(window_name);
    VizMap::iterator vm_itr = storage.find(name);
    CV_Assert(vm_itr != storage.end());
    return vm_itr->second;
}

void cv::viz::VizStorage::add(const Viz3d& window)
{
    String window_name = window.getWindowName();
    VizMap::iterator vm_itr = storage.find(window_name);
    CV_Assert(vm_itr == storage.end());
    storage.insert(std::make_pair(window_name, window));
}

bool cv::viz::VizStorage::windowExists(const String &window_name)
{
    String name = generateWindowName(window_name);
    return storage.find(name) != storage.end();
}

void cv::viz::VizStorage::removeUnreferenced()
{
    for(VizMap::iterator pos = storage.begin(); pos != storage.end();)
        if(pos->second.impl_->ref_counter == 1)
            storage.erase(pos++);
        else
            ++pos;
}

cv::String cv::viz::VizStorage::generateWindowName(const String &window_name)
{
    String output = "Viz";
    // Already is Viz
    if (window_name == output)
        return output;

    String prefixed = output + " - ";
    if (window_name.substr(0, prefixed.length()) == prefixed)
        output = window_name; // Already has "Viz - "
    else if (window_name.substr(0, output.length()) == output)
        output = prefixed + window_name; // Doesn't have prefix
    else
        output = (window_name == "" ? output : prefixed + window_name);

    return output;
}

cv::viz::Viz3d cv::viz::getWindowByName(const String &window_name) { return Viz3d (window_name); }
void cv::viz::unregisterAllWindows() { VizStorage::unregisterAll(); }

cv::viz::Viz3d cv::viz::imshow(const String& window_name, InputArray image, const Size& window_size)
{
    Viz3d viz = getWindowByName(window_name);
    viz.showImage(image, window_size);
    return viz;
}

///////////////////////////////////////////////////////////////////////////////////////////////
/// Read/write clouds. Supported formats: ply, stl, xyz, obj

void cv::viz::writeCloud(const String& file, InputArray cloud, InputArray colors, InputArray normals, bool binary)
{
    CV_Assert(file.size() > 4 && "Extention is required");
    String extention = file.substr(file.size()-4);

    vtkSmartPointer<vtkCloudMatSource> source = vtkSmartPointer<vtkCloudMatSource>::New();
    source->SetColorCloudNormals(cloud, colors, normals);

    vtkSmartPointer<vtkWriter> writer;
    if (extention == ".xyz")
    {
        writer = vtkSmartPointer<vtkXYZWriter>::New();
        vtkXYZWriter::SafeDownCast(writer)->SetFileName(file.c_str());
    }
    else if (extention == ".ply")
    {
        writer = vtkSmartPointer<vtkPLYWriter>::New();
        vtkPLYWriter::SafeDownCast(writer)->SetFileName(file.c_str());
        vtkPLYWriter::SafeDownCast(writer)->SetFileType(binary ? VTK_BINARY : VTK_ASCII);
        vtkPLYWriter::SafeDownCast(writer)->SetArrayName("Colors");
    }
    else if (extention == ".obj")
    {
        writer = vtkSmartPointer<vtkOBJWriter>::New();
        vtkOBJWriter::SafeDownCast(writer)->SetFileName(file.c_str());
    }
    else
        CV_Assert(!"Unsupported format");

    writer->SetInputConnection(source->GetOutputPort());
    writer->Write();
}

cv::Mat cv::viz::readCloud(const String& file, OutputArray colors, OutputArray normals)
{
    CV_Assert(file.size() > 4 && "Extention is required");
    String extention = file.substr(file.size()-4);

    vtkSmartPointer<vtkPolyDataAlgorithm> reader;
    if (extention == ".xyz")
    {
        reader = vtkSmartPointer<vtkSimplePointsReader>::New();
        vtkSimplePointsReader::SafeDownCast(reader)->SetFileName(file.c_str());
    }
    else if (extention == ".ply")
    {
        reader = vtkSmartPointer<vtkPLYReader>::New();
        CV_Assert(vtkPLYReader::CanReadFile(file.c_str()));
        vtkPLYReader::SafeDownCast(reader)->SetFileName(file.c_str());
    }
    else if (extention == ".obj")
    {
        reader = vtkSmartPointer<vtkOBJReader>::New();
        vtkOBJReader::SafeDownCast(reader)->SetFileName(file.c_str());
    }
    else if (extention == ".stl")
    {
        reader = vtkSmartPointer<vtkSTLReader>::New();
        vtkSTLReader::SafeDownCast(reader)->SetFileName(file.c_str());
    }
    else
        CV_Assert(!"Unsupported format");

    cv::Mat cloud;

    vtkSmartPointer<vtkCloudMatSink> sink = vtkSmartPointer<vtkCloudMatSink>::New();
    sink->SetInputConnection(reader->GetOutputPort());
    sink->SetOutput(cloud, colors, normals);
    sink->Write();

    return cloud;
}

cv::viz::Mesh cv::viz::readMesh(const String& file) { return Mesh::load(file); }

///////////////////////////////////////////////////////////////////////////////////////////////
/// Read/write poses and trajectories

bool cv::viz::readPose(const String& file, Affine3d& pose, const String& tag)
{
    FileStorage fs(file, FileStorage::READ);
    if (!fs.isOpened())
        return false;

    Mat hdr(pose.matrix, false);
    fs[tag] >> hdr;
    if (hdr.empty() || hdr.cols != pose.matrix.cols || hdr.rows != pose.matrix.rows)
        return false;

    hdr.convertTo(pose.matrix, CV_64F);
    return true;
}

void cv::viz::writePose(const String& file, const Affine3d& pose, const String& tag)
{
    FileStorage fs(file, FileStorage::WRITE);
    fs << tag << Mat(pose.matrix, false);
}

void cv::viz::readTrajectory(OutputArray _traj, const String& files_format, int start, int end, const String& tag)
{
    CV_Assert(_traj.kind() == _InputArray::STD_VECTOR || _traj.kind() == _InputArray::MAT);

    start = max(0, std::min(start, end));
    end = std::max(start, end);

    std::vector<Affine3d> traj;

    for(int i = start; i < end; ++i)
    {
        Affine3d affine;
        bool ok = readPose(cv::format(files_format.c_str(), i), affine, tag);
        if (!ok)
            break;

        traj.push_back(affine);
    }

    Mat(traj).convertTo(_traj, _traj.depth());
}

void cv::viz::writeTrajectory(InputArray _traj, const String& files_format, int start, const String& tag)
{
    if (_traj.kind() == _InputArray::STD_VECTOR_MAT)
    {
        std::vector<Mat>& v = *(std::vector<Mat>*)_traj.getObj();

        for(size_t i = 0, index = max(0, start); i < v.size(); ++i, ++index)
        {
            Affine3d affine;
            Mat pose = v[i];
            CV_Assert(pose.type() == CV_32FC(16) || pose.type() == CV_64FC(16));
            pose.copyTo(affine.matrix);
            writePose(cv::format(files_format.c_str(), index), affine, tag);
        }
        return;
    }

    if (_traj.kind() == _InputArray::STD_VECTOR || _traj.kind() == _InputArray::MAT)
    {
        CV_Assert(_traj.type() == CV_32FC(16) || _traj.type() == CV_64FC(16));

        Mat traj = _traj.getMat();

        if (traj.depth() == CV_32F)
            for(size_t i = 0, index = max(0, start); i < traj.total(); ++i, ++index)
                writePose(cv::format(files_format.c_str(), index), traj.at<Affine3f>(i), tag);

        if (traj.depth() == CV_64F)
            for(size_t i = 0, index = max(0, start); i < traj.total(); ++i, ++index)
                writePose(cv::format(files_format.c_str(), index), traj.at<Affine3d>(i), tag);
    }

    CV_Assert(!"Unsupported array kind");
}

///////////////////////////////////////////////////////////////////////////////////////////////
/// Computing normals for mesh

void cv::viz::computeNormals(const Mesh& mesh, OutputArray _normals)
{
    vtkSmartPointer<vtkPolyData> polydata = getPolyData(WMesh(mesh));
    vtkSmartPointer<vtkPolyData> with_normals = VtkUtils::ComputeNormals(polydata);

    vtkSmartPointer<vtkDataArray> generic_normals = with_normals->GetPointData()->GetNormals();
    if(generic_normals)
    {
        Mat normals(1, generic_normals->GetNumberOfTuples(), CV_64FC3);
        Vec3d *optr = normals.ptr<Vec3d>();

        for(int i = 0; i < generic_normals->GetNumberOfTuples(); ++i, ++optr)
            generic_normals->GetTuple(i, optr->val);

        normals.convertTo(_normals, mesh.cloud.type());
    }
    else
        _normals.release();
}
