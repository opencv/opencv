// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"

#include "coders_base.hpp"
#include "coders_obj.hpp"
#include "coders_ply.hpp"

#include "opencv2/core/utils/filesystem.hpp"
#include "opencv2/core/utils/filesystem.private.hpp"

#include <memory>
#include <iostream>

namespace cv
{

namespace pc
{

#if OPENCV_HAVE_FILESYSTEM_SUPPORT

static PointCloudDecoder findDecoder(const String &filename)
{
    size_t filename_length = filename.length();
    if ((filename.substr(filename_length - 4, 4) == ".obj") || (filename.substr(filename_length - 4, 4) == ".OBJ"))
    {
        return std::unique_ptr<ObjDecoder>(new ObjDecoder());
    }
    if ((filename.substr(filename_length - 4, 4) == ".ply") || (filename.substr(filename_length - 4, 4) == ".PLY"))
    {
        return std::unique_ptr<PlyDecoder>(new PlyDecoder());
    }

    return nullptr;
}

static PointCloudEncoder findEncoder(const String &filename)
{
    size_t filename_length = filename.length();
    if ((filename.substr(filename_length - 4, 4) == ".obj") || (filename.substr(filename_length - 4, 4) == ".OBJ"))
    {
        return std::unique_ptr<ObjEncoder>(new ObjEncoder());
    }
    if ((filename.substr(filename_length - 4, 4) == ".ply") || (filename.substr(filename_length - 4, 4) == ".PLY"))
    {
        return std::unique_ptr<PlyEncoder>(new PlyEncoder());
    }

    return nullptr;
}

#endif

}

void loadPointCloud(const String &filename, OutputArray vertices, OutputArray normals)
{
#if OPENCV_HAVE_FILESYSTEM_SUPPORT
    pc::PointCloudDecoder decoder;

    decoder = pc::findDecoder(filename);
    if (!decoder)
        CV_Error(Error::StsError, "File extention not supported");

    decoder->setSource(filename);

    std::vector<Point3f> vec_vertices;
    std::vector<Point3f> vec_normals;

    decoder->readData(vec_vertices, vec_normals);

    if (!vec_vertices.empty())
        Mat(static_cast<int>(vec_vertices.size()), 1, CV_32FC3, &vec_vertices[0]).copyTo(vertices);

    if (!vec_normals.empty())
        Mat(static_cast<int>(vec_normals.size()), 1, CV_32FC3, &vec_normals[0]).copyTo(normals);
#else // OPENCV_HAVE_FILESYSTEM_SUPPORT
    CV_UNUSED(filename);
    CV_UNUSED(vertices);
    CV_UNUSED(normals);
    CV_Error(Error::StsNotImplemented, "File system support is disabled in this OpenCV build!");
#endif
}

void savePointCloud(const String &filename, InputArray vertices, InputArray normals)
{
#if OPENCV_HAVE_FILESYSTEM_SUPPORT
    CV_Assert(!vertices.empty());

    pc::PointCloudEncoder encoder;

    encoder = pc::findEncoder(filename);
    if (!encoder)
        CV_Error(Error::StsError, "File extention not supported");

    encoder->setDestination(filename);

    std::vector<Point3f> vec_vertices(vertices.getMat());
    std::vector<Point3f> vec_normals;
    if (!normals.empty()){
        vec_normals = normals.getMat();
    }

    encoder->writeData(vec_vertices, vec_normals);
#else // OPENCV_HAVE_FILESYSTEM_SUPPORT
    CV_UNUSED(filename);
    CV_UNUSED(vertices);
    CV_UNUSED(normals);
    CV_Error(Error::StsNotImplemented, "File system support is disabled in this OpenCV build!");
#endif
}

void loadMesh(const String &filename, OutputArray vertices, OutputArray normals, OutputArray indices)
{
#if OPENCV_HAVE_FILESYSTEM_SUPPORT
    pc::PointCloudDecoder decoder;

    decoder = pc::findDecoder(filename);

    decoder->setSource(filename);

    std::vector<Point3f> vec_vertices;
    std::vector<Point3f> vec_normals;
    std::vector<std::vector<int32_t>> vec_idices;

    decoder->readData(vec_vertices, vec_normals, vec_idices);

    if (!vec_vertices.empty())
        Mat(static_cast<int>(vec_vertices.size()), 1, CV_32FC3, &vec_vertices[0]).copyTo(vertices);

    if (!vec_normals.empty())
        Mat(static_cast<int>(vec_normals.size()), 1, CV_32FC3, &vec_normals[0]).copyTo(normals);

    if (!vec_idices.empty())
    {
        Mat mat_indices(static_cast<int>(vec_idices.size()), static_cast<int>(vec_idices[0].size()), CV_32SC1);
        for (int i = 0; i < mat_indices.rows; ++i)
            for (int j = 0; j < mat_indices.cols; ++j)
                mat_indices.at<int32_t>(i, j) = vec_idices.at(i).at(j);
        mat_indices.copyTo(indices);
    }
#else // OPENCV_HAVE_FILESYSTEM_SUPPORT
    CV_UNUSED(filename);
    CV_UNUSED(vertices);
    CV_UNUSED(normals);
    CV_Error(Error::StsNotImplemented, "File system support is disabled in this OpenCV build!");
#endif
}

void saveMesh(const String &filename, InputArray vertices, InputArray normals, InputArray indices)
{
#if OPENCV_HAVE_FILESYSTEM_SUPPORT
    CV_Assert(!vertices.empty());

    pc::PointCloudEncoder encoder;

    encoder = pc::findEncoder(filename);

    encoder->setDestination(filename);

    std::vector<Point3f> vec_vertices(vertices.getMat());
    std::vector<Point3f> vec_normals;
    if (!normals.empty()){
        vec_normals = normals.getMat();
    }
    std::vector<std::vector<int32_t>> vec_idices;
    Mat mat_indices(indices.getMat());
    for (int i = 0; i < mat_indices.rows; ++i)
    {
        std::vector<int32_t> faceIndices;
        for (int j = 0; j < mat_indices.cols; ++j)
            faceIndices.push_back(mat_indices.at<int32_t>(i, j));
        vec_idices.push_back(faceIndices);
    }

    encoder->writeData(vec_vertices, vec_normals, vec_idices);
#else // OPENCV_HAVE_FILESYSTEM_SUPPORT
    CV_UNUSED(filename);
    CV_UNUSED(vertices);
    CV_UNUSED(normals);
    CV_Error(Error::StsNotImplemented, "File system support is disabled in this OpenCV build!");
#endif
}

}
