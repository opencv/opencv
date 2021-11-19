// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"

#include "io_base.hpp"
#include "io_obj.hpp"
#include "io_ply.hpp"
#include "utils.hpp"
#include "opencv2/core/utils/filesystem.private.hpp"
#include <opencv2/core/utils/logger.hpp>

#include <memory>

namespace cv {

#if OPENCV_HAVE_FILESYSTEM_SUPPORT

static PointCloudDecoder findDecoder(const String &filename)
{
    auto file_ext = getExtension(filename);
    if (file_ext == "obj" || file_ext == "OBJ")
    {
        return std::unique_ptr<ObjDecoder>(new ObjDecoder());
    }
    if (file_ext == "ply" || file_ext == "PLY")
    {
        return std::unique_ptr<PlyDecoder>(new PlyDecoder());
    }

    return nullptr;
}

static PointCloudEncoder findEncoder(const String &filename)
{
    auto file_ext = getExtension(filename);
    if (file_ext == "obj" || file_ext == "OBJ")
    {
        return std::unique_ptr<ObjEncoder>(new ObjEncoder());
    }
    if (file_ext == "ply" || file_ext == "PLY")
    {
        return std::unique_ptr<PlyEncoder>(new PlyEncoder());
    }

    return nullptr;
}

#endif

void loadPointCloud(const String &filename, OutputArray vertices, OutputArray normals)
{
#if OPENCV_HAVE_FILESYSTEM_SUPPORT
    auto decoder = findDecoder(filename);
    if (!decoder) {
        String file_ext = getExtension(filename);
        CV_LOG_ERROR(NULL, "File extension '" << file_ext << "' is not supported");
        return;
    }

    decoder->setSource(filename);

    std::vector<Point3f> vec_vertices;
    std::vector<Point3f> vec_normals;

    decoder->readData(vec_vertices, vec_normals);

    if (!vec_vertices.empty())
        Mat(static_cast<int>(vec_vertices.size()), 1, CV_32FC3, &vec_vertices[0]).copyTo(vertices);

    if (!vec_normals.empty() && normals.needed())
        Mat(static_cast<int>(vec_normals.size()), 1, CV_32FC3, &vec_normals[0]).copyTo(normals);
#else // OPENCV_HAVE_FILESYSTEM_SUPPORT
    CV_UNUSED(filename);
    CV_UNUSED(vertices);
    CV_UNUSED(normals);
    CV_LOG_WARNING(NULL, "File system support is disabled in this OpenCV build!");
#endif
}

void savePointCloud(const String &filename, InputArray vertices, InputArray normals)
{
#if OPENCV_HAVE_FILESYSTEM_SUPPORT
    if (vertices.empty()) {
        CV_LOG_WARNING(NULL, "Have no vertices to save");
        return;
    };

    auto encoder = findEncoder(filename);
    if (!encoder) {
        String file_ext = getExtension(filename);
        CV_LOG_ERROR(NULL, "File extension '" << file_ext << "' is not supported");
        return;
    }

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
    CV_LOG_WARNING(NULL, "File system support is disabled in this OpenCV build!");
#endif
}

void loadMesh(const String &filename, OutputArray vertices, OutputArray normals, OutputArrayOfArrays indices)
{
#if OPENCV_HAVE_FILESYSTEM_SUPPORT
    PointCloudDecoder decoder = findDecoder(filename);
    String file_ext = getExtension(filename);
    if (!decoder || (file_ext != "obj" && file_ext != "OBJ")) {
        CV_LOG_ERROR(NULL, "File extension '" << file_ext << "' is not supported");
        return;
    }

    decoder->setSource(filename);

    std::vector<Point3f> vec_vertices;
    std::vector<Point3f> vec_normals;
    std::vector<std::vector<int32_t>> vec_indices;

    decoder->readData(vec_vertices, vec_normals, vec_indices);

    if (!vec_vertices.empty()) {
        Mat(1, static_cast<int>(vec_vertices.size()), CV_32FC3, vec_vertices.data()).copyTo(vertices);
    }

    if (!vec_normals.empty()) {
        Mat(1, static_cast<int>(vec_normals.size()), CV_32FC3, vec_normals.data()).copyTo(normals);
    }

    if (!vec_indices.empty()) {
        std::vector<std::vector<int32_t>>& vec = *(std::vector<std::vector<int32_t>>*)indices.getObj();
        vec.resize(vec_indices.size());
        for (size_t i = 0; i < vec_indices.size(); ++i) {
            Mat(1, static_cast<int>(vec_indices[i].size()), CV_32SC1, vec_indices[i].data()).copyTo(vec[i]);
        }
    }

#else // OPENCV_HAVE_FILESYSTEM_SUPPORT
    CV_UNUSED(filename);
    CV_UNUSED(vertices);
    CV_UNUSED(normals);
    CV_LOG_WARNING(NULL, "File system support is disabled in this OpenCV build!");
#endif
}

void saveMesh(const String &filename, InputArray vertices, InputArray normals, InputArrayOfArrays indices)
{
#if OPENCV_HAVE_FILESYSTEM_SUPPORT
    if (vertices.empty()) {
        CV_LOG_WARNING(NULL, "Have no vertices to save");
        return;
    }

    auto encoder = findEncoder(filename);
    String file_ext = getExtension(filename);
    if (!encoder || (file_ext != "obj" && file_ext != "OBJ")) {
        CV_LOG_ERROR(NULL, "File extension '" << file_ext << "' is not supported");
        return;
    }

    encoder->setDestination(filename);

    std::vector<Point3f> vec_vertices(vertices.getMat());
    std::vector<Point3f> vec_normals;
    if (!normals.empty()){
        vec_normals = normals.getMat();
    }

    std::vector<Mat> mat_indices;
    indices.getMatVector(mat_indices);
    std::vector<std::vector<int32_t>> vec_indices(mat_indices.size());

    for (size_t i = 0; i < mat_indices.size(); ++i) {
        mat_indices[i].copyTo(vec_indices[i]);
    }

    encoder->writeData(vec_vertices, vec_normals, vec_indices);

#else // OPENCV_HAVE_FILESYSTEM_SUPPORT
    CV_UNUSED(filename);
    CV_UNUSED(vertices);
    CV_UNUSED(normals);
    CV_LOG_WARNING(NULL, "File system support is disabled in this OpenCV build!");
#endif

}

}/* namespace cv */
