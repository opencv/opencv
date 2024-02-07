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

void loadPointCloud(const String &filename, OutputArray vertices, OutputArray normals, OutputArray rgb)
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
    std::vector<Point3_<uchar>> vec_rgb;

    decoder->readData(vec_vertices, vec_normals, vec_rgb);

    if (!vec_vertices.empty())
        Mat(static_cast<int>(vec_vertices.size()), 1, CV_32FC3, &vec_vertices[0]).copyTo(vertices);

    if (!vec_normals.empty() && normals.needed())
        Mat(static_cast<int>(vec_normals.size()), 1, CV_32FC3, &vec_normals[0]).copyTo(normals);

    if (!vec_rgb.empty() && rgb.needed())
        Mat(static_cast<int>(vec_rgb.size()), 1, CV_8UC3, &vec_rgb[0]).copyTo(rgb);

#else // OPENCV_HAVE_FILESYSTEM_SUPPORT
    CV_UNUSED(filename);
    CV_UNUSED(vertices);
    CV_UNUSED(normals);
    CV_LOG_WARNING(NULL, "File system support is disabled in this OpenCV build!");
#endif
}

void savePointCloud(const String &filename, InputArray vertices, InputArray normals, InputArray rgb)
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
    std::vector<Point3_<uchar>> vec_rgb;

    if (!normals.empty()){
        vec_normals = normals.getMat();
    }

    if (!rgb.empty()){
        vec_rgb = rgb.getMat();
    }
    encoder->writeData(vec_vertices, vec_normals, vec_rgb);

#else // OPENCV_HAVE_FILESYSTEM_SUPPORT
    CV_UNUSED(filename);
    CV_UNUSED(vertices);
    CV_UNUSED(normals);
    CV_LOG_WARNING(NULL, "File system support is disabled in this OpenCV build!");
#endif
}

void loadMesh(const String &filename, OutputArray vertices, OutputArray normals, OutputArrayOfArrays indices)
{
    loadMesh(filename, vertices, normals, noArray(), indices);
}

void loadMesh(const String &filename, OutputArray vertices, OutputArrayOfArrays indices)
{
    loadMesh(filename, vertices, noArray(), noArray(), indices);
}

void loadMesh(const String &filename, OutputArray vertices, OutputArray normals, OutputArray colors, OutputArrayOfArrays indices)
{
#if OPENCV_HAVE_FILESYSTEM_SUPPORT
    CV_Assert(vertices.needed());
    CV_Assert(indices.needed());

    PointCloudDecoder decoder = findDecoder(filename);
    String file_ext = getExtension(filename);
    if (!decoder) {
        CV_LOG_ERROR(NULL, "File extension '" << file_ext << "' is not supported");
        return;
    }

    decoder->setSource(filename);

    std::vector<Point3f> vec_vertices;
    std::vector<Point3f> vec_normals;
    std::vector<Point3_<uchar>> vec_rgb;
    std::vector<std::vector<int32_t>> vec_indices;

    decoder->readData(vec_vertices, vec_normals, vec_rgb, vec_indices);

    if (!vec_vertices.empty()) {
        Mat(1, static_cast<int>(vec_vertices.size()), CV_32FC3, vec_vertices.data()).copyTo(vertices);
    }

    if (!vec_normals.empty()) {
        Mat(1, static_cast<int>(vec_normals.size()), CV_32FC3, vec_normals.data()).copyTo(normals);
    }

    if (colors.needed() && !vec_rgb.empty())
    {
        Mat(1, static_cast<int>(vec_rgb.size()), CV_8UC3, vec_rgb.data()).convertTo(colors, CV_32F, (1.0/255.0));
    }
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
    CV_UNUSED(colors);
    CV_LOG_WARNING(NULL, "File system support is disabled in this OpenCV build!");
#endif
}

void saveMesh(const String &filename, InputArray vertices, InputArray normals, InputArrayOfArrays indices)
{
    saveMesh(filename, vertices, normals, noArray(), indices);
}

void saveMesh(const String &filename, InputArray vertices, InputArrayOfArrays indices)
{
    saveMesh(filename, vertices, noArray(), noArray(), indices);
}

void saveMesh(const String &filename, InputArray vertices, InputArray normals, InputArray colors, InputArrayOfArrays indices)
{
#if OPENCV_HAVE_FILESYSTEM_SUPPORT
    if (vertices.empty()) {
        CV_LOG_WARNING(NULL, "Have no vertices to save");
        return;
    }

    auto encoder = findEncoder(filename);
    String file_ext = getExtension(filename);
    if (!encoder) {
        CV_LOG_ERROR(NULL, "File extension '" << file_ext << "' is not supported");
        return;
    }

    encoder->setDestination(filename);

    std::vector<Point3f> vec_vertices(vertices.getMat());
    std::vector<Point3f> vec_normals;
    std::vector<Point3_<uchar>> vec_rgb;
    if (!normals.empty()){
        vec_normals = normals.getMat();
    }
    if (!colors.empty())
    {
        colors.getMat().convertTo(vec_rgb, CV_8U, 255.0);
    }

    std::vector<Mat> mat_indices;
    indices.getMatVector(mat_indices);
    std::vector<std::vector<int32_t>> vec_indices(mat_indices.size());

    for (size_t i = 0; i < mat_indices.size(); ++i) {
        mat_indices[i].copyTo(vec_indices[i]);
    }

    encoder->writeData(vec_vertices, vec_normals, vec_rgb, vec_indices);

#else // OPENCV_HAVE_FILESYSTEM_SUPPORT
    CV_UNUSED(filename);
    CV_UNUSED(vertices);
    CV_UNUSED(colors);
    CV_UNUSED(normals);
    CV_LOG_WARNING(NULL, "File system support is disabled in this OpenCV build!");
#endif

}

}/* namespace cv */
