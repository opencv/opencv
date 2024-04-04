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

    std::vector<Point3f> vec_vertices, vec_normals, vec_rgb;

    decoder->readData(vec_vertices, vec_normals, vec_rgb);

    if (!vec_vertices.empty())
        Mat(static_cast<int>(vec_vertices.size()), 1, CV_32FC3, vec_vertices.data()).copyTo(vertices);

    if (!vec_normals.empty() && normals.needed())
        Mat(static_cast<int>(vec_normals.size()), 1, CV_32FC3, vec_normals.data()).copyTo(normals);

    if (!vec_rgb.empty() && rgb.needed())
        Mat(static_cast<int>(vec_rgb.size()), 1, CV_32FC3, vec_rgb.data()).copyTo(rgb);

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

    std::vector<Point3f> vec_vertices(vertices.getMat()), vec_normals, vec_rgb;

    if (!normals.empty())
    {
        vec_normals = normals.getMat();
    }

    if (!rgb.empty())
    {
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

void loadMesh(const String &filename, OutputArray vertices, OutputArrayOfArrays indices,
              OutputArray normals, OutputArray colors, OutputArray texCoords)
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

    std::vector<Point3f> vec_vertices, vec_normals, vec_rgb;
    std::vector<std::vector<int32_t>> vec_indices;

    std::vector<Point3f> vec_texCoords;
    int nTexCoords;

    decoder->readData(vec_vertices, vec_normals, vec_rgb, vec_texCoords, nTexCoords, vec_indices, 0);

    if (!vec_vertices.empty())
    {
        Mat(1, static_cast<int>(vec_vertices.size()), CV_32FC3, vec_vertices.data()).copyTo(vertices);
    }

    if (normals.needed() && !vec_normals.empty())
    {
        Mat(1, static_cast<int>(vec_normals.size()), CV_32FC3, vec_normals.data()).copyTo(normals);
    }

    if (colors.needed() && !vec_rgb.empty())
    {
        Mat(1, static_cast<int>(vec_rgb.size()), CV_32FC3, vec_rgb.data()).copyTo(colors);
    }

    if (!vec_indices.empty())
    {
        _InputArray::KindFlag kind = indices.kind();
        int vecsz = (int)vec_indices.size();
        if (kind == _InputArray::KindFlag::STD_VECTOR_VECTOR)
        {
            CV_Assert(indices.depth() == CV_32S);
            std::vector<std::vector<int32_t>>& vec = *(std::vector<std::vector<int32_t>>*)indices.getObj();
            vec.resize(vecsz);
            for (int i = 0; i < vecsz; ++i)
            {
                Mat(1, static_cast<int>(vec_indices[i].size()), CV_32SC1, vec_indices[i].data()).copyTo(vec[i]);
            }
        }
        // std::array<Mat> has fixed size, unsupported
        else if (kind == _InputArray::KindFlag::STD_VECTOR_MAT)
        {
            indices.create(vecsz, 1, CV_32S);
            for (int i = 0; i < vecsz; i++)
            {
                std::vector<int> vi = vec_indices[i];
                indices.create(1, vi.size(), CV_32S, i);
                Mat(vi).copyTo(indices.getMat(i));
            }
        }
        else
        {
            indices.create(1, (int)vec_indices.size(), CV_32SC3);
            std::vector<Vec3i>& vec = *(std::vector<Vec3i>*)indices.getObj();
            for (int i = 0; i < vecsz; ++i)
            {
                Vec3i tri;
                size_t sz = vec_indices[i].size();
                if (sz != 3)
                {
                    CV_Error(Error::StsBadArg, "Face contains " + std::to_string(sz) + " vertices, can not put it into 3-channel indices array");
                }
                else
                {
                    for (int j = 0; j < 3; j++)
                    {
                        tri[j] = vec_indices[i][j];
                    }
                }
                vec[i] = tri;
            }
        }
    }

    if (texCoords.needed())
    {
        int ch = texCoords.empty() ? 0 : texCoords.channels();
        Mat texMat = Mat(1, static_cast<int>(vec_texCoords.size()), CV_MAKETYPE(CV_32F, nTexCoords), vec_texCoords.data());
        if (ch == nTexCoords)
        {
            texMat.copyTo(texCoords);
        }
        else
        {
            Mat newTexMat;
            std::vector<Mat> varr;
            cv::split(texMat, varr);
            if (ch == 2 && nTexCoords == 3)
            {
                std::vector<Mat> marr = { varr[0], varr[1] };
                cv::merge(marr, newTexMat);
            }
            else if (ch == 3 && nTexCoords == 2)
            {
                std::vector<Mat> marr = { varr[0], varr[1], Mat::zeros(varr[0].size(), CV_32F) };
                cv::merge(marr, newTexMat);
            }
            else
            {
                newTexMat = texMat;
            }
            newTexMat.copyTo(texCoords);
        }
    }

#else // OPENCV_HAVE_FILESYSTEM_SUPPORT
    CV_UNUSED(filename);
    CV_UNUSED(vertices);
    CV_UNUSED(normals);
    CV_UNUSED(colors);
    CV_UNUSED(indices);
    CV_LOG_WARNING(NULL, "File system support is disabled in this OpenCV build!");
#endif
}

void saveMesh(const String &filename, InputArray vertices, InputArrayOfArrays indices,
              InputArray normals, InputArray colors, InputArray texCoords)
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

    std::vector<Point3f> vec_vertices(vertices.getMat()), vec_normals, vec_rgb;
    if (!normals.empty())
    {
        vec_normals = normals.getMat();
    }

    if (!colors.empty())
    {
        vec_rgb = colors.getMat();
    }

    std::vector<std::vector<int32_t>> vec_indices;
    CV_Assert(indices.depth() == CV_32S);
    if (indices.kind() == _InputArray::KindFlag::STD_VECTOR_VECTOR)
    {
        std::vector<Mat> mat_indices;
        indices.getMatVector(mat_indices);
        vec_indices.resize(mat_indices.size());
        for (size_t i = 0; i < mat_indices.size(); ++i)
        {
            mat_indices[i].copyTo(vec_indices[i]);
        }
    }
    else
    {
        CV_Assert(indices.channels() == 3);
        std::vector<Vec3i>& vec = *(std::vector<Vec3i>*)indices.getObj();
        vec_indices.resize(vec.size());
        for (size_t i = 0; i < vec.size(); ++i)
        {
            for (int j = 0; j < 3; j++)
            {
                vec_indices[i].push_back(vec[i][j]);
            }
        }
    }

    std::vector<Point3f> vec_texCoords;
    int nTexCoords = 0;
    if (!texCoords.empty())
    {
        nTexCoords = texCoords.channels();
    }
    if (nTexCoords == 2)
    {
        std::vector<Point2f> vec2_texCoords;
        texCoords.copyTo(vec2_texCoords);
        for (size_t i = 0; i < vec2_texCoords.size(); i++)
        {
            Point2f p = vec2_texCoords[i];
            vec_texCoords.push_back({p.x, p.y, 0});
        }
    }
    if (nTexCoords == 3)
    {
        texCoords.copyTo(vec_texCoords);
    }

    encoder->writeData(vec_vertices, vec_normals, vec_rgb, vec_texCoords, nTexCoords, vec_indices);

#else // OPENCV_HAVE_FILESYSTEM_SUPPORT
    CV_UNUSED(filename);
    CV_UNUSED(vertices);
    CV_UNUSED(colors);
    CV_UNUSED(normals);
    CV_UNUSED(indices);
    CV_LOG_WARNING(NULL, "File system support is disabled in this OpenCV build!");
#endif

}

}/* namespace cv */
