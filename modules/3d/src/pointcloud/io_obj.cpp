// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "io_obj.hpp"
#include <fstream>
#include <opencv2/core/utils/logger.hpp>
#include "utils.hpp"

namespace cv {

std::unordered_set<std::string> ObjDecoder::m_unsupportedKeys;

void ObjDecoder::readData(std::vector<Point3f> &points, std::vector<Point3f> &normals, std::vector<Point3_<uchar>> &rgb, std::vector<std::vector<int32_t>> &indices)
{
    points.clear();
    normals.clear();
    rgb.clear();
    indices.clear();

    std::ifstream file(m_filename, std::ios::binary);
    if (!file)
    {
        CV_LOG_ERROR(NULL, "Impossible to open the file: " << m_filename);
        return;
    }
    std::string s;

    while (!file.eof())
    {
        std::getline(file, s);
        if (s.empty())
            continue;
        std::stringstream ss(s);
        std::string key;
        ss >> key;

        if (key == "#")
            continue;
        else if (key == "v")
        {
            // (x, y, z, [w], [r, g, b])
            auto splitArr = split(s, ' ');
            if (splitArr.size() <= 3)
            {
                CV_LOG_ERROR(NULL, "Vertex should have at least 3 coordinate values.");
                return;
            }
            Point3f vertex;
            ss >> vertex.x >> vertex.y >> vertex.z;
            points.push_back(vertex);
            if (splitArr.size() == 5 || splitArr.size() == 8)
            {
                float w;
                ss >> w;
                CV_UNUSED(w);
            }
            if (splitArr.size() >= 7)
            {
                Point3f color;
                if (ss.rdbuf()->in_avail() != 0) {
                    Point3_<uchar> uc_color;
                    ss >> color.x >> color.y >> color.z;
                    uc_color.x = static_cast<uchar>(std::round(255.f * color.x));
                    uc_color.y = static_cast<uchar>(std::round(255.f * color.y));
                    uc_color.z = static_cast<uchar>(std::round(255.f * color.z));
                    rgb.push_back(uc_color);
                }
            }
        }
        else if (key == "vn")
        {
            Point3f normal;
            ss >> normal.x >> normal.y >> normal.z;
            normals.push_back(normal);
        }
        else if (key == "f")
        {
            std::vector<int> vertexInd;
            auto tokens = split(s, ' ');
            for (size_t i = 1; i < tokens.size(); i++)
            {
                auto vertexinfo = split(tokens[i], '/');
                int index;
                std::stringstream vs(vertexinfo[0]);
                vs >> index;
                vertexInd.push_back(index - 1);
            }
            indices.push_back(vertexInd);
        }
        else
        {
            if (m_unsupportedKeys.find(key) == m_unsupportedKeys.end()) {
                m_unsupportedKeys.insert(key);
                CV_LOG_WARNING(NULL, "Key " << key << " not supported");
            }
        }
    }

    file.close();
}

void ObjEncoder::writeData(const std::vector<Point3f> &points, const std::vector<Point3f> &normals, const std::vector<Point3_<uchar>> &rgb, const std::vector<std::vector<int32_t>> &indices)
{
    std::ofstream file(m_filename, std::ios::binary);
    if (!file) {
        CV_LOG_ERROR(NULL, "Impossible to open the file: " << m_filename);
        return;
    }

    if (!rgb.empty() && rgb.size() != points.size()) {
        CV_LOG_ERROR(NULL, "Vertices and Colors have different size.");
        return;
    }

    file << "# OBJ file writer" << std::endl;
    file << "o Point_Cloud" << std::endl;

    for (size_t i = 0; i < points.size(); ++i)
    {
        file << "v " << points[i].x << " " << points[i].y << " " << points[i].z;
        if (!rgb.empty()) {
            file << " " << static_cast<float>(rgb[i].x) / 255.f << " " <<
                static_cast<float>(rgb[i].y) / 255.f  << " " << static_cast<float>(rgb[i].z) / 255.f;
        }
        file << std::endl;
    }

    for (const auto& normal : normals)
    {
        file << "vn " << normal.x << " " << normal.y << " " << normal.z << std::endl;
    }

    for (const auto& faceIndices : indices)
    {
        file << "f ";
        for (const auto& index : faceIndices)
        {
            file << index + 1 << " ";
        }
        file << std::endl;
    }

    file.close();
}

} /* namespace cv */
