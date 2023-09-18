// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "io_ply.hpp"
#include "utils.hpp"
#include <opencv2/core/utils/logger.hpp>
#include <fstream>
#include <string>

namespace cv {

void PlyDecoder::readData(std::vector<Point3f> &points, std::vector<Point3f> &normals, std::vector<Point3_<uchar>> &rgb, std::vector<std::vector<int32_t>> &indices)
{
    points.clear();
    normals.clear();
    rgb.clear();
    CV_UNUSED(indices);

    std::ifstream file(m_filename, std::ios::binary);

    if (!file.is_open()) {
        CV_LOG_ERROR(NULL, "File " + m_filename + " does not exist.");
        return;
    }

    if (parseHeader(file))
    {
        parseBody(file, points, normals, rgb);
    }
}

bool PlyDecoder::parseHeader(std::ifstream &file)
{
    std::string s;
    std::getline(file, s);
    if (trimSpaces(s) != "ply")
    {
        CV_LOG_ERROR(NULL, "Provided file is not in PLY format");
        return false;
    }
    std::getline(file, s);
    auto splitArr = split(s, ' ');
    if (splitArr[0] != "format")
    {
        CV_LOG_ERROR(NULL, "Provided file doesn't have format");
        return false;
    }
    if (splitArr[1] == "ascii")
    {
        m_inputDataFormat = DataFormat::ASCII;
    }
    else if (splitArr[1] == "binary_little_endian")
    {
        m_inputDataFormat = DataFormat::BinaryLittleEndian;
    }
    else if (splitArr[1] == "binary_big_endian")
    {
        m_inputDataFormat = DataFormat::BinaryBigEndian;
    }
    else
    {
        CV_LOG_ERROR(NULL, "Provided PLY file format is not supported");
        return false;
    }

    bool onVertexRead = false;
    while (std::getline(file, s))
    {
        if (startsWith(s, "element"))
        {
            auto splitArrElem = split(s, ' ');
            if (splitArrElem[1] == "vertex")
            {
                onVertexRead = true;
                std::istringstream iss(splitArrElem[2]);
                iss >> m_vertexCount;
            }
            else
            {
                onVertexRead = false;
            }
            continue;
        }
        if (startsWith(s, "property"))
        {
            if (onVertexRead)
            {
                auto splitArrElem = split(s, ' ');
                if (splitArrElem[2] == "x" || splitArrElem[2] == "y" || splitArrElem[2] == "z")
                {
                    if (splitArrElem[1] != "float") {
                        CV_LOG_ERROR(NULL, "Provided property '" << splitArrElem[2] << "' with format '" << splitArrElem[1]
                        << "' is not supported");
                        return false;
                    }
                }
                if (splitArrElem[2] == "red" || splitArrElem[2] == "green" || splitArrElem[2] == "blue")
                {
                    if (splitArrElem[1] != "uchar") {
                        CV_LOG_ERROR(NULL, "Provided property '" << splitArrElem[2] << "' with format '" << splitArrElem[1]
                        << "' is not supported");
                        return false;
                    }
                    m_hasColour = true;
                }
                if (splitArrElem[2] == "nx")
                {
                    if (splitArrElem[1] != "float") {
                        CV_LOG_ERROR(NULL, "Provided property '" << splitArrElem[2] << "' with format '" << splitArrElem[1]
                        << "' is not supported");
                        return false;
                    }
                    m_hasNormal = true;
                }
            }
            continue;
        }
        if (startsWith(s, "end_header"))
            break;

    }
    return true;
}

template <typename T>
T readNext(std::ifstream &file, DataFormat format)
{
    T val;
    if (format == DataFormat::ASCII)
    {
        file >> val;
        return val;
    }
    file.read((char *)&val, sizeof(T));
#ifdef WORDS_BIGENDIAN
    if (!(format == DataFormat::BinaryBigEndian) )
    {
        swapEndian<T>(val);
    }
#else
    if (format == DataFormat::BinaryBigEndian)
    {
        swapEndian<T>(val);
    }
#endif
    return val;
}

void PlyDecoder::parseBody(std::ifstream &file, std::vector<Point3f> &points, std::vector<Point3f> &normals, std::vector<Point3_<uchar>> &rgb)
{
    points.reserve(m_vertexCount);
    if (m_hasNormal)
    {
        normals.reserve(m_vertexCount);
    }
    for (size_t i = 0; i < m_vertexCount; i++)
    {
        Point3f vertex;
        vertex.x = readNext<float>(file, m_inputDataFormat);
        vertex.y = readNext<float>(file, m_inputDataFormat);
        vertex.z = readNext<float>(file, m_inputDataFormat);
        points.push_back(vertex);
        if (m_hasColour)
        {
            Point3_<uchar> colour;
            colour.x = readNext<int>(file, m_inputDataFormat) & 0xff;
            colour.y = readNext<int>(file, m_inputDataFormat) & 0xff;
            colour.z = readNext<int>(file, m_inputDataFormat) & 0xff;
            rgb.push_back(colour);
        }
        if (m_hasNormal)
        {
            Point3f normal;
            normal.x = readNext<float>(file, m_inputDataFormat);
            normal.y = readNext<float>(file, m_inputDataFormat);
            normal.z = readNext<float>(file, m_inputDataFormat);
            normals.push_back(normal);
        }
    }
}

void PlyEncoder::writeData(const std::vector<Point3f> &points, const std::vector<Point3f> &normals, const std::vector<Point3_<uchar>> &rgb, const std::vector<std::vector<int32_t>> &indices)
{
    CV_UNUSED(indices);
    std::ofstream file(m_filename, std::ios::binary);
    if (!file) {
        CV_LOG_ERROR(NULL, "Impossible to open the file: " << m_filename);
        return;
    }
    bool hasNormals = !normals.empty(), hasColor = !rgb.empty();

    file << "ply" << std::endl;
    file << "format ascii 1.0" << std::endl;
    file << "comment created by OpenCV" << std::endl;
    file << "element vertex " << points.size() << std::endl;

    file << "property float x" << std::endl;
    file << "property float y" << std::endl;
    file << "property float z" << std::endl;

    if(hasColor) {
        file << "property uchar red" << std::endl;
        file << "property uchar green" << std::endl;
        file << "property uchar blue" << std::endl;
    }

    if (hasNormals)
    {
        file << "property float nx" << std::endl;
        file << "property float ny" << std::endl;
        file << "property float nz" << std::endl;
    }

    file << "end_header" << std::endl;


    for (size_t i = 0; i < points.size(); i++)
    {
        file << points[i].x << " " << points[i].y << " " << points[i].z;
        if (hasColor) {
            file << " " << static_cast<int>(rgb[i].x) << " " << static_cast<int>(rgb[i].y) << " " << static_cast<int>(rgb[i].z);
        }
        if (hasNormals) {
            file << " " << normals[i].x << " " << normals[i].y << " " << normals[i].z;
        }
        file << std::endl;
    }
    file.close();
}

} /* namespace cv */
