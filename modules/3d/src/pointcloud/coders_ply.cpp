// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "coders_ply.hpp"
#include "utils.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

namespace cv
{
namespace pc
{

void PlyDecoder::readData(std::vector<Point3f> &points, std::vector<Point3f> &normals, std::vector<std::vector<int32_t>> &indices)
{
    points.clear();
    normals.clear();
    CV_UNUSED(indices);

    std::ifstream file(m_filename, std::ios::binary);
    parseHeader(file);

    parseBody(file, points, normals);
}

void PlyDecoder::parseHeader(std::ifstream &file)
{
    std::string s;
    std::getline(file, s);
    if (trimSpaces(s) != "ply")
    {
        CV_Error(Error::StsError, "Fail file is not a PLY format");
    }
    std::getline(file, s);
    auto splitArr = split(s, ' ');
    if (splitArr[0] != "format")
    {
        CV_Error(Error::StsError, "Fail no format");
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
        CV_Error(Error::StsError, "Fail ply file format is not supported");
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
                if (splitArrElem[2] == "x")
                {
                    CV_Assert(splitArrElem[1] == "float");
                }
                if (splitArrElem[2] == "red")
                {
                    CV_Assert(splitArrElem[1] == "float");
                    m_hasColour = true;
                }
                if (splitArrElem[2] == "nx")
                {
                    CV_Assert(splitArrElem[1] == "float");
                    m_hasNormal = true;
                }
            }
            continue;
        }
        if (startsWith(s, "end_header"))
            break;

    }
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

void PlyDecoder::parseBody(std::ifstream &file, std::vector<Point3f> &points, std::vector<Point3f> &normals)
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
            readNext<float>(file, m_inputDataFormat);
            readNext<float>(file, m_inputDataFormat);
            readNext<float>(file, m_inputDataFormat);
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

void PlyEncoder::writeData(std::vector<Point3f> &points, std::vector<Point3f> &normals, std::vector<std::vector<int32_t>> &indices)
{
    CV_UNUSED(indices);
    std::ofstream file(m_filename, std::ios::binary);
    if (!file)
    {
        CV_Error(Error::StsError, "Impossible to open the file !\n");
    }
    file << "ply" << std::endl;
    file << "format ascii 1.0" << std::endl;
    file << "comment created by OpenCV" << std::endl;
    file << "element vertex " << points.size() << std::endl;

    file << "property float x" << std::endl;
    file << "property float y" << std::endl;
    file << "property float z" << std::endl;

    file << "end_header" << std::endl;

    if (normals.size() != 0)
    {
        file << "property float nx" << std::endl;
        file << "property float ny" << std::endl;
        file << "property float nz" << std::endl;
    }

    bool isNormals = (normals.size() != 0);

    for (size_t i = 0; i < points.size(); i++)
    {
        file << points[i].x << " " << points[i].y << " " << points[i].z;
        if (isNormals)
        {
            file << normals[i].x << " " << normals[i].y << " " << normals[i].z;
        }
        file << std::endl;
    }
    file.close();
}

}

}
