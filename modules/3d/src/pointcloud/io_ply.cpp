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
    indices.clear();

    std::ifstream file(m_filename, std::ios::binary);
    if (parseHeader(file))
    {
        parseBody(file, points, normals, rgb, indices);
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

    enum ReadElement
    {
        READ_OTHER  = 0,
        READ_VERTEX = 1,
        READ_FACE   = 2
    };
    ReadElement elemRead = READ_OTHER;
    while (std::getline(file, s))
    {
        if (startsWith(s, "element"))
        {
            std::vector<std::string> splitArrElem = split(s, ' ');
            std::string elemName = splitArrElem.at(1);
            if (elemName == "vertex")
            {
                elemRead = READ_VERTEX;
                if(splitArrElem.size() != 3)
                {
                    CV_LOG_ERROR(NULL, "Vertex element description has " << splitArrElem.size()
                                 << " words instead of 3");
                    return false;
                }
                std::istringstream iss(splitArrElem[2]);
                iss >> m_vertexCount;
            }
            else if (elemName == "face")
            {
                elemRead = READ_FACE;
                if(splitArrElem.size() != 3)
                {
                    CV_LOG_ERROR(NULL, "Face element description has " << splitArrElem.size()
                                 << " words instead of 3");
                    return false;
                }
                std::istringstream iss(splitArrElem[2]);
                iss >> m_faceCount;
            }
            else
            {
                elemRead = READ_OTHER;
            }
            continue;
        }
        if (startsWith(s, "property"))
        {
            if (elemRead == READ_VERTEX)
            {
                auto splitArrElem = split(s, ' ');
                if (splitArrElem.size() < 3)
                {
                    CV_LOG_ERROR(NULL, "Vertex property has " << splitArrElem.size()
                                 << " words instead of at least 3");
                    return false;
                }
                std::string propType = splitArrElem[1];
                std::string propName = splitArrElem[2];
                if (propName == "x" || propName == "y" || propName == "z")
                {
                    if (propType != "float")
                    {
                        CV_LOG_ERROR(NULL, "Provided property '" << propName << "' with format '" << propType
                                     << "' is not supported");
                        return false;
                    }
                }
                if (propName == "red" || propName == "green" || propName == "blue")
                {
                    if (propType != "uchar")
                    {
                        CV_LOG_ERROR(NULL, "Provided property '" << propName << "' with format '" << propType
                                     << "' is not supported");
                        return false;
                    }
                    m_hasColour = true;
                }
                if (propName == "nx")
                {
                    if (propType != "float")
                    {
                        CV_LOG_ERROR(NULL, "Provided property '" << propName << "' with format '" << propType
                                     << "' is not supported");
                        return false;
                    }
                    m_hasNormal = true;
                }
                //TODO: skip unknown data types
            }
            else if (elemRead == READ_FACE)
            {
                std::vector<std::string> splitArrElem = split(s, ' ');
                if (splitArrElem.size() < 5)
                {
                    CV_LOG_ERROR(NULL, "Face property has " << splitArrElem.size()
                                 << " words instead of at least 5");
                    return false;
                }
                std::string propName = splitArrElem[1];
                if (propName != "list")
                {
                    CV_LOG_ERROR(NULL, "Face property is " << propName
                                 << " instead of \"list\"");
                    return false;
                }
                std::string amtTypeString = splitArrElem[2];
                if (amtTypeString != "uchar")
                {
                    CV_LOG_ERROR(NULL, "Face property is " << amtTypeString
                                 << " instead of \"uchar\"");
                    return false;
                }
                std::string idxTypeString = splitArrElem[3];
                if (idxTypeString != "int" && idxTypeString != "uint")
                {
                    CV_LOG_ERROR(NULL, "Face property is " << idxTypeString
                                 << " instead of \"int\" or \"uint\"");
                    return false;
                }
                std::string propTypeName = splitArrElem[4];
                if (propTypeName != "vertex_indices" && propTypeName != "vertex_index")
                {
                    CV_LOG_ERROR(NULL, "Face property is " << propTypeName
                                 << " instead of \"vertex_index\" or \"vertex_indices\"");
                    return false;
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

template <>
uchar readNext<uchar>(std::ifstream &file, DataFormat format)
{
    int val {0};
    if (format == DataFormat::ASCII)
    {
        file >> val;
    }
    else
    {
        file.read((char *)&val, sizeof(uchar));
    }
    return (uchar)val;
}

void PlyDecoder::parseBody(std::ifstream &file, std::vector<Point3f> &points, std::vector<Point3f> &normals, std::vector<Point3_<uchar>> &rgb,
                           std::vector<std::vector<int32_t>> &indices)
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

    indices.reserve(m_faceCount);
    for (size_t i = 0; i < m_faceCount; i++)
    {
        // PLY can have faces with >3 vertices in TRIANGLE_FAN format
        // in this case we load them as separate triangles
        size_t nVerts = readNext<uchar>(file, m_inputDataFormat);
        if (nVerts < 3)
        {
            CV_LOG_ERROR(NULL, "Face should have at least 3 vertices but has " << nVerts);
            return;
        }
        int vert1 = readNext<int>(file, m_inputDataFormat);
        int vert2 = readNext<int>(file, m_inputDataFormat);
        for (size_t j = 2; j < nVerts; j++)
        {
            int vert3 = readNext<int>(file, m_inputDataFormat);
            indices.push_back({vert1, vert2, vert3});
            vert2 = vert3;
        }
    }
}

void PlyEncoder::writeData(const std::vector<Point3f> &points, const std::vector<Point3f> &normals, const std::vector<Point3_<uchar>> &rgb, const std::vector<std::vector<int32_t>> &indices)
{
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

    if (!indices.empty())
    {
        file << "element face " << indices.size() << std::endl;
        file << "property list uchar int vertex_indices" << std::endl;
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

    for (const auto& faceIndices : indices)
    {
        file << faceIndices.size();
        for (const auto& index : faceIndices)
        {
            file << " " << index;
        }
        file << std::endl;
    }
    file.close();
}

} /* namespace cv */
