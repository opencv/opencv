// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "io_ply.hpp"
#include "utils.hpp"
#include <opencv2/core/utils/logger.hpp>
#include <fstream>
#include <string>
#include <iomanip>

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
    // "\r" symbols are not trimmed by default
    for (auto& e : splitArr)
    {
        e = trimSpaces(e);
    }
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

    const std::map<std::string, int> dataTypes =
    {
        { "char",   CV_8S  }, { "int8",    CV_8S  },
        { "uchar",  CV_8U  }, { "uint8",   CV_8U  },
        { "short",  CV_16S }, { "int16",   CV_16S },
        { "ushort", CV_16U }, { "uint16",  CV_16U },
        { "int",    CV_32S }, { "int32",   CV_32S },
        { "uint",   CV_32U }, { "uint32",  CV_32U },
        { "float",  CV_32F }, { "float32", CV_32F },
        { "double", CV_64F }, { "float64", CV_64F },
    };

    enum ReadElement
    {
        READ_OTHER  = 0,
        READ_VERTEX = 1,
        READ_FACE   = 2
    };
    ReadElement elemRead = READ_OTHER;
    m_vertexDescription = ElementDescription();
    m_faceDescription = ElementDescription();
    while (std::getline(file, s))
    {
        if (startsWith(s, "element"))
        {
            std::vector<std::string> splitArrElem = split(s, ' ');
            // "\r" symbols are not trimmed by default
            for (auto& e : splitArrElem)
            {
                e = trimSpaces(e);
            }
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
                iss >> m_vertexDescription.amount;
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
                iss >> m_faceDescription.amount;
            }
            else
            {
                elemRead = READ_OTHER;
            }
            continue;
        }
        if (startsWith(s, "property"))
        {
            Property property;
            std::string elName = (elemRead == READ_VERTEX) ? "Vertex" : "Face";
            std::vector<std::string> splitArrElem = split(s, ' ');
            // "\r" symbols are not trimmed by default
            for (auto& e : splitArrElem)
            {
                e = trimSpaces(e);
            }
            if (splitArrElem.size() < 3)
            {
                CV_LOG_ERROR(NULL, elName << " property has " << splitArrElem.size()
                             << " words instead of at least 3");
                return false;
            }
            std::string propType = splitArrElem[1];
            if (propType == "list")
            {
                property.isList = true;
                if (splitArrElem.size() < 5)
                {
                    CV_LOG_ERROR(NULL, elName << " property has " << splitArrElem.size()
                                 << " words instead of at least 5");
                    return false;
                }
                std::string amtTypeString = splitArrElem[2];
                if (dataTypes.count(amtTypeString) == 0)
                {
                    CV_LOG_ERROR(NULL, "Property type " << amtTypeString
                                 << " is not supported");
                    return false;
                }
                else
                {
                    property.counterType = dataTypes.at(amtTypeString);
                }
                std::string idxTypeString = splitArrElem[3];
                if (dataTypes.count(idxTypeString) == 0)
                {
                    CV_LOG_ERROR(NULL, "Property type " << idxTypeString
                                 << " is not supported");
                    return false;
                }
                else
                {
                    property.valType = dataTypes.at(idxTypeString);
                }

                property.name = splitArrElem[4];
            }
            else
            {
                property.isList = false;
                if (dataTypes.count(propType) == 0)
                {
                    CV_LOG_ERROR(NULL, "Property type " << propType
                                 << " is not supported");
                    return false;
                }
                else
                {
                    property.valType = dataTypes.at(propType);
                }
                property.name = splitArrElem[2];
            }

            if (elemRead == READ_VERTEX)
            {
                m_vertexDescription.properties.push_back(property);
            }
            else if (elemRead == READ_FACE)
            {
                m_faceDescription.properties.push_back(property);
            }

            continue;
        }
        if (startsWith(s, "end_header"))
            break;
    }

    bool good = true;
    m_vertexCount = m_vertexDescription.amount;
    std::map<std::string, int> amtProps;
    for (const auto& p : m_vertexDescription.properties)
    {
        bool known = false;
        if (p.name ==  "x" || p.name ==  "y" || p.name ==  "z")
        {
            known = true;
            if (p.valType != CV_32F)
            {
                CV_LOG_ERROR(NULL, "Vertex property " << p.name
                                                      << " should be float");
                good = false;
            }
        }
        if (p.name == "nx" || p.name == "ny" || p.name == "nz")
        {
            known = true;
            if (p.valType != CV_32F)
            {
                CV_LOG_ERROR(NULL, "Vertex property " << p.name
                                                      << " should be float");
                good = false;
            }
            m_hasNormal = true;
        }
        if (p.name ==  "red" || p.name ==  "green" || p.name ==  "blue")
        {
            known = true;
            if (p.valType != CV_8U)
            {
                CV_LOG_ERROR(NULL, "Vertex property " << p.name
                                                      << " should be uchar");
                good = false;
            }
            m_hasColour = true;
        }
        if (p.isList)
        {
            CV_LOG_ERROR(NULL, "List properties for vertices are not supported");
            good = false;
        }
        if (known)
        {
            amtProps[p.name]++;
        }
    }

    // check if we have no duplicates
    for (const auto& a : amtProps)
    {
        if (a.second > 1)
        {
            CV_LOG_ERROR(NULL, "Vertex property " << a.first << " is duplicated");
            good = false;
        }
    }
    const std::array<std::string, 3> vertKeys = {"x", "y", "z"};
    for (const std::string& c : vertKeys)
    {
        if (amtProps.count(c) == 0)
        {
            CV_LOG_ERROR(NULL, "Vertex property " << c << " is not presented in the file");
            good = false;
        }
    }

    m_faceCount = m_faceDescription.amount;
    int amtLists = 0;
    for (const auto& p : m_faceDescription.properties)
    {
        if (p.isList)
        {
            amtLists++;
            if (!(p.counterType == CV_8U && (p.valType == CV_32S || p.valType == CV_32U)))
            {
                CV_LOG_ERROR(NULL, "List property " << p.name
                             << " should have type uint8 for counter and uint32 for values");
                good = false;
            }
        }
    }
    if (amtLists > 1)
    {
        CV_LOG_ERROR(NULL, "Only 1 list property is supported per face");
        good = false;
    }

    return good;
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
    if (format == DataFormat::ASCII)
    {
        int val;
        file >> val;
        return (uchar)val;
    }
    uchar val;
    file.read((char *)&val, sizeof(uchar));
    // 1 byte does not have to be endian-swapped
    return val;
}

void PlyDecoder::parseBody(std::ifstream &file, std::vector<Point3f> &points, std::vector<Point3f> &normals, std::vector<Point3_<uchar>> &rgb,
                           std::vector<std::vector<int32_t>> &indices)
{
    points.reserve(m_vertexCount);
    if (m_hasColour)
    {
        rgb.reserve(m_vertexCount);
    }
    if (m_hasNormal)
    {
        normals.reserve(m_vertexCount);
    }

    // to avoid string matching at file loading
    size_t ivx = (size_t)-1, ivy = (size_t)-1, ivz = (size_t)-1, inx = (size_t)-1, iny = (size_t)-1, inz = (size_t)-1;
    size_t ir = (size_t)-1, ig = (size_t)-1, ib = (size_t)-1;
    for (size_t j = 0; j < m_vertexDescription.properties.size(); j++)
    {
        const auto& p = m_vertexDescription.properties[j];
        ivx = (p.name == "x") ? j : ivx;
        ivy = (p.name == "y") ? j : ivy;
        ivz = (p.name == "z") ? j : ivz;
        inx = (p.name == "nx") ? j : inx;
        iny = (p.name == "ny") ? j : iny;
        inz = (p.name == "nz") ? j : inz;
        ir = (p.name == "red"  ) ? j : ir;
        ig = (p.name == "green") ? j : ig;
        ib = (p.name == "blue" ) ? j : ib;
    }

    for (size_t i = 0; i < m_vertexCount; i++)
    {
        Point3f vertex, normal;
        Point3_<uchar> color;

        for (size_t j = 0; j < m_vertexDescription.properties.size(); j++)
        {
            const auto& p = m_vertexDescription.properties[j];
            uint ival = 0; float fval = 0;
            // here signedness is not important
            switch (p.valType)
            {
            case CV_8U: case CV_8S:
                ival = readNext<uchar>(file, m_inputDataFormat);
                break;
            case CV_16U: case CV_16S:
                ival = readNext<ushort>(file, m_inputDataFormat);
                break;
            case CV_32S: case CV_32U:
                ival = readNext<uint>(file, m_inputDataFormat);
                break;
            case CV_32F:
                fval = readNext<float>(file, m_inputDataFormat);
                break;
            case CV_64F:
                fval = (float)readNext<double>(file, m_inputDataFormat);
                break;
            default:
                break;
            }
            if (j == ivx)
            {
                vertex.x = fval;
            }
            if (j == ivy)
            {
                vertex.y = fval;
            }
            if (j == ivz)
            {
                vertex.z = fval;
            }
            if (j == inx)
            {
                normal.x = fval;
            }
            if (j == iny)
            {
                normal.y = fval;
            }
            if (j == inz)
            {
                normal.z = fval;
            }
            if (j == ir)
            {
                color.x = (uchar)ival;
            }
            if (j == ig)
            {
                color.y = (uchar)ival;
            }
            if (j == ib)
            {
                color.z = (uchar)ival;
            }
        }

        points.push_back(vertex);
        if (m_hasColour)
        {
            rgb.push_back(color);
        }
        if (m_hasNormal)
        {
            normals.push_back(normal);
        }
    }

    indices.reserve(m_faceCount);
    for (size_t i = 0; i < m_faceCount; i++)
    {
        for (const auto& p : m_faceDescription.properties)
        {
            if (p.isList)
            {
                size_t nVerts = readNext<uchar>(file, m_inputDataFormat);
                if (nVerts < 3)
                {
                    CV_LOG_ERROR(NULL, "Face should have at least 3 vertices but has " << nVerts);
                    return;
                }
                // PLY can have faces with >3 vertices in TRIANGLE_FAN format
                // in this case we load them as separate triangles
                int vert1 = readNext<int>(file, m_inputDataFormat);
                int vert2 = readNext<int>(file, m_inputDataFormat);
                for (size_t j = 2; j < nVerts; j++)
                {
                    int vert3 = readNext<int>(file, m_inputDataFormat);
                    indices.push_back({vert1, vert2, vert3});
                    vert2 = vert3;
                }
            }
            else
            {
                // read and discard
                switch (p.valType)
                {
                case CV_8U: case CV_8S:
                    readNext<uchar>(file, m_inputDataFormat);
                    break;
                case CV_16U: case CV_16S:
                    readNext<ushort>(file, m_inputDataFormat);
                    break;
                case CV_32S: case CV_32U:
                    readNext<uint>(file, m_inputDataFormat);
                    break;
                case CV_32F:
                    readNext<float>(file, m_inputDataFormat);
                    break;
                case CV_64F:
                    readNext<double>(file, m_inputDataFormat);
                    break;
                default:
                    break;
                }
            }
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

    if(hasColor)
    {
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
        file << std::setprecision(8) << points[i].x << " " << points[i].y << " " << points[i].z;
        if (hasColor)
        {
            file << " " << static_cast<int>(rgb[i].x) << " " << static_cast<int>(rgb[i].y) << " " << static_cast<int>(rgb[i].z);
        }
        if (hasNormals)
        {
            file << " " << std::setprecision(8) << normals[i].x << " " << normals[i].y << " " << normals[i].z;
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
