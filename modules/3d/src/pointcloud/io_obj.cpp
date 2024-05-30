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

void ObjDecoder::readData(std::vector<Point3f>& points, std::vector<Point3f>& normals, std::vector<Point3f>& rgb)
{
    std::vector<Point3f> texCoords;
    int nTexCoords;
    std::vector<std::vector<int32_t>> indices;
    this->readData(points, normals, rgb, texCoords, nTexCoords, indices, READ_AS_IS_FLAG);
}

void ObjDecoder::readData(std::vector<Point3f>& points, std::vector<Point3f>& normals, std::vector<Point3f>& rgb,
                          std::vector<Point3f>& texCoords, int& nTexCoords,
                          std::vector<std::vector<int32_t>>& indices, int flags)
{
    std::vector<Point3f> ptsList, nrmList, texCoordList, rgbList;
    std::vector<std::vector<int32_t>> idxList, texIdxList, normalIdxList;

    nTexCoords = 0;

    bool duplicateVertices = false;

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
        // "\r" symbols are not trimmed by default
        s = trimSpaces(s);
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
            ptsList.push_back(vertex);
            if (splitArr.size() == 5 || splitArr.size() == 8)
            {
                float w;
                ss >> w;
                CV_UNUSED(w);
            }
            if (splitArr.size() >= 7)
            {
                Point3f color;
                if (ss.rdbuf()->in_avail() != 0)
                {
                    ss >> color.x >> color.y >> color.z;
                    rgbList.push_back(color);
                }
            }
        }
        else if (key == "vn")
        {
            Point3f normal;
            ss >> normal.x >> normal.y >> normal.z;
            nrmList.push_back(normal);
        }
        else if (key == "f")
        {
            // format: "f v0 / t0 / n0 v1 / t1 / n1 v2/t2/n2 ..."
            std::vector<int> vertexInd, normInd, texInd;
            vertexInd.reserve(3); normInd.reserve(3); texInd.reserve(3);
            auto tokens = split(s, ' ');
            for (size_t i = 1; i < tokens.size(); i++)
            {
                auto vertexinfo = split(tokens[i], '/');
                std::array<int, 3> idx = { -1, -1, -1 };
                for (int j = 0; j < (int)vertexinfo.size(); j++)
                {
                    std::string sj = vertexinfo[j];
                    // trimming spaces; as a result s can become empty - this is not an error
                    auto si = std::find_if(sj.begin(),  sj.end(),  [](char c) { return (c >= '0' && c <= '9'); });
                    auto ei = std::find_if(sj.rbegin(), sj.rend(), [](char c) { return (c >= '0' && c <= '9'); });
                    if (si != sj.end() && ei != sj.rend())
                    {
                        auto first = std::distance(si, sj.begin());
                        auto last  = std::distance(ei, sj.rend());
                        sj = sj.substr(first, last - first + 1);
                        try
                        {
                            idx[j] = std::stoi(sj);
                        }
                        // std::invalid_exception, std::out_of_range
                        catch(const std::exception&)
                        {
                            CV_LOG_ERROR(NULL, "Failed to parse face index: " + sj);
                            return;
                        }
                    }
                }
                int vertexIndex   = idx[0];
                int texCoordIndex = idx[1];
                int normalIndex   = idx[2];

                if (vertexIndex <= 0)
                {
                    CV_LOG_ERROR(NULL, "Vertex index is not present or incorrect");
                    return;
                }

                if ((vertexIndex != texCoordIndex && texCoordIndex >= 0) ||
                    (vertexIndex != normalIndex   && normalIndex   >= 0))
                {
                    duplicateVertices = !(flags & READ_AS_IS_FLAG);
                }

                vertexInd.push_back(vertexIndex - 1);
                normInd.push_back(normalIndex - 1);
                texInd.push_back(texCoordIndex - 1);
            }
            idxList.push_back(vertexInd);
            texIdxList.push_back(texInd);
            normalIdxList.push_back(normInd);
        }
        else if (key == "vt")
        {
            // (u, [v, [w]])
            auto splitArr = split(s, ' ');
            int ncoords = (int)splitArr.size() - 1;
            if (!nTexCoords)
            {
                nTexCoords = ncoords;
                if (nTexCoords < 1 || nTexCoords > 3)
                {
                    CV_LOG_ERROR(NULL, "The amount of texture coordinates should be between 1 and 3");
                    return;
                }
            }

            if (ncoords != nTexCoords)
            {
                CV_LOG_ERROR(NULL, "All points should have the same number of texture coordinates");
                return;
            }

            Vec3f tc;
            for (int i = 0; i < nTexCoords; i++)
            {
                ss >> tc[i];
            }
            texCoordList.push_back(tc);
        }
        else
        {
            if (m_unsupportedKeys.find(key) == m_unsupportedKeys.end()) {
                m_unsupportedKeys.insert(key);
                CV_LOG_WARNING(NULL, "Key " << key << " not supported");
            }
        }
    }

    if (duplicateVertices)
    {
        points.clear();
        normals.clear();
        rgb.clear();
        texCoords.clear();
        indices.clear();

        for (int tri = 0; tri < (int)idxList.size(); tri++)
        {
            auto vi = idxList[tri];
            auto ti = texIdxList[tri];
            auto ni = normalIdxList[tri];

            std::vector<int32_t> newvi;
            newvi.reserve(3);
            for (int i = 0; i < (int)vi.size(); i++)
            {
                newvi.push_back((int)points.size());
                points.push_back(ptsList.at(vi[i]));
                if (!rgbList.empty())
                {
                    rgb.push_back(rgbList.at(vi[i]));
                }

                texCoords.push_back(texCoordList.at(ti[i]));
                normals.push_back(nrmList.at(ni[i]));
            }

            indices.push_back(newvi);
        }
    }
    else
    {
        points    = std::move(ptsList);
        normals   = std::move(nrmList);
        rgb       = std::move(rgbList);
        texCoords = std::move(texCoordList);
        indices   = std::move(idxList);
    }

    file.close();
}

void ObjEncoder::writeData(const std::vector<Point3f>& points, const std::vector<Point3f>& normals, const std::vector<Point3f>& rgb,
                           const std::vector<Point3f>& texCoords, int nTexCoords,
                           const std::vector<std::vector<int32_t>>& indices)
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

    if (texCoords.empty() && nTexCoords > 0)
    {
        CV_LOG_ERROR(NULL, "No texture coordinates provided while having nTexCoord > 0");
        return;
    }

    file << "# OBJ file writer" << std::endl;
    file << "o Point_Cloud" << std::endl;

    for (size_t i = 0; i < points.size(); ++i)
    {
        file << "v " << points[i].x << " " << points[i].y << " " << points[i].z;
        if (!rgb.empty())
        {
            file << " " << rgb[i].x << " " << rgb[i].y  << " " << rgb[i].z;
        }
        file << std::endl;
    }

    for (const auto& normal : normals)
    {
        file << "vn " << normal.x << " " << normal.y << " " << normal.z << std::endl;
    }

    for (const auto& tc : texCoords)
    {
        file << "vt " << tc.x;
        if (nTexCoords > 1)
        {
            file << " " << tc.y;
        }
        if (nTexCoords > 2)
        {
            file << " " << tc.z;
        }
        file << std::endl;
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
