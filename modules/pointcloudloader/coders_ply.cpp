#include "coders_ply.hpp"
#include <iostream>

namespace cv
{
namespace pc
{

PlyDecoder::PlyDecoder()
{
    m_vertices_size = 0;
}

PlyDecoder::~PlyDecoder()
{

}

void PlyDecoder::readData( std::vector<Point3f>& points, std::vector<Point3f>& normals )
{
    std::vector<Point3f> temp_vertices;
    std::vector<Point3f> temp_normals;

    FILE * file = fopen(m_filename.c_str(), "r");
    if( file == NULL ){
        printf("Impossible to open the file !\n");
        return;
    }

    char lineHeader[128];
    while (1) {

        // read the first word of the line
        int res = fscanf(file, "%s", lineHeader);
        if (res == EOF)
            break; // EOF = End Of File. Quit the loop.

        // else : parse lineHeader


        if (strcmp(lineHeader, "v") == 0)
        {
            Point3f vertex;
            fscanf(file, "%f %f %f\n", &vertex.x, &vertex.y, &vertex.z);
            temp_vertices.push_back(vertex);
        }
        else if (strcmp(lineHeader, "vn") == 0)
        {
            Point3f normal;
            fscanf(file, "%f %f %f\n", &normal.x, &normal.y, &normal.z);
            temp_normals.push_back(normal);
        }
    }

    points = temp_vertices;
    normals = temp_normals;
}

PointCloudDecoder PlyDecoder::newDecoder() const
{
    return PointCloudDecoder();
}

}

}