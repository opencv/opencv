#include "coders_obj.hpp"
#include <iostream>

namespace cv
{
namespace pc
{

ObjDecoder::ObjDecoder()
{
    m_vertices_size = 0;
}

ObjDecoder::~ObjDecoder()
{

}

void ObjDecoder::readData( std::vector<Point3f>& points, std::vector<Point3f>& normals )
{
    std::vector<Point3f> temp_vertices;
    std::vector<Point3f> temp_normals;

    FILE * file = fopen(m_filename.c_str(), "r");
    if( file == NULL ){
        printf("Impossible to open the file !\n");
        return;
    }

    char lineHeader[128];
    char c;

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

    fclose(file);

    points = temp_vertices;
    normals = temp_normals;
}

PointCloudDecoder ObjDecoder::newDecoder() const
{
    return PointCloudDecoder();
}



ObjEncoder::ObjEncoder()
{
    
}

ObjEncoder::~ObjEncoder()
{

}

void ObjEncoder::writeData( std::vector<Point3f>& points, std::vector<Point3f>& normals )
{

    FILE * file = fopen(m_filename.c_str(), "w");
    if( file == NULL ){
        printf("Impossible to open the file !\n");
        return;
    }

    fprintf(file, "# OBJ file writer for GSOC WIP\n");
    fprintf(file, "o Point_Cloud\n");

    for (int i = 0; i < points.size(); i++)
    {
        Point3f vertex = points[i];
        fprintf(file, "v %f %f %f\n", vertex.x, vertex.y, vertex.z);
    }
    
    for (int i = 0; i < normals.size(); i++)
    {
        Point3f normal = normals[i];
        fprintf(file, "vn %f %f %f\n", normal.x, normal.y, normal.z);
    }

    fclose(file);
}

PointCloudEncoder ObjEncoder::newEncoder() const
{
    return PointCloudEncoder();
}

}

}