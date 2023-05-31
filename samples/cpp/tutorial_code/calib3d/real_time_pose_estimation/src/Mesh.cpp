/*
 * Mesh.cpp
 *
 *  Created on: Apr 9, 2014
 *      Author: edgar
 */

#include "Mesh.h"
#include "CsvReader.h"


// --------------------------------------------------- //
//                   TRIANGLE CLASS                    //
// --------------------------------------------------- //

/**  The custom constructor of the Triangle Class */
Triangle::Triangle(const cv::Point3f& V0, const cv::Point3f& V1, const cv::Point3f& V2) :
    v0_(V0), v1_(V1), v2_(V2)
{
}

/**  The default destructor of the Class */
Triangle::~Triangle()
{
    // TODO Auto-generated destructor stub
}


// --------------------------------------------------- //
//                     RAY CLASS                       //
// --------------------------------------------------- //

/**  The custom constructor of the Ray Class */
Ray::Ray(const cv::Point3f& P0, const cv::Point3f& P1) :
    p0_(P0), p1_(P1)
{
}

/**  The default destructor of the Class */
Ray::~Ray()
{
    // TODO Auto-generated destructor stub
}


// --------------------------------------------------- //
//                 OBJECT MESH CLASS                   //
// --------------------------------------------------- //

/** The default constructor of the ObjectMesh Class */
Mesh::Mesh() : num_vertices_(0), num_triangles_(0),
    list_vertex_(0) , list_triangles_(0)
{
}

/** The default destructor of the ObjectMesh Class */
Mesh::~Mesh()
{
    // TODO Auto-generated destructor stub
}

/** Load a CSV with *.ply format **/
void Mesh::load(const std::string& path)
{
    // Create the reader
    CsvReader csvReader(path);

    // Clear previous data
    list_vertex_.clear();
    list_triangles_.clear();

    // Read from .ply file
    csvReader.readPLY(list_vertex_, list_triangles_);

    // Update mesh attributes
    num_vertices_ = (int)list_vertex_.size();
    num_triangles_ = (int)list_triangles_.size();
}
