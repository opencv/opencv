/*
 * Mesh.h
 *
 *  Created on: Apr 9, 2014
 *      Author: edgar
 */

#ifndef MESH_H_
#define MESH_H_

#include <iostream>
#include <opencv2/core/core.hpp>


// --------------------------------------------------- //
//                 TRIANGLE CLASS                      //
// --------------------------------------------------- //

class Triangle {
public:

  explicit Triangle(int id, cv::Point3f V0, cv::Point3f V1, cv::Point3f V2);
  virtual ~Triangle();

  cv::Point3f getV0() const { return v0_; }
  cv::Point3f getV1() const { return v1_; }
  cv::Point3f getV2() const { return v2_; }

private:
  /** The identifier number of the triangle */
  int id_;
  /** The three vertices that defines the triangle */
  cv::Point3f v0_, v1_, v2_;
};


// --------------------------------------------------- //
//                     RAY CLASS                       //
// --------------------------------------------------- //

class Ray {
public:

  explicit Ray(cv::Point3f P0, cv::Point3f P1);
  virtual ~Ray();

  cv::Point3f getP0() { return p0_; }
  cv::Point3f getP1() { return p1_; }

private:
  /** The two points that defines the ray */
  cv::Point3f p0_, p1_;
};


// --------------------------------------------------- //
//                OBJECT MESH CLASS                    //
// --------------------------------------------------- //

class Mesh
{
public:

  Mesh();
  virtual ~Mesh();

  std::vector<std::vector<int> > getTrianglesList() const { return list_triangles_; }
  cv::Point3f getVertex(int pos) const { return list_vertex_[pos]; }
  int getNumVertices() const { return num_vertexs_; }

  void load(const std::string path_file);

private:
  /** The identification number of the mesh */
  int id_;
  /** The current number of vertices in the mesh */
  int num_vertexs_;
  /** The current number of triangles in the mesh */
  int num_triangles_;
  /* The list of triangles of the mesh */
  std::vector<cv::Point3f> list_vertex_;
  /* The list of triangles of the mesh */
  std::vector<std::vector<int> > list_triangles_;
};

#endif /* OBJECTMESH_H_ */
