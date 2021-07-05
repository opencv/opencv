#ifndef __OPENCV_LOAD_POINT_CLOUD_HPP__
#define __OPENCV_LOAD_POINT_CLOUD_HPP__

#include <opencv2/core.hpp>

//////////////////////////////// pont cloud codec ////////////////////////////////
namespace cv
{
namespace pc
{

/** @brief Loads a point cloud from a file.
 
    The function loadPointCloud loads point cloud from the specified file and returns it.
    If the cloud cannot be read, the function will return empty arrays

    Currently, the following file formats are supported:
    -  Wavefront obj file \*.obj (WIP)
    -  Polygon File Format \*.ply (WIP)

    @param filename Name of the file.
    @param vertices (vector of Point3f) Point coordinates of a point cloud
    @param normals (vector of Point3f) Point normals of a point cloud
*/
void loadPointCloud( const String& filename, OutputArray vertices, OutputArray normals);


/** @brief Saves a point cloud to a specified file.
 
    The function savePointCloud saves point cloud to the specified file.
    File format is chosen based on the filename extension.

    @param filename Name of the file.
    @param vertices (vector of Point3f) Point coordinates of a point cloud
    @param normals (vector of Point3f) Point normals of a point cloud
*/
void savePointCloud( const String& filename, InputArray  vertices, InputArray  normals);


/** @brief Loads a mesh from a file.
 
    The function loadMesh loads mesh from the specified file and returns it.
    If the mesh cannot be read, the function will return empty arrays
    
    Currently, the following file formats are supported:
    -  Wavefront obj file \*.obj (WIP)
    -  Polygon File Format \*.ply (WIP)

    @param filename Name of the file.
    @param vertices (vector of Point3f) Vertex coordinates of a mesh
    @param normals (vector of Point3f) Vertex normals of a mesh
    @param indices (vector of int) Vertex indices in a mesh
*/
void loadMesh( const String& filename, OutputArray vertices, OutputArray normals, OutputArray indices, int flags );


/** @brief Saves a mesh to a specified file.
 
    The function saveMesh saves mesh to the specified file.
    File format is chosen based on the filename extension.

    @param filename Name of the file.
    @param vertices (vector of Point3f) Vertex coordinates of a mesh
    @param normals (vector of Point3f) Vertex normals of a mesh
    @param indices (vector of int) Vertex indices in a mesh
*/
void saveMesh( const String& filename, OutputArray vertices, OutputArray normals, OutputArray indices, int flags );


}

}

#endif
