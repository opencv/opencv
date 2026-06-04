// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#ifndef POINT_CLOUD_MODULE_HPP
#define POINT_CLOUD_MODULE_HPP

#include "opencv2/core.hpp"
#include "opencv2/ptcloud/depth.hpp"
#include "opencv2/ptcloud/volume.hpp"
#include "opencv2/ptcloud/odometry.hpp"
#include "opencv2/ptcloud/odometry_frame.hpp"
#include "opencv2/ptcloud/odometry_settings.hpp"

/**
@defgroup ptcloud Point Clound Processing
*/

//! @addtogroup ptcloud
//! @{

namespace cv {

/** @brief Loads a point cloud from a file.
 *
 * The function loads point cloud from the specified file and returns it.
 * If the cloud cannot be read, throws an error.
 * Vertex coordinates, normals and colors are returned as they are saved in the file
 * even if these arrays have different sizes and their elements do not correspond to each other
 * (which is typical for OBJ files for example)
 *
 * Currently, the following file formats are supported:
 * -  [Wavefront obj file *.obj](https://en.wikipedia.org/wiki/Wavefront_.obj_file)
 * -  [Polygon File Format *.ply](https://en.wikipedia.org/wiki/PLY_(file_format))
 *
 * @param filename Name of the file
 * @param vertices vertex coordinates, each value contains 3 floats
 * @param normals per-vertex normals, each value contains 3 floats
 * @param rgb per-vertex colors, each value contains 3 floats
 */
CV_EXPORTS_W void loadPointCloud(const String &filename, OutputArray vertices, OutputArray normals = noArray(), OutputArray rgb = noArray());

/** @brief Saves a point cloud to a specified file.
 *
 * The function saves point cloud to the specified file.
 * File format is chosen based on the filename extension.
 *
 * @param filename Name of the file
 * @param vertices vertex coordinates, each value contains 3 floats
 * @param normals per-vertex normals, each value contains 3 floats
 * @param rgb per-vertex colors, each value contains 3 floats
 */
CV_EXPORTS_W void savePointCloud(const String &filename, InputArray vertices, InputArray normals = noArray(), InputArray rgb = noArray());

/** @brief Loads a mesh from a file.
 *
 * The function loads mesh from the specified file and returns it.
 * If the mesh cannot be read, throws an error
 * Vertex attributes (i.e. space and texture coodinates, normals and colors) are returned in same-sized
 * arrays with corresponding elements having the same indices.
 * This means that if a face uses a vertex with a normal or a texture coordinate with different indices
 * (which is typical for OBJ files for example), this vertex will be duplicated for each face it uses.
 *
 * Currently, the following file formats are supported:
 * -  [Wavefront obj file *.obj](https://en.wikipedia.org/wiki/Wavefront_.obj_file) (ONLY TRIANGULATED FACES)
 * -  [Polygon File Format *.ply](https://en.wikipedia.org/wiki/PLY_(file_format))
 * @param filename Name of the file
 * @param vertices vertex coordinates, each value contains 3 floats
 * @param indices per-face list of vertices, each value is a vector of ints
 * @param normals per-vertex normals, each value contains 3 floats
 * @param colors per-vertex colors, each value contains 3 floats
 * @param texCoords per-vertex texture coordinates, each value contains 2 or 3 floats
 */
CV_EXPORTS_W void loadMesh(const String &filename, OutputArray vertices, OutputArrayOfArrays indices,
                           OutputArray normals = noArray(), OutputArray colors = noArray(),
                           OutputArray texCoords = noArray());

/** @brief Saves a mesh to a specified file.
 *
 * The function saves mesh to the specified file.
 * File format is chosen based on the filename extension.
 *
 * @param filename Name of the file.
 * @param vertices vertex coordinates, each value contains 3 floats
 * @param indices per-face list of vertices, each value is a vector of ints
 * @param normals per-vertex normals, each value contains 3 floats
 * @param colors per-vertex colors, each value contains 3 floats
 * @param texCoords per-vertex texture coordinates, each value contains 2 or 3 floats
 */
CV_EXPORTS_W void saveMesh(const String &filename, InputArray vertices, InputArrayOfArrays indices,
                           InputArray normals = noArray(), InputArray colors = noArray(), InputArray texCoords = noArray());


//! Triangle fill settings
enum TriangleShadingType
{
    RASTERIZE_SHADING_WHITE  = 0, //!< a white color is used for the whole triangle
    RASTERIZE_SHADING_FLAT   = 1, //!< a color of 1st vertex of each triangle is used
    RASTERIZE_SHADING_SHADED = 2  //!< a color is interpolated between 3 vertices with perspective correction
};

//! Face culling settings: what faces are drawn after face culling
enum TriangleCullingMode
{
    RASTERIZE_CULLING_NONE = 0, //!< all faces are drawn, no culling is actually performed
    RASTERIZE_CULLING_CW   = 1, //!< triangles which vertices are given in clockwork order are drawn
    RASTERIZE_CULLING_CCW  = 2  //!< triangles which vertices are given in counterclockwork order are drawn
};

//! GL compatibility settings
enum TriangleGlCompatibleMode
{
    RASTERIZE_COMPAT_DISABLED = 0, //!< Color and depth have their natural values and converted to internal formats if needed
    RASTERIZE_COMPAT_INVDEPTH = 1  //!< Color is natural, Depth is transformed from [-zNear; -zFar] to [0; 1]
    //!< by the following formula: \f$ \frac{z_{far} \left(z + z_{near}\right)}{z \left(z_{far} - z_{near}\right)} \f$ \n
    //!< In this mode the input/output depthBuf is considered to be in this format,
    //!< therefore it's faster since there're no conversions performed
};

/**
 * @brief Structure to keep settings for rasterization
 */
struct CV_EXPORTS_W_SIMPLE TriangleRasterizeSettings
{
    TriangleRasterizeSettings();

    CV_WRAP TriangleRasterizeSettings& setShadingType(TriangleShadingType st) { shadingType = st; return *this; }
    CV_WRAP TriangleRasterizeSettings& setCullingMode(TriangleCullingMode cm) { cullingMode = cm; return *this; }
    CV_WRAP TriangleRasterizeSettings& setGlCompatibleMode(TriangleGlCompatibleMode gm) { glCompatibleMode = gm; return *this; }

    TriangleShadingType shadingType;
    TriangleCullingMode cullingMode;
    TriangleGlCompatibleMode glCompatibleMode;
};


/** @brief Renders a set of triangles on a depth and color image
 *
 * Triangles can be drawn white (1.0, 1.0, 1.0), flat-shaded or with a color interpolation between vertices.
 * In flat-shaded mode the 1st vertex color of each triangle is used to fill the whole triangle.
 *
 * The world2cam is an inverted camera pose matrix in fact. It transforms vertices from world to
 * camera coordinate system.
 *
 * The camera coordinate system emulates the OpenGL's coordinate system having coordinate origin in a screen center,
 * X axis pointing right, Y axis pointing up and Z axis pointing towards the viewer
 * except that image is vertically flipped after the render.
 * This means that all visible objects are placed in z-negative area, or exactly in -zNear > z > -zFar since
 * zNear and zFar are positive.
 * For example, at fovY = PI/2 the point (0, 1, -1) will be projected to (width/2, 0) screen point,
 * (1, 0, -1) to (width/2 + height/2, height/2). Increasing fovY makes projection smaller and vice versa.
 *
 * The function does not create or clear output images before the rendering. This means that it can be used
 * for drawing over an existing image or for rendering a model into a 3D scene using pre-filled Z-buffer.
 *
 * Empty scene results in a depth buffer filled by the maximum value since every pixel is infinitely far from the camera.
 * Therefore, before rendering anything from scratch the depthBuf should be filled by zFar values (or by ones in INVDEPTH mode).
 *
 * There are special versions of this function named triangleRasterizeDepth and triangleRasterizeColor
 * for cases if a user needs a color image or a depth image alone; they may run slightly faster.
 *
 * @param vertices vertices coordinates array. Should contain values of CV_32FC3 type or a compatible one (e.g. cv::Vec3f, etc.)
 * @param indices triangle vertices index array, 3 per triangle. Each index indicates a vertex in a vertices array.
 * Should contain CV_32SC3 values or compatible
 * @param colors per-vertex colors of CV_32FC3 type or compatible. Can be empty or the same size as vertices array.
 * If the values are out of [0; 1] range, the result correctness is not guaranteed
 * @param colorBuf an array representing the final rendered image. Should containt CV_32FC3 values and be the same size as depthBuf.
 * Not cleared before rendering, i.e. the content is reused as there is some pre-rendered scene.
 * @param depthBuf an array of floats containing resulting Z buffer. Should contain float values and be the same size as colorBuf.
 * Not cleared before rendering, i.e. the content is reused as there is some pre-rendered scene.
 * Empty scene corresponds to all values set to zFar (or to 1.0 in INVDEPTH mode)
 * @param world2cam a 4x3 or 4x4 float or double matrix containing inverted (sic!) camera pose
 * @param fovY field of view in vertical direction, given in radians
 * @param zNear minimum Z value to render, everything closer is clipped
 * @param zFar maximum Z value to render, everything farther is clipped
 * @param settings see TriangleRasterizeSettings. By default the smooth shading is on,
 * with CW culling and with disabled GL compatibility
 */
CV_EXPORTS_W void triangleRasterize(InputArray vertices, InputArray indices, InputArray colors,
                                    InputOutputArray colorBuf, InputOutputArray depthBuf,
                                    InputArray world2cam, double fovY, double zNear, double zFar,
                                    const TriangleRasterizeSettings& settings = TriangleRasterizeSettings());

/** @brief Overloaded version of triangleRasterize() with depth-only rendering
 *
 * @param vertices vertices coordinates array. Should contain values of CV_32FC3 type or a compatible one (e.g. cv::Vec3f, etc.)
 * @param indices triangle vertices index array, 3 per triangle. Each index indicates a vertex in a vertices array.
 * Should contain CV_32SC3 values or compatible
 * @param depthBuf an array of floats containing resulting Z buffer. Should contain float values and be the same size as colorBuf.
 * Not cleared before rendering, i.e. the content is reused as there is some pre-rendered scene.
 * Empty scene corresponds to all values set to zFar (or to 1.0 in INVDEPTH mode)
 * @param world2cam a 4x3 or 4x4 float or double matrix containing inverted (sic!) camera pose
 * @param fovY field of view in vertical direction, given in radians
 * @param zNear minimum Z value to render, everything closer is clipped
 * @param zFar maximum Z value to render, everything farther is clipped
 * @param settings see TriangleRasterizeSettings. By default the smooth shading is on,
 * with CW culling and with disabled GL compatibility
 */
CV_EXPORTS_W void triangleRasterizeDepth(InputArray vertices, InputArray indices, InputOutputArray depthBuf,
                                         InputArray world2cam, double fovY, double zNear, double zFar,
                                         const TriangleRasterizeSettings& settings = TriangleRasterizeSettings());

/** @brief Overloaded version of triangleRasterize() with color-only rendering
 *
 * @param vertices vertices coordinates array. Should contain values of CV_32FC3 type or a compatible one (e.g. cv::Vec3f, etc.)
 * @param indices triangle vertices index array, 3 per triangle. Each index indicates a vertex in a vertices array.
 * Should contain CV_32SC3 values or compatible
 * @param colors per-vertex colors of CV_32FC3 type or compatible. Can be empty or the same size as vertices array.
 * If the values are out of [0; 1] range, the result correctness is not guaranteed
 * @param colorBuf an array representing the final rendered image. Should containt CV_32FC3 values and be the same size as depthBuf.
 * Not cleared before rendering, i.e. the content is reused as there is some pre-rendered scene.
 * @param world2cam a 4x3 or 4x4 float or double matrix containing inverted (sic!) camera pose
 * @param fovY field of view in vertical direction, given in radians
 * @param zNear minimum Z value to render, everything closer is clipped
 * @param zFar maximum Z value to render, everything farther is clipped
 * @param settings see TriangleRasterizeSettings. By default the smooth shading is on,
 * with CW culling and with disabled GL compatibility
 */
CV_EXPORTS_W void triangleRasterizeColor(InputArray vertices, InputArray indices, InputArray colors, InputOutputArray colorBuf,
                                         InputArray world2cam, double fovY, double zNear, double zFar,
                                         const TriangleRasterizeSettings& settings = TriangleRasterizeSettings());

/** @brief Octree for 3D vision.
 *
 * In 3D vision filed, the Octree is used to process and accelerate the pointcloud data. The class Octree represents
 * the Octree data structure. Each Octree will have a fixed depth. The depth of Octree refers to the distance from
 * the root node to the leaf node.All OctreeNodes will not exceed this depth.Increasing the depth will increase
 * the amount of calculation exponentially. And the small number of depth refers low resolution of Octree.
 * Each node contains 8 children, which are used to divide the space cube into eight parts. Each octree node represents
 * a cube. And these eight children will have a fixed order, the order is described as follows:
 *
 * For illustration, assume,
 *
 * rootNode: origin == (0, 0, 0), size == 2
 *
 * Then,
 *
 * children[0]: origin == (0, 0, 0), size == 1
 *
 * children[1]: origin == (1, 0, 0), size == 1, along X-axis next to child 0
 *
 * children[2]: origin == (0, 1, 0), size == 1, along Y-axis next to child 0
 *
 * children[3]: origin == (1, 1, 0), size == 1, in X-Y plane
 *
 * children[4]: origin == (0, 0, 1), size == 1, along Z-axis next to child 0
 *
 * children[5]: origin == (1, 0, 1), size == 1, in X-Z plane
 *
 * children[6]: origin == (0, 1, 1), size == 1, in Y-Z plane
 *
 * children[7]: origin == (1, 1, 1), size == 1, furthest from child 0
 */

class CV_EXPORTS_W Octree
{
public:
    //! Default constructor.
    Octree();

    /** @overload
     * @brief Creates an empty Octree with given maximum depth
     *
     * @param maxDepth The max depth of the Octree
     * @param size bounding box size for the Octree
     * @param origin Initial center coordinate
     * @param withColors Whether to keep per-point colors or not
     * @return resulting Octree
     */
    CV_WRAP static Ptr<Octree> createWithDepth(int maxDepth, double size, const Point3f& origin = { }, bool withColors = false);

    /** @overload
     * @brief Create an Octree from the PointCloud data with the specific maxDepth
     *
     * @param maxDepth Max depth of the octree
     * @param pointCloud point cloud data, should be 3-channel float array
     * @param colors color attribute of point cloud in the same 3-channel float format
     * @return resulting Octree
     */
    CV_WRAP static Ptr<Octree> createWithDepth(int maxDepth, InputArray pointCloud, InputArray colors = noArray());

    /** @overload
     * @brief Creates an empty Octree with given resolution
     *
     * @param resolution The size of the octree leaf node
     * @param size bounding box size for the Octree
     * @param origin Initial center coordinate
     * @param withColors Whether to keep per-point colors or not
     * @return resulting Octree
     */
    CV_WRAP static Ptr<Octree> createWithResolution(double resolution, double size, const Point3f& origin = { }, bool withColors = false);

    /** @overload
     * @brief Create an Octree from the PointCloud data with the specific resolution
     *
     * @param resolution The size of the octree leaf node
     * @param pointCloud point cloud data, should be 3-channel float array
     * @param colors color attribute of point cloud in the same 3-channel float format
     * @return resulting octree
     */
    CV_WRAP static Ptr<Octree> createWithResolution(double resolution, InputArray pointCloud, InputArray colors = noArray());

    //! Default destructor
    ~Octree();

    /** @overload
     * @brief Insert a point data with color to a OctreeNode.
     *
     * @param point The point data in Point3f format.
     * @param color The color attribute of point in Point3f format.
     * @return Returns whether the insertion is successful.
     */
    CV_WRAP bool insertPoint(const Point3f& point, const Point3f& color = { });

    /** @brief Determine whether the point is within the space range of the specific cube.
     *
     * @param point The point coordinates.
     * @return If point is in bound, return ture. Otherwise, false.
     */
    CV_WRAP bool isPointInBound(const Point3f& point) const;

    //! returns true if the rootnode is NULL.
    CV_WRAP bool empty() const;

    /** @brief Reset all octree parameter.
     *
     *  Clear all the nodes of the octree and initialize the parameters.
     */
    CV_WRAP void clear();

    /** @brief Delete a given point from the Octree.
     *
     * Delete the corresponding element from the pointList in the corresponding leaf node. If the leaf node
     * does not contain other points after deletion, this node will be deleted. In the same way,
     * its parent node may also be deleted if its last child is deleted.
     * @param point The point coordinates, comparison is epsilon-based
     * @return return ture if the point is deleted successfully.
     */
    CV_WRAP bool deletePoint(const Point3f& point);

    /** @brief restore point cloud data from Octree.
     *
     * Restore the point cloud data from existing octree. The points in same leaf node will be seen as the same point.
     * This point is the center of the leaf node. If the resolution is small, it will work as a downSampling function.
     * @param restoredPointCloud The output point cloud data, can be replaced by noArray() if not needed
     * @param restoredColor The color attribute of point cloud data, can be omitted if not needed
     */
    CV_WRAP void getPointCloudByOctree(OutputArray restoredPointCloud, OutputArray restoredColor = noArray());

    /** @brief Radius Nearest Neighbor Search in Octree.
     *
     * Search all points that are less than or equal to radius.
     * And return the number of searched points.
     * @param query Query point.
     * @param radius Retrieved radius value.
     * @param points Point output. Contains searched points in 3-float format, and output vector is not in order,
     * can be replaced by noArray() if not needed
     * @param squareDists Dist output. Contains searched squared distance in floats, and output vector is not in order,
     * can be omitted if not needed
     * @return the number of searched points.
     */
    CV_WRAP int radiusNNSearch(const Point3f& query, float radius, OutputArray points, OutputArray squareDists = noArray()) const;

    /** @overload
     *  @brief Radius Nearest Neighbor Search in Octree.
     *
     * Search all points that are less than or equal to radius.
     * And return the number of searched points.
     * @param query Query point.
     * @param radius Retrieved radius value.
     * @param points Point output. Contains searched points in 3-float format, and output vector is not in order,
     * can be replaced by noArray() if not needed
     * @param colors Color output. Contains colors corresponding to points in pointSet, can be replaced by noArray() if not needed
     * @param squareDists Dist output. Contains searched squared distance in floats, and output vector is not in order,
     * can be replaced by noArray() if not needed
     * @return the number of searched points.
     */
    CV_WRAP int radiusNNSearch(const Point3f& query, float radius, OutputArray points, OutputArray colors, OutputArray squareDists) const;

    /** @brief K Nearest Neighbor Search in Octree.
     *
     * Find the K nearest neighbors to the query point.
     * @param query Query point.
     * @param K amount of nearest neighbors to find
     * @param points Point output. Contains K points in 3-float format, arranged in order of distance from near to far,
     * can be replaced by noArray() if not needed
     * @param squareDists Dist output. Contains K squared distance in floats, arranged in order of distance from near to far,
     * can be omitted if not needed
     */
    CV_WRAP void KNNSearch(const Point3f& query, const int K, OutputArray points, OutputArray squareDists = noArray()) const;

    /** @overload
     *  @brief K Nearest Neighbor Search in Octree.
     *
     * Find the K nearest neighbors to the query point.
     * @param query Query point.
     * @param K amount of nearest neighbors to find
     * @param points Point output. Contains K points in 3-float format, arranged in order of distance from near to far,
     * can be replaced by noArray() if not needed
     * @param colors Color output. Contains colors corresponding to points in pointSet, can be replaced by noArray() if not needed
     * @param squareDists Dist output. Contains K squared distance in floats, arranged in order of distance from near to far,
     * can be replaced by noArray() if not needed
     */
    CV_WRAP void KNNSearch(const Point3f& query, const int K, OutputArray points, OutputArray colors, OutputArray squareDists) const;

protected:
    struct Impl;
    Ptr<Impl> p;
};

} // namespace cv

//! @} ptcloud

#endif
