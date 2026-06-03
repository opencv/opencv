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

} // namespace cv

#endif
