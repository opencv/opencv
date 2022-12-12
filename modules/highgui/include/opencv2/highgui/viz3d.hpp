// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef __OPENCV_HIGHGUI_VIZ3D_HPP__
#define __OPENCV_HIGHGUI_VIZ3D_HPP__

#include "opencv2/core.hpp"

namespace cv { namespace viz3d {

//! Render modes for cv::viz3d::showBox / cv::viz3d::showPlane / cv::viz3d::showSphere
enum RenderMode
{
    RENDER_SIMPLE      = 0,
    RENDER_SHADING     = 1,
    RENDER_WIREFRAME   = 2,
};

/** @brief Sets the view's perspective on a viz3d window.
@param win_name Name of the viz3d window.
@param fov Vertical field of view in radians.
@param z_near Minimum clip distance.
@param z_far Maximum clip distance.
 */
CV_EXPORTS_W void setPerspective(const String& win_name, float fov, float z_near, float z_far);

/** @brief Hides or shows the grid on a viz3d window.
@param win_name Name of the viz3d window.
@param visible Should the grid be shown?
*/
CV_EXPORTS_W void setGridVisible(const String& win_name, bool visible);

/** @brief Sets the properties of the light used on a viz3d window.
@param win_name Name of the viz3d window.
@param direction Direction any point to the sun.
@param ambient Ambient light color.
@param diffuse Diffuse light color.
 */
CV_EXPORTS_W void setSun(const String& win_name, const Vec3f& direction, const Vec3f& ambient, const Vec3f& diffuse);

/** @brief Sets the properties of the sky used on a viz3d window.
@param win_name Name of the viz3d window.
@param color Sky color.
 */
CV_EXPORTS_W void setSky(const String& win_name, const Vec3f& color);

/** @brief Shows a box object in the specified viz3d window. See cv::viz3d::destroyObject.

@param win_name Name of the viz3d window.
@param obj_name Name of the object.
@param size Box size.
@param color Box color.
@param mode Render mode.
 */
CV_EXPORTS_W void showBox(const String& win_name, const String& obj_name, const Vec3f& size, const Vec3f& color, RenderMode mode = RENDER_SIMPLE);

/** @brief Shows a plane object in the specified viz3d window. See cv::viz3d::destroyObject.

The place faces the positive Y axis by default.

@param win_name Name of the viz3d window.
@param obj_name Name of the object.
@param size Plane size.
@param color Plane color.
@param mode Render mode.
 */
CV_EXPORTS_W void showPlane(const String& win_name, const String& obj_name, const Vec2f& size, const Vec3f& color, RenderMode mode = RENDER_SIMPLE);

/** @brief Shows a sphere object in the specified viz3d window. See cv::viz3d::destroyObject.

@param win_name Name of the viz3d window.
@param obj_name Name of the object.
@param radius Sphere radius.
@param color Sphere color.
@param mode Render mode.
@param divs The higher this integer is the more detail (triangles) the sphere has.
 */
CV_EXPORTS_W void showSphere(const String& win_name, const String& obj_name, float radius, const Vec3f& color, RenderMode mode = RENDER_SIMPLE, int divs = 3);

/** @brief Shows a camera trajectory object in the specified viz3d window. See cv::viz3d::destroyObject.

The camera trajectory data array must be 2D. Each row has a width of 6, where the first
3 components are the position and the second 3 components are the direction of the camera.

@param win_name Name of the viz3d window.
@param obj_name Name of the object.
@param trajectory Camera trajectory data.
@param aspect Aspect ratio of the camera frustum.
@param scale Scale applied to camera frustums.
@param frustum_color Color of the frustums.
@param line_color Color of the line.
 */
CV_EXPORTS_W void showCameraTrajectory(
    const String& win_name, const String& obj_name, InputArray trajectory,
    float aspect, float scale, Vec3f frustum_color = Vec3f(1.0f, 1.0f, 1.0f),
    Vec3f line_color = Vec3f(0.5f, 0.5f, 0.5f));

/** @brief Shows a mesh object in the specified viz3d window. See cv::viz3d::destroyObject.

The vertices array must be 2D. If shading is disabled (the default option), each row has a width of
3 (xyz), 6 (xyz, rgb) or 9 (xyz, rgb, uvw) where uvw is the normal, so that each row represents a vertex.
Shading is only enabled when using normals.

@param win_name Name of the viz3d window.
@param obj_name Name of the object.
@param verts Vertices input array.
@param indices Indices input array.
@param shading Should the mesh have shading?
 */
CV_EXPORTS void showMesh(const String& win_name, const String& obj_name, InputArray verts, InputArray indices);

/** @overload Shows a mesh object in the specified viz3d window. See cv::viz3d::destroyObject.

The vertices array must be 2D, where each row has a width of 3 (xyz), 6 (xyz, rgb) or 9 (xyz, rgb, uvw)
where uvw is the normal, so that each row represents a vertex. Shading is only enabled when using normals.
Points are grouped in triplets to make triangles (points 1, 2 and 3, points 4, 5 and 6, etc).

@param win_name Name of the viz3d window.
@param obj_name Name of the object.
@param verts Vertices input array.
@param indices Indices input array.
 */
CV_EXPORTS_AS(showMesh2) void showMesh(const String& win_name, const String& obj_name, InputArray verts);

/** @brief Shows a point cloud object in the specified viz3d window. See cv::viz3d::destroyObject.

The points array must be 2D, where each row has either a width of 6 (xyz, rgb), so that
each row represents a point.

@param win_name Name of the viz3d window.
@param obj_name Name of the object.
@param points Vertices input array.
 */
CV_EXPORTS_W void showPoints(const String& win_name, const String& obj_name, InputArray points);

/** @brief Shows a colored depth map as a point cloud in the specified viz3d window. See cv::viz3d::destroyObject.

The colored depth map must be a 2D image with 4 channels (RGBD). The points are created assuming
that positive depth means positive position in the Z axis.

@param win_name Name of the viz3d window.
@param obj_name Name of the object.
@param img Input colored depth map.
@param intrinsics Camera intrinsics matrix.
@param scale Scale applied to the coordinates of the points.
 */
CV_EXPORTS_W void showRGBD(const String& win_name, const String& obj_name, InputArray img, const Matx33f& intrinsics, float scale = 1.0f);

/** @brief Shows a lines object in the specified viz3d window. See cv::viz3d::destroyObject.

The points array must be 2D, where each row has either a width of 6 (xyz, rgb), so that
each row represents a point. Points are grouped in pairs to make lines (points 1 and 2,
points 3 and 4, etc).

@param win_name Name of the viz3d window.
@param obj_name Name of the object.
@param points Vertices input array.
 */
CV_EXPORTS_W void showLines(const String& win_name, const String& obj_name, InputArray points);

/** @brief Sets an object's position in the specified viz3d window.
@param win_name Name of the viz3d window.
@param obj_name Name of the object.
@param position New position.
*/
CV_EXPORTS_W void setObjectPosition(const String& win_name, const String& obj_name, const Vec3f& position);

/** @brief Sets an object's rotation in the specified viz3d window.
@param win_name Name of the viz3d window.
@param obj_name Name of the object.
@param rotation New rotation in euler angles (radians).
*/
CV_EXPORTS_W void setObjectRotation(const String& win_name, const String& obj_name, const Vec3f& rotation);

/** @brief Destroys an object created with cv::viz3d::showMesh or cv::viz3d::showPoints.

@param win_name Name of the viz3d window.
@param obj_name Name of the object.
 */
CV_EXPORTS_W void destroyObject(const String& win_name, const String& obj_name);

} // namespace viz3d
} // namespace cv


#endif
