#pragma once

#include <opencv2/core/cvdef.h>
#include <opencv2/core.hpp>
#include <opencv2/viz/types.hpp>
//#include <vtkMatrix4x4.h>

namespace temp_viz
{
    //CV_EXPORTS Eigen::Matrix4d vtkToEigen (vtkMatrix4x4* vtk_matrix);
    //CV_EXPORTS Eigen::Vector2i worldToView (const Eigen::Vector4d &world_pt, const Eigen::Matrix4d &view_projection_matrix, int width, int height);
    //CV_EXPORTS void getViewFrustum (const Eigen::Matrix4d &view_projection_matrix, double planes[24]);

//    enum FrustumCull
//    {
//        PCL_INSIDE_FRUSTUM,
//        PCL_INTERSECT_FRUSTUM,
//        PCL_OUTSIDE_FRUSTUM
//    };

    //CV_EXPORTS int cullFrustum (double planes[24], const Eigen::Vector3d &min_bb, const Eigen::Vector3d &max_bb);
    //CV_EXPORTS float viewScreenArea (const Eigen::Vector3d &eye, const Eigen::Vector3d &min_bb, const Eigen::Vector3d &max_bb, const Eigen::Matrix4d &view_projection_matrix, int width, int height);

    enum RenderingProperties
    {
        VIZ_POINT_SIZE,
        VIZ_OPACITY,
        VIZ_LINE_WIDTH,
        VIZ_FONT_SIZE,
        VIZ_COLOR,
        VIZ_REPRESENTATION,
        VIZ_IMMEDIATE_RENDERING,
        VIZ_SHADING
    };

    enum RenderingRepresentationProperties
    {
        REPRESENTATION_POINTS,
        REPRESENTATION_WIREFRAME,
        REPRESENTATION_SURFACE
    };

    enum ShadingRepresentationProperties
    {
        SHADING_FLAT,
        SHADING_GOURAUD,
        SHADING_PHONG
    };

    class CV_EXPORTS Camera
    {
    public:
        /** Focal point or lookAt. The view direction can be obtained by (focal-pos).normalized () */
        Vec3d focal;

        /** \brief Position of the camera. */
        Vec3d pos;

        /** \brief Up vector of the camera. */
        Vec3d view_up;

        /** \brief Near/far clipping planes depths */
        Vec2d clip;

        /** \brief Field of view angle in y direction (radians). */
        double fovy;

        // the following variables are the actual position and size of the window on the screen and NOT the viewport!
        // except for the size, which is the same the viewport is assumed to be centered and same size as the window.
        Vec2i window_size;
        Vec2i window_pos;

        /** \brief Computes View matrix for Camera (Based on gluLookAt)
          * \param[out] view_mat the resultant matrix
          */
        void computeViewMatrix(Affine3d& view_mat) const;

        /** \brief Computes Projection Matrix for Camera
          *  \param[out] proj the resultant matrix
          */
        void computeProjectionMatrix(Matx44d& proj) const;

        /** \brief converts point to window coordiantes
          * \param[in] pt xyz point to be converted
          * \param[out] window_cord vector containing the pts' window X,Y, Z and 1
          *
          * This function computes the projection and view matrix every time.
          * It is very inefficient to use this for every point in the point cloud!
          */
        void cvtWindowCoordinates (const cv::Point3f& pt, Vec4d& window_cord) const
        {
            Affine3d view;
            computeViewMatrix (view);

            Matx44d proj;
            computeProjectionMatrix (proj);
            cvtWindowCoordinates (pt, proj * view.matrix, window_cord);
            return;
        }

        /** \brief converts point to window coordiantes
          * \param[in] pt xyz point to be converted
          * \param[out] window_cord vector containing the pts' window X,Y, Z and 1
          * \param[in] composite_mat composite transformation matrix (proj*view)
          *
          * Use this function to compute window coordinates with a precomputed
          * transformation function.  The typical composite matrix will be
          * the projection matrix * the view matrix.  However, additional
          * matrices like a camera disortion matrix can also be added.
          */
        void cvtWindowCoordinates (const Point3f& pt, const Matx44d& composite_mat, Vec4d& window_cord) const
        {
            Vec4d pte (pt.x, pt.y, pt.z, 1);
            window_cord = composite_mat * pte;
            window_cord = window_cord/window_cord[3];

            window_cord[0] = (window_cord[0]+1.0) / 2.0*window_size[0];
            window_cord[1] = (window_cord[1]+1.0) / 2.0*window_size[1];
            window_cord[2] = (window_cord[2]+1.0) / 2.0;
        }
    };

}
