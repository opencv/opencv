#pragma once

#include <opencv2/core/cvdef.h>
#include <opencv2/core.hpp>
#include <opencv2/viz/types.hpp>

namespace cv
{
    namespace viz
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

    }

}
