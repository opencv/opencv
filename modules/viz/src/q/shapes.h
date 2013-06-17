#pragma once

#include <Eigen/Core>
#include <opencv2/viz/types.hpp>
#include "precomp.hpp"

namespace temp_viz
{
    CV_EXPORTS vtkSmartPointer<vtkDataSet> createLine (const cv::Point3f& pt1, const cv::Point3f& pt2);
    CV_EXPORTS vtkSmartPointer<vtkDataSet> createSphere (const cv::Point3f &center, float radius, int sphere_resolution = 10);
    CV_EXPORTS vtkSmartPointer<vtkDataSet> createCylinder (const Point3f& pt_on_axis, const Point3f& axis_direction, double radius, int numsides = 30);
    CV_EXPORTS vtkSmartPointer<vtkDataSet> createPlane (const Vec4f& coefs);
    CV_EXPORTS vtkSmartPointer<vtkDataSet> createPlane (const Vec4f& coefs, const Point3f& pt);
    CV_EXPORTS vtkSmartPointer<vtkDataSet> create2DCircle (const Point3f& pt, double radius);
    CV_EXPORTS vtkSmartPointer<vtkDataSet> createCube(const Point3f& pt_min, const Point3f& pt_max);
//     CV_EXPORTS vtkSmartPointer<vtkDataSet> createCube (const Point3f& pt, const Quaternionf& qt,  );
//     CV_EXPORTS vtkSmartPointer<vtkDataSet> createCube (const Eigen::Vector3f &translation, const Eigen::Quaternionf &rotation, double width, double height, double depth);
//     CV_EXPORTS vtkSmartPointer<vtkDataSet> createCube (double x_min, double x_max, double y_min, double y_max, double z_min, double z_max);
//     
//     
    /** \brief Create a cylinder shape from a set of model coefficients.
      * \param[in] coefficients the model coefficients (point_on_axis, axis_direction, radius)
      * \param[in] numsides (optional) the number of sides used for rendering the cylinder
      *
      * \code
      * // The following are given (or computed using sample consensus techniques -- see SampleConsensusModelCylinder)
      * // Eigen::Vector3f pt_on_axis, axis_direction;
      * // float radius;
      *
      * temp_viz::ModelCoefficients cylinder_coeff;
      * cylinder_coeff.values.resize (7);    // We need 7 values
      * cylinder_coeff.values[0] = pt_on_axis.x ();
      * cylinder_coeff.values[1] = pt_on_axis.y ();
      * cylinder_coeff.values[2] = pt_on_axis.z ();
      *
      * cylinder_coeff.values[3] = axis_direction.x ();
      * cylinder_coeff.values[4] = axis_direction.y ();
      * cylinder_coeff.values[5] = axis_direction.z ();
      *
      * cylinder_coeff.values[6] = radius;
      *
      * vtkSmartPointer<vtkDataSet> data = temp_viz::createCylinder (cylinder_coeff, numsides);
      * \endcode
      *
      * \ingroup visualization
      */
    CV_EXPORTS vtkSmartPointer<vtkDataSet> createCylinder (const temp_viz::ModelCoefficients &coefficients, int numsides = 30);


    /** \brief Create a planar shape from a set of model coefficients.
      * \param[in] coefficients the model coefficients (a, b, c, d with ax+by+cz+d=0)
      *
      * \code
      * // The following are given (or computed using sample consensus techniques -- see SampleConsensusModelPlane)
      * // Eigen::Vector4f plane_parameters;
      *
      * temp_viz::ModelCoefficients plane_coeff;
      * plane_coeff.values.resize (4);    // We need 4 values
      * plane_coeff.values[0] = plane_parameters.x ();
      * plane_coeff.values[1] = plane_parameters.y ();
      * plane_coeff.values[2] = plane_parameters.z ();
      * plane_coeff.values[3] = plane_parameters.w ();
      *
      * vtkSmartPointer<vtkDataSet> data = temp_viz::createPlane (plane_coeff);
      * \endcode
      *
      * \ingroup visualization
      */
    CV_EXPORTS vtkSmartPointer<vtkDataSet> createPlane (const temp_viz::ModelCoefficients &coefficients);

    /** \brief Create a planar shape from a set of model coefficients.
      * \param[in] coefficients the model coefficients (a, b, c, d with ax+by+cz+d=0)
      * \param[in] x,y,z projection of this point on the plane is used to get the center of the plane.
      * \ingroup visualization
      */
    CV_EXPORTS vtkSmartPointer<vtkDataSet> createPlane (const temp_viz::ModelCoefficients &coefficients, double x, double y, double z);
    
    /** \brief Create a 2d circle shape from a set of model coefficients.
      * \param[in] coefficients the model coefficients (x, y, radius)
      * \param[in] z (optional) specify a z value (default: 0)
      *
      * \code
      * // The following are given (or computed using sample consensus techniques -- see SampleConsensusModelCircle2D)
      * // float x, y, radius;
      *
      * temp_viz::ModelCoefficients circle_coeff;
      * circle_coeff.values.resize (3);    // We need 3 values
      * circle_coeff.values[0] = x;
      * circle_coeff.values[1] = y;
      * circle_coeff.values[2] = radius;
      *
      * vtkSmartPointer<vtkDataSet> data = temp_viz::create2DCircle (circle_coeff, z);
      * \endcode
      *
      * \ingroup visualization
      */
    CV_EXPORTS vtkSmartPointer<vtkDataSet> create2DCircle (const temp_viz::ModelCoefficients &coefficients, double z = 0.0);


    /** \brief Creaet a cube shape from a set of model coefficients.
      * \param[in] coefficients the cube coefficients (Tx, Ty, Tz, Qx, Qy, Qz, Qw, width, height, depth)
      * \ingroup visualization 
      */
    CV_EXPORTS vtkSmartPointer<vtkDataSet> createCube (const temp_viz::ModelCoefficients &coefficients);

    /** \brief Creaet a cube shape from a set of model coefficients.
      *
      * \param[in] translation a translation to apply to the cube from 0,0,0
      * \param[in] rotation a quaternion-based rotation to apply to the cube 
      * \param[in] width the cube's width
      * \param[in] height the cube's height
      * \param[in] depth the cube's depth
      * \ingroup visualization 
      */
    CV_EXPORTS vtkSmartPointer<vtkDataSet> createCube (const Eigen::Vector3f &translation, const Eigen::Quaternionf &rotation, double width, double height, double depth);
    
    /** \brief Create a cube from a set of bounding points
      * \param[in] x_min is the minimum x value of the box
      * \param[in] x_max is the maximum x value of the box
      * \param[in] y_min is the minimum y value of the box 
      * \param[in] y_max is the maximum y value of the box
      * \param[in] z_min is the minimum z value of the box
      * \param[in] z_max is the maximum z value of the box
      * \param[in] id the cube id/name (default: "cube")
      * \param[in] viewport (optional) the id of the new viewport (default: 0)
      */
    CV_EXPORTS vtkSmartPointer<vtkDataSet> createCube (double x_min, double x_max, double y_min, double y_max, double z_min, double z_max);
    
    /** \brief Allocate a new unstructured grid smartpointer. For internal use only.
      * \param[out] polydata the resultant unstructured grid. 
      */
    CV_EXPORTS void allocVtkUnstructuredGrid (vtkSmartPointer<vtkUnstructuredGrid> &polydata);
}
