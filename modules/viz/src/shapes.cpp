#include <q/shapes.h>

inline float rad2deg (float alpha)
{ return (alpha * 57.29578f); }

inline double rad2deg (double alpha){return (alpha * 57.29578);}

vtkSmartPointer<vtkDataSet> temp_viz::createCylinder (const cv::Point3f& pt_on_axis, const cv::Point3f& axis_direction, double radius, int numsides)
{
    const cv::Point3f pt2 = pt_on_axis + axis_direction;
    vtkSmartPointer<vtkLineSource> line = vtkSmartPointer<vtkLineSource>::New ();
    line->SetPoint1 (pt_on_axis.x, pt_on_axis.y, pt_on_axis.z);
    line->SetPoint2 (pt2.x, pt2.y, pt2.z);
    
    vtkSmartPointer<vtkTubeFilter> tuber = vtkSmartPointer<vtkTubeFilter>::New ();
    tuber->SetInputConnection (line->GetOutputPort ());
    tuber->SetRadius (radius);
    tuber->SetNumberOfSides (numsides);
    return (tuber->GetOutput ());
}

vtkSmartPointer<vtkDataSet> temp_viz::createPlane (const cv::Vec4f& coefs)
{
    vtkSmartPointer<vtkPlaneSource> plane = vtkSmartPointer<vtkPlaneSource>::New ();
    plane->SetNormal (coefs[0], coefs[1], coefs[2]);
    double norm = cv::norm (cv::Vec3f (coefs[0], coefs[1], coefs[2]));
    plane->Push (-coefs[3] / norm);
    return (plane->GetOutput ());
}

vtkSmartPointer<vtkDataSet> temp_viz::createPlane(const cv::Vec4f& coefs, const cv::Point3f& pt)
{
    vtkSmartPointer<vtkPlaneSource> plane = vtkSmartPointer<vtkPlaneSource>::New ();
    cv::Point3f coefs3 (coefs[0], coefs[1], coefs[2]);    
    double norm_sqr = 1.0 / coefs3.dot (coefs3);
    plane->SetNormal (coefs[0], coefs[1], coefs[2]);

    double t = coefs3.dot (pt) + coefs[3];
    cv::Vec3f p_center;
    p_center = pt - coefs3 * t * norm_sqr;
    plane->SetCenter (p_center[0], p_center[1], p_center[2]);

    return (plane->GetOutput ());
}

vtkSmartPointer<vtkDataSet> temp_viz::create2DCircle (const cv::Point3f& pt, double radius)
{
    vtkSmartPointer<vtkDiskSource> disk = vtkSmartPointer<vtkDiskSource>::New ();
    // Maybe the resolution should be lower e.g. 50 or 25
    disk->SetCircumferentialResolution (100);
    disk->SetInnerRadius (radius - 0.001);
    disk->SetOuterRadius (radius + 0.001);
    disk->SetCircumferentialResolution (20);

    // Set the circle origin
    vtkSmartPointer<vtkTransform> t = vtkSmartPointer<vtkTransform>::New ();
    t->Identity ();
    t->Translate (pt.x, pt.y, pt.z);

    vtkSmartPointer<vtkTransformPolyDataFilter> tf = vtkSmartPointer<vtkTransformPolyDataFilter>::New ();
    tf->SetTransform (t);
    tf->SetInputConnection (disk->GetOutputPort ());
    
    return (tf->GetOutput ());
}

vtkSmartPointer<vtkDataSet> temp_viz::createCube(const cv::Point3f& pt_min, const cv::Point3f& pt_max)
{
    vtkSmartPointer<vtkCubeSource> cube = vtkSmartPointer<vtkCubeSource>::New ();
    cube->SetBounds (pt_min.x, pt_max.x, pt_min.y, pt_max.y, pt_min.z, pt_max.z);
    return (cube->GetOutput ());
}

vtkSmartPointer<vtkDataSet> temp_viz::createSphere (const Point3f& pt, double radius)
{
    vtkSmartPointer<vtkSphereSource> sphere = vtkSmartPointer<vtkSphereSource>::New ();
    sphere->SetRadius (radius);
    sphere->SetCenter (pt.x, pt.y, pt.z);
    sphere->SetPhiResolution (10);
    sphere->SetThetaResolution (10);
    sphere->LatLongTessellationOff ();
    sphere->Update ();
    
    return (sphere->GetOutput ());
}

vtkSmartPointer<vtkDataSet> temp_viz::createArrow (const Point3f& pt1, const Point3f& pt2)
{
    vtkSmartPointer<vtkArrowSource> arrowSource = vtkSmartPointer<vtkArrowSource>::New ();
    
    float startPoint[3], endPoint[3];
    startPoint[0] = pt1.x;
    startPoint[1] = pt1.y;
    startPoint[2] = pt1.z;
    endPoint[0] = pt2.x;
    endPoint[1] = pt2.y;
    endPoint[2] = pt2.z;
    float normalizedX[3], normalizedY[3], normalizedZ[3];
    
    // The X axis is a vector from start to end
    vtkMath::Subtract(endPoint, startPoint, normalizedX);
    float length = vtkMath::Norm(normalizedX);
    vtkMath::Normalize(normalizedX);

    // The Z axis is an arbitrary vecotr cross X
    float arbitrary[3];
    arbitrary[0] = vtkMath::Random(-10,10);
    arbitrary[1] = vtkMath::Random(-10,10);
    arbitrary[2] = vtkMath::Random(-10,10);
    vtkMath::Cross(normalizedX, arbitrary, normalizedZ);
    vtkMath::Normalize(normalizedZ);

    // The Y axis is Z cross X
    vtkMath::Cross(normalizedZ, normalizedX, normalizedY);
    vtkSmartPointer<vtkMatrix4x4> matrix = vtkSmartPointer<vtkMatrix4x4>::New();

    // Create the direction cosine matrix
    matrix->Identity();
    for (unsigned int i = 0; i < 3; i++)
    {
        matrix->SetElement(i, 0, normalizedX[i]);
        matrix->SetElement(i, 1, normalizedY[i]);
        matrix->SetElement(i, 2, normalizedZ[i]);
    }    

    // Apply the transforms
    vtkSmartPointer<vtkTransform> transform = 
    vtkSmartPointer<vtkTransform>::New();
    transform->Translate(startPoint);
    transform->Concatenate(matrix);
    transform->Scale(length, length, length);

    // Transform the polydata
    vtkSmartPointer<vtkTransformPolyDataFilter> transformPD = 
    vtkSmartPointer<vtkTransformPolyDataFilter>::New();
    transformPD->SetTransform(transform);
    transformPD->SetInputConnection(arrowSource->GetOutputPort());
    
    return (transformPD->GetOutput());
}

////////////////////////////////////////////////////////////////////////////////////////////
vtkSmartPointer<vtkDataSet> temp_viz::createCylinder (const temp_viz::ModelCoefficients &coefficients, int numsides)
{
  vtkSmartPointer<vtkLineSource> line = vtkSmartPointer<vtkLineSource>::New ();
  line->SetPoint1 (coefficients.values[0], coefficients.values[1], coefficients.values[2]);
  line->SetPoint2 (coefficients.values[3]+coefficients.values[0], coefficients.values[4]+coefficients.values[1], coefficients.values[5]+coefficients.values[2]);

  vtkSmartPointer<vtkTubeFilter> tuber = vtkSmartPointer<vtkTubeFilter>::New ();
  tuber->SetInputConnection (line->GetOutputPort ());
  tuber->SetRadius (coefficients.values[6]);
  tuber->SetNumberOfSides (numsides);

  return (tuber->GetOutput ());
}

////////////////////////////////////////////////////////////////////////////////////////////
vtkSmartPointer<vtkDataSet> temp_viz::createCube (const temp_viz::ModelCoefficients &coefficients)
{
  // coefficients = [Tx, Ty, Tz, Qx, Qy, Qz, Qw, width, height, depth]
  vtkSmartPointer<vtkTransform> t = vtkSmartPointer<vtkTransform>::New ();
  t->Identity ();
  t->Translate (coefficients.values[0], coefficients.values[1], coefficients.values[2]);

  Eigen::AngleAxisf a (Eigen::Quaternionf (coefficients.values[6], coefficients.values[3],
                                           coefficients.values[4], coefficients.values[5]));
  t->RotateWXYZ (rad2deg (a.angle ()), a.axis ()[0], a.axis ()[1], a.axis ()[2]);

  vtkSmartPointer<vtkCubeSource> cube = vtkSmartPointer<vtkCubeSource>::New ();
  cube->SetXLength (coefficients.values[7]);
  cube->SetYLength (coefficients.values[8]);
  cube->SetZLength (coefficients.values[9]);

  vtkSmartPointer<vtkTransformPolyDataFilter> tf = vtkSmartPointer<vtkTransformPolyDataFilter>::New ();
  tf->SetTransform (t);
  tf->SetInputConnection (cube->GetOutputPort ());

  return (tf->GetOutput ());
}

////////////////////////////////////////////////////////////////////////////////////////////
vtkSmartPointer<vtkDataSet> temp_viz::createCube (const Eigen::Vector3f &translation, const Eigen::Quaternionf &rotation, double width, double height, double depth)
{
  // coefficients = [Tx, Ty, Tz, Qx, Qy, Qz, Qw, width, height, depth]
  vtkSmartPointer<vtkTransform> t = vtkSmartPointer<vtkTransform>::New ();
  t->Identity ();
  t->Translate (translation.x (), translation.y (), translation.z ());

  Eigen::AngleAxisf a (rotation);
  t->RotateWXYZ (rad2deg (a.angle ()), a.axis ()[0], a.axis ()[1], a.axis ()[2]);

  vtkSmartPointer<vtkCubeSource> cube = vtkSmartPointer<vtkCubeSource>::New ();
  cube->SetXLength (width);
  cube->SetYLength (height);
  cube->SetZLength (depth);

  vtkSmartPointer<vtkTransformPolyDataFilter> tf = vtkSmartPointer<vtkTransformPolyDataFilter>::New ();
  tf->SetTransform (t);
  tf->SetInputConnection (cube->GetOutputPort ());

  return (tf->GetOutput ());
}

////////////////////////////////////////////////////////////////////////////////////////////
vtkSmartPointer<vtkDataSet> temp_viz::createCube (double x_min, double x_max, double y_min, double y_max, double z_min, double z_max)
{
  vtkSmartPointer<vtkTransform> t = vtkSmartPointer<vtkTransform>::New ();
  vtkSmartPointer<vtkCubeSource> cube = vtkSmartPointer<vtkCubeSource>::New ();
  cube->SetBounds (x_min, x_max, y_min, y_max, z_min, z_max);
  return (cube->GetOutput ());
}

////////////////////////////////////////////////////////////////////////////////////////////
vtkSmartPointer<vtkDataSet> temp_viz::createPlane (const temp_viz::ModelCoefficients &coefficients)
{
  vtkSmartPointer<vtkPlaneSource> plane = vtkSmartPointer<vtkPlaneSource>::New ();
  plane->SetNormal (coefficients.values[0], coefficients.values[1], coefficients.values[2]);

  double norm_sqr = coefficients.values[0] * coefficients.values[0]
                  + coefficients.values[1] * coefficients.values[1]
                  + coefficients.values[2] * coefficients.values[2];

  plane->Push (-coefficients.values[3] / sqrt(norm_sqr));
  return (plane->GetOutput ());
}

////////////////////////////////////////////////////////////////////////////////////////////
vtkSmartPointer<vtkDataSet> temp_viz::createPlane (const temp_viz::ModelCoefficients &coefficients, double x, double y, double z)
{
  vtkSmartPointer<vtkPlaneSource> plane = vtkSmartPointer<vtkPlaneSource>::New ();


  double norm_sqr = 1.0 / (coefficients.values[0] * coefficients.values[0] +
                           coefficients.values[1] * coefficients.values[1] +
                           coefficients.values[2] * coefficients.values[2] );

//  double nx = coefficients.values [0] * norm;
//  double ny = coefficients.values [1] * norm;
//  double nz = coefficients.values [2] * norm;
//  double d  = coefficients.values [3] * norm;

//  plane->SetNormal (nx, ny, nz);
  plane->SetNormal (coefficients.values[0], coefficients.values[1], coefficients.values[2]);

  double t = x * coefficients.values[0] + y * coefficients.values[1] + z * coefficients.values[2] + coefficients.values[3];
  x -= coefficients.values[0] * t * norm_sqr;
  y -= coefficients.values[1] * t * norm_sqr;
  z -= coefficients.values[2] * t * norm_sqr;
  plane->SetCenter (x, y, z);

  return (plane->GetOutput ());
}


////////////////////////////////////////////////////////////////////////////////////////////
vtkSmartPointer<vtkDataSet> temp_viz::create2DCircle (const temp_viz::ModelCoefficients &coefficients, double z)
{
  vtkSmartPointer<vtkDiskSource> disk = vtkSmartPointer<vtkDiskSource>::New ();
  // Maybe the resolution should be lower e.g. 50 or 25
  disk->SetCircumferentialResolution (100);
  disk->SetInnerRadius (coefficients.values[2] - 0.001);
  disk->SetOuterRadius (coefficients.values[2] + 0.001);
  disk->SetCircumferentialResolution (20);

  // An alternative to <vtkDiskSource> could be <vtkRegularPolygonSource> with <vtkTubeFilter>
  /*
  vtkSmartPointer<vtkRegularPolygonSource> circle = vtkSmartPointer<vtkRegularPolygonSource>::New();
  circle->SetRadius (coefficients.values[2]);
  circle->SetNumberOfSides (100);

  vtkSmartPointer<vtkTubeFilter> tube = vtkSmartPointer<vtkTubeFilter>::New();
  tube->SetInput (circle->GetOutput());
  tube->SetNumberOfSides (25);
  tube->SetRadius (0.001);
  */

  // Set the circle origin
  vtkSmartPointer<vtkTransform> t = vtkSmartPointer<vtkTransform>::New ();
  t->Identity ();
  t->Translate (coefficients.values[0], coefficients.values[1], z);

  vtkSmartPointer<vtkTransformPolyDataFilter> tf = vtkSmartPointer<vtkTransformPolyDataFilter>::New ();
  tf->SetTransform (t);
  tf->SetInputConnection (disk->GetOutputPort ());
  /*
  tf->SetInputConnection (tube->GetOutputPort ());
  */

  return (tf->GetOutput ());
}

////////////////////////////////////////////////////////////////////////////////////////////
vtkSmartPointer<vtkDataSet> temp_viz::createSphere (const cv::Point3f& center, float radius, int sphere_resolution)
{
  // Set the sphere origin
  vtkSmartPointer<vtkTransform> t = vtkSmartPointer<vtkTransform>::New ();
  t->Identity ();
  t->Translate (center.x, center.y, center.z);

  vtkSmartPointer<vtkSphereSource> s_sphere = vtkSmartPointer<vtkSphereSource>::New ();
  s_sphere->SetRadius (radius);
  s_sphere->SetPhiResolution (sphere_resolution);
  s_sphere->SetThetaResolution (sphere_resolution);
  s_sphere->LatLongTessellationOff ();

  vtkSmartPointer<vtkTransformPolyDataFilter> tf = vtkSmartPointer<vtkTransformPolyDataFilter>::New ();
  tf->SetTransform (t);
  tf->SetInputConnection (s_sphere->GetOutputPort ());
  tf->Update ();

  return (tf->GetOutput ());
}

////////////////////////////////////////////////////////////////////////////////////////////
vtkSmartPointer<vtkDataSet> temp_viz::createLine (const cv::Point3f& pt1, const cv::Point3f& pt2)
{
  vtkSmartPointer<vtkLineSource> line = vtkSmartPointer<vtkLineSource>::New ();
  line->SetPoint1 (pt1.x, pt1.y, pt1.z);
  line->SetPoint2 (pt2.x, pt2.y, pt2.z);
  line->Update ();
  return line->GetOutput ();
}
//////////////////////////////////////////////////////////////////////////////////////////////
void temp_viz::allocVtkUnstructuredGrid (vtkSmartPointer<vtkUnstructuredGrid> &polydata)
{
  polydata = vtkSmartPointer<vtkUnstructuredGrid>::New ();
}


