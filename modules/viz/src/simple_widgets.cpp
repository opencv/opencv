#include "precomp.hpp"

///////////////////////////////////////////////////////////////////////////////////////////////
/// line widget implementation
temp_viz::LineWidget::LineWidget(const Point3f &pt1, const Point3f &pt2, const Color &color)
{
    vtkSmartPointer<vtkLineSource> line = vtkSmartPointer<vtkLineSource>::New();
    line->SetPoint1 (pt1.x, pt1.y, pt1.z);
    line->SetPoint2 (pt2.x, pt2.y, pt2.z);
    line->Update ();

    vtkSmartPointer<vtkDataSetMapper> mapper = vtkSmartPointer<vtkDataSetMapper>::New ();
    mapper->SetInput(line->GetOutput ());

    vtkSmartPointer<vtkLODActor> actor = WidgetAccessor::getActor(*this);
    actor->SetMapper(mapper);

    setColor(color);
}

void temp_viz::LineWidget::setLineWidth(float line_width)
{
    vtkSmartPointer<vtkLODActor> actor = WidgetAccessor::getActor(*this);
    actor->GetProperty()->SetLineWidth(line_width);
}

float temp_viz::LineWidget::getLineWidth()
{
    vtkSmartPointer<vtkLODActor> actor = WidgetAccessor::getActor(*this);
    return actor->GetProperty()->GetLineWidth();
}

///////////////////////////////////////////////////////////////////////////////////////////////
/// plane widget implementation

temp_viz::PlaneWidget::PlaneWidget(const Vec4f& coefs, const Color &color)
{
    vtkSmartPointer<vtkPlaneSource> plane = vtkSmartPointer<vtkPlaneSource>::New ();
    plane->SetNormal (coefs[0], coefs[1], coefs[2]);
    double norm = cv::norm(cv::Vec3f(coefs.val));
    plane->Push (-coefs[3] / norm);

    vtkSmartPointer<vtkDataSetMapper> mapper = vtkSmartPointer<vtkDataSetMapper>::New ();
    mapper->SetInput(plane->GetOutput ());

    vtkSmartPointer<vtkLODActor> actor = WidgetAccessor::getActor(*this);
    actor->SetMapper(mapper);

    setColor(color);
}

temp_viz::PlaneWidget::PlaneWidget(const Vec4f& coefs, const Point3f& pt, const Color &color)
{
    vtkSmartPointer<vtkPlaneSource> plane = vtkSmartPointer<vtkPlaneSource>::New ();
    cv::Point3f coefs3(coefs[0], coefs[1], coefs[2]);
    double norm_sqr = 1.0 / coefs3.dot (coefs3);
    plane->SetNormal(coefs[0], coefs[1], coefs[2]);

    double t = coefs3.dot(pt) + coefs[3];
    cv::Vec3f p_center = pt - coefs3 * t * norm_sqr;
    plane->SetCenter (p_center[0], p_center[1], p_center[2]);
    
    vtkSmartPointer<vtkDataSetMapper> mapper = vtkSmartPointer<vtkDataSetMapper>::New ();
    mapper->SetInput(plane->GetOutput ());

    vtkSmartPointer<vtkLODActor> actor = WidgetAccessor::getActor(*this);
    actor->SetMapper(mapper);

    setColor(color);
}

///////////////////////////////////////////////////////////////////////////////////////////////
/// sphere widget implementation

temp_viz::SphereWidget::SphereWidget(const cv::Point3f &center, float radius, int sphere_resolution, const Color &color)
{
    vtkSmartPointer<vtkSphereSource> sphere = vtkSmartPointer<vtkSphereSource>::New ();
    sphere->SetRadius (radius);
    sphere->SetCenter (center.x, center.y, center.z);
    sphere->SetPhiResolution (sphere_resolution);
    sphere->SetThetaResolution (sphere_resolution);
    sphere->LatLongTessellationOff ();
    sphere->Update ();
    
    vtkSmartPointer<vtkDataSetMapper> mapper = vtkSmartPointer<vtkDataSetMapper>::New ();
    mapper->SetInput(sphere->GetOutput ());

    vtkSmartPointer<vtkLODActor> actor = WidgetAccessor::getActor(*this);
    actor->SetMapper(mapper);

    setColor(color);
}

///////////////////////////////////////////////////////////////////////////////////////////////
/// arrow widget implementation

temp_viz::ArrowWidget::ArrowWidget(const Point3f& pt1, const Point3f& pt2, const Color &color)
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
    
    vtkSmartPointer<vtkDataSetMapper> mapper = vtkSmartPointer<vtkDataSetMapper>::New ();
    mapper->SetInput(transformPD->GetOutput ());

    vtkSmartPointer<vtkLODActor> actor = WidgetAccessor::getActor(*this);
    actor->SetMapper(mapper);
    
    setColor(color);
}

///////////////////////////////////////////////////////////////////////////////////////////////
/// circle widget implementation

temp_viz::CircleWidget::CircleWidget(const Point3f& pt, double radius, const Color &color)
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
    
    vtkSmartPointer<vtkDataSetMapper> mapper = vtkSmartPointer<vtkDataSetMapper>::New ();
    mapper->SetInput(tf->GetOutput ());

    vtkSmartPointer<vtkLODActor> actor = WidgetAccessor::getActor(*this);
    actor->SetMapper(mapper);
    
    setColor(color);
}

