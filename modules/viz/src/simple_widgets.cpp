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