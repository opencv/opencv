#include <opencv2/viz/types.hpp>

//////////////////////////////////////////////////////////////////////////////////////////////////////
/// cv::Color

temp_viz::Color::Color() : Scalar(0, 0, 0) {}
temp_viz::Color::Color(double gray) : Scalar(gray, gray, gray) {}
temp_viz::Color::Color(double blue, double green, double red) : Scalar(blue, green, red) {}
temp_viz::Color::Color(const Scalar& color) : Scalar(color) {}

temp_viz::Color temp_viz::Color::black()   { return Color(  0,   0, 0); }
temp_viz::Color temp_viz::Color::green()   { return Color(  0, 255, 0); }
temp_viz::Color temp_viz::Color::blue()    { return Color(255,   0, 0); }
temp_viz::Color temp_viz::Color::cyan()    { return Color(255, 255, 0); }

temp_viz::Color temp_viz::Color::red()     { return Color(  0,   0, 255); }
temp_viz::Color temp_viz::Color::magenta() { return Color(  0, 255, 255); }
temp_viz::Color temp_viz::Color::yellow()  { return Color(255,   0, 255); }
temp_viz::Color temp_viz::Color::white()   { return Color(255, 255, 255); }

temp_viz::Color temp_viz::Color::gray()    { return Color(128, 128, 128); }

class temp_viz::Widget::Impl
{
public:
    String id;
    vtkSmartPointer<vtkLODActor> actor;
    
    Impl() 
    {
        actor = vtkSmartPointer<vtkLODActor>::New ();
    }
    
    vtkSmartPointer<vtkLODActor> getActor()
    {
        return actor;
    }
    
    void setColor(const Color & color)
    {
        Color c = vtkcolor(color);
        actor->GetMapper ()->ScalarVisibilityOff ();
        actor->GetProperty ()->SetColor (c.val);
        actor->GetProperty ()->SetEdgeColor (c.val);
        actor->GetProperty ()->SetAmbient (0.8);
        actor->GetProperty ()->SetDiffuse (0.8);
        actor->GetProperty ()->SetSpecular (0.8);
        actor->GetProperty ()->SetLighting (0);
        actor->Modified ();
    }
    
    void setPose(const Affine3f &pose)
    {
        vtkSmartPointer<vtkMatrix4x4> matrix = convertToVtkMatrix(pose.matrix);
        actor->SetUserMatrix (matrix);
        actor->Modified ();
    }

    void updatePose(const Affine3f &pose)
    {
        vtkSmartPointer<vtkMatrix4x4> matrix = actor->GetUserMatrix();
        Matx44f matrix_cv;
        convertToCvMatrix(matrix, matrix_cv);
        matrix = convertToVtkMatrix ((pose * Affine3f(matrix_cv)).matrix);

        actor->SetUserMatrix (matrix);
        actor->Modified ();
    }
    
    Affine3f getPose() const
    {
        vtkSmartPointer<vtkMatrix4x4> matrix = actor->GetUserMatrix();
        Matx44f matrix_cv;
        convertToCvMatrix(matrix, matrix_cv);
        return Affine3f(matrix_cv);
    }
    
protected:
    
    vtkSmartPointer<vtkMatrix4x4> convertToVtkMatrix (const cv::Matx44f &m) const
    {
        vtkSmartPointer<vtkMatrix4x4> vtk_matrix = vtkSmartPointer<vtkMatrix4x4>::New ();
        for (int i = 0; i < 4; i++)
            for (int k = 0; k < 4; k++)
                vtk_matrix->SetElement(i, k, m(i, k));
        return vtk_matrix;
    }
    
    void convertToCvMatrix (const vtkSmartPointer<vtkMatrix4x4> &vtk_matrix, cv::Matx44f &m) const
    {
        for (int i = 0; i < 4; i++)
            for (int k = 0; k < 4; k++)
                m(i,k) = vtk_matrix->GetElement (i, k);
    }    
};

temp_viz::Widget::Widget()
{
    impl_ = new Impl();
}

temp_viz::Widget::Widget(const Widget &other)
{
    impl_ = other.impl_;
}

temp_viz::Widget& temp_viz::Widget::operator =(const Widget &other)
{
    if (this != &other)
        impl_ = other.impl_;
    return *this;
}

void temp_viz::Widget::copyTo(Widget &dst)
{
    // TODO Deep copy the data if there is any
}

void temp_viz::Widget::setColor(const Color &color)
{
    impl_->setColor(color);
}

void temp_viz::Widget::setPose(const Affine3f &pose)
{
    impl_->setPose(pose);
}

void temp_viz::Widget::updatePose(const Affine3f &pose)
{
    impl_->updatePose(pose);
}

temp_viz::Affine3f temp_viz::Widget::getPose() const
{
    return impl_->getPose();
}

#include "opencv2/viz/widget_accessor.hpp"

vtkSmartPointer<vtkLODActor> temp_viz::WidgetAccessor::getActor(const temp_viz::Widget &widget)
{
    return widget.impl_->actor;
}

temp_viz::LineWidget::LineWidget(const Point3f &pt1, const Point3f &pt2, const Color &color) : Widget()
{
    // Create the line and set actor's data
    vtkSmartPointer<vtkDataSetMapper> mapper = vtkSmartPointer<vtkDataSetMapper>::New ();
    mapper->SetInput(createLine(pt1,pt2));
    temp_viz::WidgetAccessor::getActor(*this)->SetMapper(mapper);
    setColor(color);
}