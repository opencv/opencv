#include "precomp.hpp"

class temp_viz::Widget::Impl
{
public:
    vtkSmartPointer<vtkProp> actor;
    int ref_counter;

    Impl() : actor(vtkSmartPointer<vtkLODActor>::New()) {}
    
    Impl(bool text_widget) 
    {
        if (text_widget)
            actor = vtkSmartPointer<vtkTextActor>::New();
        else
            actor = vtkSmartPointer<vtkLeaderActor2D>::New();
    }

    void setColor(const Color& color)
    {
        vtkSmartPointer<vtkLODActor> lod_actor = vtkLODActor::SafeDownCast(actor);
        Color c = vtkcolor(color);
        lod_actor->GetMapper ()->ScalarVisibilityOff ();
        lod_actor->GetProperty ()->SetColor (c.val);
        lod_actor->GetProperty ()->SetEdgeColor (c.val);
        lod_actor->GetProperty ()->SetAmbient (0.8);
        lod_actor->GetProperty ()->SetDiffuse (0.8);
        lod_actor->GetProperty ()->SetSpecular (0.8);
        lod_actor->GetProperty ()->SetLighting (0);
        lod_actor->Modified ();
    }

    void setPose(const Affine3f& pose)
    {
        vtkSmartPointer<vtkLODActor> lod_actor = vtkLODActor::SafeDownCast(actor);
        vtkSmartPointer<vtkMatrix4x4> matrix = convertToVtkMatrix(pose.matrix);
        lod_actor->SetUserMatrix (matrix);
        lod_actor->Modified ();
    }

    void updatePose(const Affine3f& pose)
    {
        vtkSmartPointer<vtkLODActor> lod_actor = vtkLODActor::SafeDownCast(actor);
        vtkSmartPointer<vtkMatrix4x4> matrix = lod_actor->GetUserMatrix();
        if (!matrix)
        {
            setPose(pose);
            return ;
        }
        Matx44f matrix_cv = convertToMatx(matrix);

        Affine3f updated_pose = pose * Affine3f(matrix_cv);
        matrix = convertToVtkMatrix(updated_pose.matrix);

        lod_actor->SetUserMatrix (matrix);
        lod_actor->Modified ();
    }

    Affine3f getPose() const
    {
        vtkSmartPointer<vtkLODActor> lod_actor = vtkLODActor::SafeDownCast(actor);
        vtkSmartPointer<vtkMatrix4x4> matrix = lod_actor->GetUserMatrix();
        Matx44f matrix_cv = convertToMatx(matrix);
        return Affine3f(matrix_cv);
    }

protected:

    static vtkSmartPointer<vtkMatrix4x4> convertToVtkMatrix (const cv::Matx44f& m)
    {
        vtkSmartPointer<vtkMatrix4x4> vtk_matrix = vtkSmartPointer<vtkMatrix4x4>::New ();
        for (int i = 0; i < 4; i++)
            for (int k = 0; k < 4; k++)
                vtk_matrix->SetElement(i, k, m(i, k));
        return vtk_matrix;
    }

    static cv::Matx44f convertToMatx(const vtkSmartPointer<vtkMatrix4x4>& vtk_matrix)
    {
        cv::Matx44f m;
        for (int i = 0; i < 4; i++)
            for (int k = 0; k < 4; k++)
                m(i, k) = vtk_matrix->GetElement (i, k);
        return m;
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////
/// stream accessor implementaion

vtkSmartPointer<vtkProp> temp_viz::WidgetAccessor::getActor(const Widget& widget)
{
    return widget.impl_->actor;
}

///////////////////////////////////////////////////////////////////////////////////////////////
/// widget implementaion

temp_viz::Widget::Widget() : impl_(0)
{
    create();
}

temp_viz::Widget::Widget(bool text_widget) : impl_(0)
{
    create(text_widget);
}

temp_viz::Widget::Widget(const Widget& other) : impl_(other.impl_)
{
    if (impl_)
        CV_XADD(&impl_->ref_counter, 1);
}

temp_viz::Widget& temp_viz::Widget::operator =(const Widget &other)
{
    if (this != &other)
    {
        release();
        impl_ = other.impl_;
        if (impl_)
            CV_XADD(&impl_->ref_counter, 1);
    }
    return *this;
}

temp_viz::Widget::~Widget()
{
    release();
}

void temp_viz::Widget::copyTo(Widget& /*dst*/)
{
    // TODO Deep copy the data if there is any
}

void temp_viz::Widget::setColor(const Color& color) { impl_->setColor(color); }
void temp_viz::Widget::setPose(const Affine3f& pose) { impl_->setPose(pose); }
void temp_viz::Widget::updatePose(const Affine3f& pose) { impl_->updatePose(pose); }
temp_viz::Affine3f temp_viz::Widget::getPose() const { return impl_->getPose(); }

void temp_viz::Widget::create()
{
    if (impl_)
        release();
    impl_ = new Impl();
    impl_->ref_counter = 1;
}

void temp_viz::Widget::release()
{
    if (impl_ && CV_XADD(&impl_->ref_counter, -1) == 1)
    {
        delete impl_;
        impl_ = 0;
    }
}

void temp_viz::Widget::create(bool text_widget)
{
    if (impl_)
        release();
    impl_ = new Impl(text_widget);
    impl_->ref_counter = 1;
}

