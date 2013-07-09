#include "precomp.hpp"

///////////////////////////////////////////////////////////////////////////////////////////////
/// widget implementation

class temp_viz::Widget::Impl
{
public:
    vtkSmartPointer<vtkProp> actor;
    
    Impl() : actor(0) {}
};

temp_viz::Widget::Widget() : impl_(0)
{
    impl_ = new Impl();
}

temp_viz::Widget::Widget(const Widget &other) : impl_(other.impl_)
{
    
}

temp_viz::Widget& temp_viz::Widget::operator =(const Widget &other)
{
    if (this != &other)
    {
        delete impl_;
        impl_ = other.impl_;
    }
    return *this;
}

temp_viz::Widget::~Widget()
{
    if (impl_)
    {
        delete impl_;
        impl_ = 0;
    }
}



// class temp_viz::Widget::Impl
// {
// public:
//     vtkSmartPointer<vtkProp> actor;
//     int ref_counter;
// 
//     Impl() : actor(vtkSmartPointer<vtkLODActor>::New()) {}
//     
//     Impl(bool text_widget) 
//     {
//         if (text_widget)
//             actor = vtkSmartPointer<vtkTextActor>::New();
//         else
//             actor = vtkSmartPointer<vtkLeaderActor2D>::New();
//     }
// 
//     void setColor(const Color& color)
//     {
//         vtkLODActor *lod_actor = vtkLODActor::SafeDownCast(actor);
//         Color c = vtkcolor(color);
//         lod_actor->GetMapper ()->ScalarVisibilityOff ();
//         lod_actor->GetProperty ()->SetColor (c.val);
//         lod_actor->GetProperty ()->SetEdgeColor (c.val);
//         lod_actor->GetProperty ()->SetAmbient (0.8);
//         lod_actor->GetProperty ()->SetDiffuse (0.8);
//         lod_actor->GetProperty ()->SetSpecular (0.8);
//         lod_actor->GetProperty ()->SetLighting (0);
//         lod_actor->Modified ();
//     }
// 
//     void setPose(const Affine3f& pose)
//     {
//         vtkLODActor *lod_actor = vtkLODActor::SafeDownCast(actor);
//         vtkSmartPointer<vtkMatrix4x4> matrix = convertToVtkMatrix(pose.matrix);
//         lod_actor->SetUserMatrix (matrix);
//         lod_actor->Modified ();
//     }
// 
//     void updatePose(const Affine3f& pose)
//     {
//         vtkLODActor *lod_actor = vtkLODActor::SafeDownCast(actor);
//         vtkSmartPointer<vtkMatrix4x4> matrix = lod_actor->GetUserMatrix();
//         if (!matrix)
//         {
//             setPose(pose);
//             return ;
//         }
//         Matx44f matrix_cv = convertToMatx(matrix);
// 
//         Affine3f updated_pose = pose * Affine3f(matrix_cv);
//         matrix = convertToVtkMatrix(updated_pose.matrix);
// 
//         lod_actor->SetUserMatrix (matrix);
//         lod_actor->Modified ();
//     }
// 
//     Affine3f getPose() const
//     {
//         vtkLODActor *lod_actor = vtkLODActor::SafeDownCast(actor);
//         vtkSmartPointer<vtkMatrix4x4> matrix = lod_actor->GetUserMatrix();
//         Matx44f matrix_cv = convertToMatx(matrix);
//         return Affine3f(matrix_cv);
//     }
// 
// protected:
// 
//     static vtkSmartPointer<vtkMatrix4x4> convertToVtkMatrix (const cv::Matx44f& m)
//     {
//         vtkSmartPointer<vtkMatrix4x4> vtk_matrix = vtkSmartPointer<vtkMatrix4x4>::New ();
//         for (int i = 0; i < 4; i++)
//             for (int k = 0; k < 4; k++)
//                 vtk_matrix->SetElement(i, k, m(i, k));
//         return vtk_matrix;
//     }
// 
//     static cv::Matx44f convertToMatx(const vtkSmartPointer<vtkMatrix4x4>& vtk_matrix)
//     {
//         cv::Matx44f m;
//         for (int i = 0; i < 4; i++)
//             for (int k = 0; k < 4; k++)
//                 m(i, k) = vtk_matrix->GetElement (i, k);
//         return m;
//     }
// };


///////////////////////////////////////////////////////////////////////////////////////////////
/// stream accessor implementaion

vtkSmartPointer<vtkProp> temp_viz::WidgetAccessor::getActor(const Widget& widget)
{
    return widget.impl_->actor;
}

void temp_viz::WidgetAccessor::setVtkProp(Widget& widget, vtkSmartPointer<vtkProp> actor)
{
    widget.impl_->actor = actor;
}

///////////////////////////////////////////////////////////////////////////////////////////////
/// widget3D implementation

struct temp_viz::Widget3D::MatrixConverter
{
    static cv::Matx44f convertToMatx(const vtkSmartPointer<vtkMatrix4x4>& vtk_matrix)
    {
        cv::Matx44f m;
        for (int i = 0; i < 4; i++)
            for (int k = 0; k < 4; k++)
                m(i, k) = vtk_matrix->GetElement (i, k);
        return m;
    }
    
    static vtkSmartPointer<vtkMatrix4x4> convertToVtkMatrix (const cv::Matx44f& m)
    {
        vtkSmartPointer<vtkMatrix4x4> vtk_matrix = vtkSmartPointer<vtkMatrix4x4>::New ();
        for (int i = 0; i < 4; i++)
            for (int k = 0; k < 4; k++)
                vtk_matrix->SetElement(i, k, m(i, k));
        return vtk_matrix;
    }
};

temp_viz::Widget3D::Widget3D()
{

}

void temp_viz::Widget3D::setPose(const Affine3f &pose)
{
    vtkProp3D *actor = vtkProp3D::SafeDownCast(WidgetAccessor::getActor(*this));
    CV_Assert(actor);
    
    vtkSmartPointer<vtkMatrix4x4> matrix = convertToVtkMatrix(pose.matrix);
    actor->SetUserMatrix (matrix);
    actor->Modified ();
}

void temp_viz::Widget3D::updatePose(const Affine3f &pose)
{
    vtkProp3D *actor = vtkProp3D::SafeDownCast(WidgetAccessor::getActor(*this));
    CV_Assert(actor);
    
    vtkSmartPointer<vtkMatrix4x4> matrix = actor->GetUserMatrix();
    if (!matrix)
    {
        setPose(pose);
        return ;
    }
    Matx44f matrix_cv = MatrixConverter::convertToMatx(matrix);

    Affine3f updated_pose = pose * Affine3f(matrix_cv);
    matrix = MatrixConverter::convertToVtkMatrix(updated_pose.matrix);

    actor->SetUserMatrix (matrix);
    actor->Modified ();
}

temp_viz::Affine3f temp_viz::Widget3D::getPose() const
{
    vtkProp3D *actor = vtkProp3D::SafeDownCast(WidgetAccessor::getActor(*this));
    CV_Assert(actor);
    
    vtkSmartPointer<vtkMatrix4x4> matrix = actor->GetUserMatrix();
    Matx44f matrix_cv = MatrixConverter::convertToMatx(matrix);
    return Affine3f(matrix_cv);
}

void temp_viz::Widget3D::setColor(const Color &color)
{
    // Cast to actor instead of prop3d since prop3d doesn't provide getproperty
    vtkActor *actor = vtkActor::SafeDownCast(WidgetAccessor::getActor(*this));
    CV_Assert(actor);
    
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

///////////////////////////////////////////////////////////////////////////////////////////////
/// widget2D implementation
        
temp_viz::Widget2D::Widget2D()
{

}
