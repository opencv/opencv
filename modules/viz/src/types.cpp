#include "precomp.hpp"

//////////////////////////////////////////////////////////////////////////////////////////////////////
/// cv::viz::Color

cv::viz::Color::Color() : Scalar(0, 0, 0) {}
cv::viz::Color::Color(double gray) : Scalar(gray, gray, gray) {}
cv::viz::Color::Color(double blue, double green, double red) : Scalar(blue, green, red) {}
cv::viz::Color::Color(const Scalar& color) : Scalar(color) {}

cv::viz::Color cv::viz::Color::black()   { return Color(  0,   0, 0); }
cv::viz::Color cv::viz::Color::green()   { return Color(  0, 255, 0); }
cv::viz::Color cv::viz::Color::blue()    { return Color(255,   0, 0); }
cv::viz::Color cv::viz::Color::cyan()    { return Color(255, 255, 0); }

cv::viz::Color cv::viz::Color::red()     { return Color(  0,   0, 255); }
cv::viz::Color cv::viz::Color::magenta() { return Color(  0, 255, 255); }
cv::viz::Color cv::viz::Color::yellow()  { return Color(255,   0, 255); }
cv::viz::Color cv::viz::Color::white()   { return Color(255, 255, 255); }

cv::viz::Color cv::viz::Color::gray()    { return Color(128, 128, 128); }

////////////////////////////////////////////////////////////////////
/// cv::viz::KeyboardEvent

cv::viz::KeyboardEvent::KeyboardEvent (bool _action, const std::string& _key_sym, unsigned char key, bool alt, bool ctrl, bool shift)
  : action_ (_action), modifiers_ (0), key_code_(key), key_sym_ (_key_sym)
{
  if (alt)
    modifiers_ = Alt;

  if (ctrl)
    modifiers_ |= Ctrl;

  if (shift)
    modifiers_ |= Shift;
}

bool cv::viz::KeyboardEvent::isAltPressed () const { return (modifiers_ & Alt) != 0; }
bool cv::viz::KeyboardEvent::isCtrlPressed () const { return (modifiers_ & Ctrl) != 0; }
bool cv::viz::KeyboardEvent::isShiftPressed () const { return (modifiers_ & Shift) != 0; }
unsigned char cv::viz::KeyboardEvent::getKeyCode () const { return key_code_; }
const cv::String& cv::viz::KeyboardEvent::getKeySym () const { return key_sym_; }
bool cv::viz::KeyboardEvent::keyDown () const { return action_; }
bool cv::viz::KeyboardEvent::keyUp () const { return !action_; }

////////////////////////////////////////////////////////////////////
/// cv::viz::MouseEvent

cv::viz::MouseEvent::MouseEvent (const Type& _type, const MouseButton& _button, const Point& _p,  bool alt, bool ctrl, bool shift)
    : type(_type), button(_button), pointer(_p), key_state(0)
{
    if (alt)
        key_state = KeyboardEvent::Alt;

    if (ctrl)
        key_state |= KeyboardEvent::Ctrl;

    if (shift)
        key_state |= KeyboardEvent::Shift;
}

////////////////////////////////////////////////////////////////////
/// cv::viz::Mesh3d

struct cv::viz::Mesh3d::loadMeshImpl
{
    static cv::viz::Mesh3d loadMesh(const String &file)
    {
        Mesh3d mesh;

        vtkSmartPointer<vtkPLYReader> reader = vtkSmartPointer<vtkPLYReader>::New();
        reader->SetFileName(file.c_str());
        reader->Update();
        vtkSmartPointer<vtkPolyData> poly_data = reader->GetOutput ();
        
        vtkSmartPointer<vtkPoints> mesh_points = poly_data->GetPoints ();
        vtkIdType nr_points = mesh_points->GetNumberOfPoints ();
        //vtkIdType nr_polygons = poly_data->GetNumberOfPolys ();

        mesh.cloud.create(1, nr_points, CV_32FC3);

        Vec3f *mesh_cloud = mesh.cloud.ptr<Vec3f>();
        for (vtkIdType i = 0; i < mesh_points->GetNumberOfPoints (); i++)
        {
            Vec3d point;
            mesh_points->GetPoint (i, point.val);
            mesh_cloud[i] = point;
        }

        // Then the color information, if any
        vtkUnsignedCharArray* poly_colors = NULL;
        if (poly_data->GetPointData() != NULL)
            poly_colors = vtkUnsignedCharArray::SafeDownCast (poly_data->GetPointData ()->GetScalars ("Colors"));

        // some applications do not save the name of scalars (including PCL's native vtk_io)
        if (!poly_colors && poly_data->GetPointData () != NULL)
            poly_colors = vtkUnsignedCharArray::SafeDownCast (poly_data->GetPointData ()->GetScalars ("scalars"));

        if (!poly_colors && poly_data->GetPointData () != NULL)
            poly_colors = vtkUnsignedCharArray::SafeDownCast (poly_data->GetPointData ()->GetScalars ("RGB"));

        // TODO: currently only handles rgb values with 3 components
        if (poly_colors && (poly_colors->GetNumberOfComponents () == 3))
        {
            mesh.colors.create(1, nr_points, CV_8UC3);
            Vec3b *mesh_colors = mesh.colors.ptr<cv::Vec3b>();

            for (vtkIdType i = 0; i < mesh_points->GetNumberOfPoints (); i++)
            {
                Vec3b point_color;
                poly_colors->GetTupleValue (i, point_color.val);

                std::swap(point_color[0], point_color[2]); // RGB -> BGR
                mesh_colors[i] = point_color;
            }
        }
        else
            mesh.colors.release();

        // Now handle the polygons
        vtkIdType* cell_points;
        vtkIdType nr_cell_points;
        vtkCellArray * mesh_polygons = poly_data->GetPolys ();
        mesh_polygons->InitTraversal ();
        
        mesh.polygons.create(1, mesh_polygons->GetSize(), CV_32SC1);
        
        int* polygons = mesh.polygons.ptr<int>();
        while (mesh_polygons->GetNextCell (nr_cell_points, cell_points))
        {
            *polygons++ = nr_cell_points;
            for (int i = 0; i < nr_cell_points; ++i)
                *polygons++ = static_cast<int> (cell_points[i]);
        }

        return mesh;
    }
};

cv::viz::Mesh3d cv::viz::Mesh3d::loadMesh(const String& file)
{
    return loadMeshImpl::loadMesh(file);
}

////////////////////////////////////////////////////////////////////
/// Camera implementation

cv::viz::Camera::Camera(float f_x, float f_y, float c_x, float c_y, const Size &window_size)
{
    CV_Assert(window_size.width > 0 && window_size.height > 0);
    setClip(Vec2d(0.01, 1000.01));// Default clipping
    
    fov_[0] = (atan2(c_x,f_x) + atan2(window_size.width-c_x,f_x)) * 180 / CV_PI;
    fov_[1] = (atan2(c_y,f_y) + atan2(window_size.height-c_y,f_y)) * 180 / CV_PI;
    
    principal_point_[0] = c_x;
    principal_point_[1] = c_y;
    
    focal_[0] = f_x;
    focal_[1] = f_y;
}

cv::viz::Camera::Camera(const Vec2f &fov, const Size &window_size)
{
    CV_Assert(window_size.width > 0 && window_size.height > 0);
    setClip(Vec2d(0.01, 1000.01)); // Default clipping
    principal_point_ = Vec2f(-1.0f, -1.0f); // Default symmetric lens
    focal_ = Vec2f(-1.0f, -1.0f);
    setFov(fov);
    setWindowSize(window_size);
}

cv::viz::Camera::Camera(const cv::Mat & K, const Size &window_size)
{
    CV_Assert(K.rows == 3 && K.cols == 3);
    CV_Assert(window_size.width > 0 && window_size.height > 0);
    
    float f_x = K.at<float>(0,0);
    float f_y = K.at<float>(1,1);
    float c_x = K.at<float>(0,2);
    float c_y = K.at<float>(1,2);
    Camera(f_x, f_y, c_x, c_y, window_size);
}

void cv::viz::Camera::setWindowSize(const Size &window_size)
{
    CV_Assert(window_size.width > 0 && window_size.height > 0);
    
    // Vertical field of view is fixed! 
    // Horizontal field of view is expandable based on the aspect ratio
    float aspect_ratio_new = static_cast<float>(window_size.width) / static_cast<float>(window_size.height);
    
    if (principal_point_[0] < 0.0f)
        fov_[0] = 2.f * atan(tan(fov_[1] * 0.5) * aspect_ratio_new); // This assumes that the lens is symmetric!
    else
        fov_[0] = (atan2(principal_point_[0],focal_[0]) + atan2(window_size.width-principal_point_[0],focal_[0])) * 180 / CV_PI;
    
    window_size_ = window_size;
}

void cv::viz::Camera::computeProjectionMatrix(Matx44f &proj) const
{
    double top    = clip_[0] * tan (0.5 * fov_[1]);
    double left   = -(top * window_size_.width) / window_size_.height;
    double right  = -left;
    double bottom = -top;

    double temp1 = 2.0 * clip_[0];
    double temp2 = 1.0 / (right - left);
    double temp3 = 1.0 / (top - bottom);
    double temp4 = 1.0 / clip_[1] - clip_[0];

    proj = Matx44d::zeros();

    proj(0,0) = temp1 * temp2;
    proj(1,1) = temp1 * temp3;
    proj(0,2) = (right + left) * temp2;
    proj(1,2) = (top + bottom) * temp3;
    proj(2,2) = (-clip_[1] - clip_[0]) * temp4;
    proj(3,2) = -1.0;
    proj(2,3) = (-temp1 * clip_[1]) * temp4;
}