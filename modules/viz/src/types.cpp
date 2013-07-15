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
    static cv::viz::Mesh3d::Ptr loadMesh(const String &file)
    {
        Mesh3d::Ptr mesh = new Mesh3d();

        vtkSmartPointer<vtkPLYReader> reader = vtkSmartPointer<vtkPLYReader>::New();
        reader->SetFileName(file.c_str());
        reader->Update();
        vtkSmartPointer<vtkPolyData> poly_data = reader->GetOutput ();
        
        vtkSmartPointer<vtkPoints> mesh_points = poly_data->GetPoints ();
        vtkIdType nr_points = mesh_points->GetNumberOfPoints ();
        //vtkIdType nr_polygons = poly_data->GetNumberOfPolys ();

        mesh->cloud.create(1, nr_points, CV_32FC3);

        Vec3f *mesh_cloud = mesh->cloud.ptr<Vec3f>();
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
            mesh->colors.create(1, nr_points, CV_8UC3);
            Vec3b *mesh_colors = mesh->colors.ptr<cv::Vec3b>();

            for (vtkIdType i = 0; i < mesh_points->GetNumberOfPoints (); i++)
            {
                Vec3b point_color;
                poly_colors->GetTupleValue (i, point_color.val);

                //RGB or BGR? should we swap channels????
                mesh_colors[i] = point_color;
            }
        }
        else
            mesh->colors.release();

        // Now handle the polygons
        vtkIdType* cell_points;
        vtkIdType nr_cell_points;
        vtkCellArray * mesh_polygons = poly_data->GetPolys ();
        mesh_polygons->InitTraversal ();
        
        mesh->polygons.create(1, mesh_polygons->GetSize(), CV_32SC1);
        
        int * polygons = mesh->polygons.ptr<int>();
        while (mesh_polygons->GetNextCell (nr_cell_points, cell_points))
        {
            *polygons++ = nr_cell_points;
            for (int i = 0; i < nr_cell_points; ++i)
                *polygons++ = static_cast<int> (cell_points[i]);
        }

        return mesh;
    }
};

cv::viz::Mesh3d::Ptr cv::viz::Mesh3d::loadMesh(const String& file)
{
    return loadMeshImpl::loadMesh(file);
}
