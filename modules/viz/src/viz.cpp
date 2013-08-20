#include "precomp.hpp"


cv::Affine3f cv::viz::makeTransformToGlobal(const Vec3f& axis_x, const Vec3f& axis_y, const Vec3f& axis_z, const Vec3f& origin)
{
    Affine3f::Mat3 R;
    R.val[0] = axis_x.val[0];
    R.val[3] = axis_x.val[1];
    R.val[6] = axis_x.val[2];

    R.val[1] = axis_y.val[0];
    R.val[4] = axis_y.val[1];
    R.val[7] = axis_y.val[2];

    R.val[2] = axis_z.val[0];
    R.val[5] = axis_z.val[1];
    R.val[8] = axis_z.val[2];

    return Affine3f(R, origin);
}

cv::Affine3f cv::viz::makeCameraPose(const Vec3f& position, const Vec3f& focal_point, const Vec3f& y_dir)
{
    // Compute the transformation matrix for drawing the camera frame in a scene
    Vec3f n = normalize(focal_point - position);
    Vec3f u = normalize(y_dir.cross(n));
    Vec3f v = n.cross(u);
    
    Matx44f pose_mat = Matx44f::zeros();
    pose_mat(0,0) = u[0];
    pose_mat(0,1) = u[1];
    pose_mat(0,2) = u[2];
    pose_mat(1,0) = v[0];
    pose_mat(1,1) = v[1];
    pose_mat(1,2) = v[2];
    pose_mat(2,0) = n[0];
    pose_mat(2,1) = n[1];
    pose_mat(2,2) = n[2];
    pose_mat(3,0) = position[0];
    pose_mat(3,1) = position[1];
    pose_mat(3,2) = position[2];
    pose_mat(3,3) = 1.0f;
    pose_mat = pose_mat.t();
    return pose_mat;
}

vtkSmartPointer<vtkMatrix4x4> cv::viz::convertToVtkMatrix (const cv::Matx44f &m)
{
    vtkSmartPointer<vtkMatrix4x4> vtk_matrix = vtkSmartPointer<vtkMatrix4x4>::New();
    for (int i = 0; i < 4; i++)
        for (int k = 0; k < 4; k++)
            vtk_matrix->SetElement(i, k, m(i, k));
    return vtk_matrix;
}

cv::Matx44f cv::viz::convertToMatx(const vtkSmartPointer<vtkMatrix4x4>& vtk_matrix)
{
    cv::Matx44f m;
    for (int i = 0; i < 4; i++)
        for (int k = 0; k < 4; k++)
            m(i, k) = vtk_matrix->GetElement (i, k);
    return m;
}

namespace cv
{
    namespace viz
    {
        template<typename _Tp> Vec<_Tp, 3>* vtkpoints_data(vtkSmartPointer<vtkPoints>& points);

        template<> Vec3f* vtkpoints_data<float>(vtkSmartPointer<vtkPoints>& points)
        {
            CV_Assert(points->GetDataType() == VTK_FLOAT);
            vtkDataArray *data = points->GetData();
            float *pointer = static_cast<vtkFloatArray*>(data)->GetPointer(0);
            return reinterpret_cast<Vec3f*>(pointer);
        }

        template<> Vec3d* vtkpoints_data<double>(vtkSmartPointer<vtkPoints>& points)
        {
            CV_Assert(points->GetDataType() == VTK_DOUBLE);
            vtkDataArray *data = points->GetData();
            double *pointer = static_cast<vtkDoubleArray*>(data)->GetPointer(0);
            return reinterpret_cast<Vec3d*>(pointer);
        }
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////
/// Viz accessor implementation

cv::viz::VizAccessor * cv::viz::VizAccessor::instance_ = 0;
bool cv::viz::VizAccessor::is_instantiated_ = false;
cv::viz::VizMap cv::viz::VizAccessor::viz_map_;

cv::viz::VizAccessor::VizAccessor() {}

cv::viz::VizAccessor::~VizAccessor()
{
    is_instantiated_ = false;
}

cv::viz::VizAccessor * cv::viz::VizAccessor::getInstance()
{
    if (is_instantiated_)
    {
        instance_ = new VizAccessor();
        is_instantiated_ = true;
    }
    return instance_;
}

cv::viz::Viz3d cv::viz::VizAccessor::get(const String & window_name)
{
    // Add the prefix Viz
    String name("Viz");
    name = window_name.empty() ? name : name + " - " + window_name;
    
    VizMap::iterator vm_itr = viz_map_.find(name);
    bool exists = vm_itr != viz_map_.end();
    if (exists) return vm_itr->second;
    else return viz_map_.insert(VizPair(window_name, Viz3d(window_name))).first->second;
}

void cv::viz::VizAccessor::add(Viz3d window)
{
    String window_name = window.getWindowName();
    VizMap::iterator vm_itr = viz_map_.find(window_name);
    bool exists = vm_itr != viz_map_.end();
    if (exists) return ;
    viz_map_.insert(std::pair<String,Viz3d>(window_name, window));
}

void cv::viz::VizAccessor::remove(const String &window_name)
{
    VizMap::iterator vm_itr = viz_map_.find(window_name);
    bool exists = vm_itr != viz_map_.end();
    if (!exists) return ;
    viz_map_.erase(vm_itr);
}

cv::viz::Viz3d cv::viz::get(const String &window_name)
{
    return cv::viz::VizAccessor::getInstance()->get(window_name);
}