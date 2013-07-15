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