#include "precomp.hpp"

#include <vtkPLYReader.h>
#include <vtkSmartPointer.h>
#include <vtkPolyData.h>
#include <vtkPointData.h>
#include <vtkCellArray.h>

temp_viz::Mesh3d::Ptr temp_viz::Mesh3d::mesh_load(const String& file)
{
    Mesh3d::Ptr mesh = new Mesh3d();

    vtkSmartPointer<vtkPLYReader> reader = vtkSmartPointer<vtkPLYReader>::New();
    reader->SetFileName(file.c_str());
    reader->Update();
    vtkSmartPointer<vtkPolyData> poly_data = reader->GetOutput ();

    typedef unsigned int uint32_t;
    mesh->polygons.clear();

    vtkSmartPointer<vtkPoints> mesh_points = poly_data->GetPoints ();
    vtkIdType nr_points = mesh_points->GetNumberOfPoints ();
    vtkIdType nr_polygons = poly_data->GetNumberOfPolys ();

    mesh->cloud.create(1, nr_points, CV_32FC3);

    double point_xyz[3];
    for (vtkIdType i = 0; i < mesh_points->GetNumberOfPoints (); i++)
    {
        mesh_points->GetPoint (i, &point_xyz[0]);
        mesh->cloud.ptr<cv::Point3f>()[i] = cv::Point3d(point_xyz[0], point_xyz[1], point_xyz[2]);;
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
        unsigned char point_color[3];

        for (vtkIdType i = 0; i < mesh_points->GetNumberOfPoints (); i++)
        {
            poly_colors->GetTupleValue (i, &point_color[0]);

            //RGB or BGR?????
            mesh->colors.ptr<cv::Vec3b>()[i] = cv::Vec3b(point_color[0], point_color[1], point_color[2]);
        }
    }
    else
        mesh->colors.release();

    // Now handle the polygons
    mesh->polygons.resize (nr_polygons);
    vtkIdType* cell_points;
    vtkIdType nr_cell_points;
    vtkCellArray * mesh_polygons = poly_data->GetPolys ();
    mesh_polygons->InitTraversal ();
    int id_poly = 0;
    while (mesh_polygons->GetNextCell (nr_cell_points, cell_points))
    {
        mesh->polygons[id_poly].vertices.resize (nr_cell_points);
        for (int i = 0; i < nr_cell_points; ++i)
            mesh->polygons[id_poly].vertices[i] = static_cast<int> (cell_points[i]);
        ++id_poly;
    }

    return mesh;
}
