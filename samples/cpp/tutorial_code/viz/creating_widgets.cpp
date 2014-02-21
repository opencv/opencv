/**
 * @file creating_widgets.cpp
 * @brief Creating custom widgets using VTK
 * @author Ozan Cagri Tonkal
 */

#include <opencv2/viz/vizcore.hpp>
#include <opencv2/viz/widget_accessor.hpp>
#include <iostream>

#include <vtkPoints.h>
#include <vtkTriangle.h>
#include <vtkCellArray.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkIdList.h>
#include <vtkActor.h>
#include <vtkProp.h>

using namespace cv;
using namespace std;

/**
 * @function help
 * @brief Display instructions to use this tutorial program
 */
void help()
{
    cout
    << "--------------------------------------------------------------------------"   << endl
    << "This program shows how to create a custom widget. You can create your own "
    << "widgets by extending Widget2D/Widget3D, and with the help of WidgetAccessor." << endl
    << "Usage:"                                                                       << endl
    << "./creating_widgets"                                                           << endl
    << endl;
}

/**
 * @class TriangleWidget
 * @brief Defining our own 3D Triangle widget
 */
class WTriangle : public viz::Widget3D
{
    public:
        WTriangle(const Point3f &pt1, const Point3f &pt2, const Point3f &pt3, const viz::Color & color = viz::Color::white());
};

/**
 * @function TriangleWidget::TriangleWidget
 * @brief Constructor
 */
WTriangle::WTriangle(const Point3f &pt1, const Point3f &pt2, const Point3f &pt3, const viz::Color & color)
{
    // Create a triangle
    vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
    points->InsertNextPoint(pt1.x, pt1.y, pt1.z);
    points->InsertNextPoint(pt2.x, pt2.y, pt2.z);
    points->InsertNextPoint(pt3.x, pt3.y, pt3.z);

    vtkSmartPointer<vtkTriangle> triangle = vtkSmartPointer<vtkTriangle>::New();
    triangle->GetPointIds()->SetId(0,0);
    triangle->GetPointIds()->SetId(1,1);
    triangle->GetPointIds()->SetId(2,2);

    vtkSmartPointer<vtkCellArray> cells = vtkSmartPointer<vtkCellArray>::New();
    cells->InsertNextCell(triangle);

    // Create a polydata object
    vtkSmartPointer<vtkPolyData> polyData = vtkSmartPointer<vtkPolyData>::New();

    // Add the geometry and topology to the polydata
    polyData->SetPoints(points);
    polyData->SetPolys(cells);

    // Create mapper and actor
    vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
#if VTK_MAJOR_VERSION <= 5
    mapper->SetInput(polyData);
#else
    mapper->SetInputData(polyData);
#endif

    vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);

    // Store this actor in the widget in order that visualizer can access it
    viz::WidgetAccessor::setProp(*this, actor);

    // Set the color of the widget. This has to be called after WidgetAccessor.
    setColor(color);
}

/**
 * @function main
 */
int main()
{
    help();

    /// Create a window
    viz::Viz3d myWindow("Creating Widgets");

    /// Create a triangle widget
    WTriangle tw(Point3f(0.0,0.0,0.0), Point3f(1.0,1.0,1.0), Point3f(0.0,1.0,0.0), viz::Color::red());

    /// Show widget in the visualizer window
    myWindow.showWidget("TRIANGLE", tw);

    /// Start event loop
    myWindow.spin();

    return 0;
}
