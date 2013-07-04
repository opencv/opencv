#pragma once

#include <opencv2/viz/widgets.hpp>
#include <vtkSmartPointer.h>
#include <vtkLODActor.h>

namespace temp_viz
{
    //The class is only that depends on VTK in its interface.
    //It is indended for those user who want to develop own widgets system using VTK library API.
    struct CV_EXPORTS WidgetAccessor
    {
        static  vtkSmartPointer<vtkLODActor> getActor(const Widget &widget);
    };
}
