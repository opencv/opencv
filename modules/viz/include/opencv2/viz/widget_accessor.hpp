#pragma once

#include <opencv2/core/cvdef.h>
#include <vtkSmartPointer.h>
#include <vtkProp.h>

namespace cv
{
    namespace viz
    {
        class Widget;

        //The class is only that depends on VTK in its interface.
        //It is indended for those users who want to develop own widgets system using VTK library API.
        struct CV_EXPORTS WidgetAccessor
        {
            static vtkSmartPointer<vtkProp> getProp(const Widget &widget);
            static void setProp(Widget &widget, vtkSmartPointer<vtkProp> prop);
        };
    }
}
