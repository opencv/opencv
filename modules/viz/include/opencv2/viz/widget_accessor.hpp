#pragma once

#include <opencv2/viz/types.hpp>

namespace temp_viz
{
    struct WidgetAccessor
    {
        static CV_EXPORTS vtkSmartPointer<vtkLODActor> getActor(const Widget &widget);
    };
}