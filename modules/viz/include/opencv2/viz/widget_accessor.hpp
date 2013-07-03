#pragma once

#include "precomp.hpp"
#include "types.hpp"

namespace temp_viz
{
    struct WidgetAccessor
    {
        static CV_EXPORTS vtkSmartPointer<vtkActor> getActor(const Widget &widget);
    };
}