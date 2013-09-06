#pragma once

#include <opencv2/core/cvdef.h>
#include <opencv2/core.hpp>
#include <opencv2/viz/types.hpp>

namespace cv
{
    namespace viz
    {
        enum RenderingProperties
        {
            VIZ_POINT_SIZE,
            VIZ_OPACITY,
            VIZ_LINE_WIDTH,
            VIZ_FONT_SIZE,
            VIZ_COLOR,
            VIZ_REPRESENTATION,
            VIZ_IMMEDIATE_RENDERING,
            VIZ_SHADING
        };

        enum RenderingRepresentationProperties
        {
            REPRESENTATION_POINTS,
            REPRESENTATION_WIREFRAME,
            REPRESENTATION_SURFACE
        };

        enum ShadingRepresentationProperties
        {
            SHADING_FLAT,
            SHADING_GOURAUD,
            SHADING_PHONG
        };

    }

}
