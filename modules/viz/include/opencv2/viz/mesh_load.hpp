#pragma once

#include <opencv2/core.hpp>
#include <opencv2/viz/types.hpp>
#include <vector>

namespace temp_viz
{
    CV_EXPORTS Mesh3d::Ptr mesh_load(const String& file);
}
