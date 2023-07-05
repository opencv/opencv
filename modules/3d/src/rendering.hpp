#pragma once
#include "precomp.hpp"

namespace cv {

struct Triangle
{
    Vec3f vertices[3];
    Vec3f color[3];

    void setVertexPosition(int index, Vec3f vertex);
    void setVertexColor(int index, Vec3f color);

    Vec3f getTriangleColor() const { return color[0]; }
};
}
