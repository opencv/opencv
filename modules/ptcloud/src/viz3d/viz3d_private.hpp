// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_PTCLOUD_VIZ3D_PRIVATE_HPP
#define OPENCV_PTCLOUD_VIZ3D_PRIVATE_HPP

#include "../precomp.hpp"
#include "opencv2/core/private.hpp"  // HAVE_OPENGL from cvconfig.h
#include "opencv2/core/opengl.hpp"
#include "opencv2/highgui.hpp"       // window + OpenGL context (imported from highgui)
#include "opencv2/ptcloud/viz3d.hpp" // public viz3d API declarations

#include <map>

#ifdef HAVE_OPENGL

namespace cv { namespace viz3d {

// Stores a view's matrices
class View
{
public:
    View();

    void setAspect(float aspect);
    void setViewport(Size viewport);
    void setPerspective(float fov, float z_near, float z_far);

    void rotate(float dx, float dy); // Rotates the camera using mouse input
    void move(float dx, float dy);   // Moves the camera using mouse input
    void scaleDistance(float amount);

    inline Vec3f getOrigin() const { return this->origin; }
    inline Vec3f getPosition() const { return this->position; }
    inline float getDistance() const { return this->distance; }
    inline Matx44f getView() const { return this->view; }
    inline Matx44f getProj() const { return this->proj; }
    inline Size getViewport() const { return this->viewport; }

private:
    void lookAt(const Vec3f& point, const Vec3f& up);

    float aspect;
    float fov;
    float z_near;
    float z_far;
    Size viewport;

    Matx44f proj;
    Matx44f view;

    Vec3f origin;
    Vec3f position;
    Vec3f up;
    float distance;
};

// Stores information about a light
struct Light
{
    Vec3f direction;
    Vec3f ambient;
    Vec3f diffuse;
};

// Base class for viz3d objects which can be rendered
class Object
{
public:
    Object();
    virtual ~Object() = default;

    void setPosition(const Vec3f& position);
    void setRotation(const Vec3f& rotation);

    virtual void draw(const View& view, const Light& light) = 0;

    virtual String getShaderName() = 0;
    virtual ogl::Program buildShader() = 0;
    virtual void setShader(ogl::Program program) = 0;

    // Alpha-blended objects must be drawn after all opaque ones. See Window::draw.
    virtual bool isTransparent() const { return false; }

    inline Matx44f getModel() const { return this->model; }

private:
    void updateModel();

    Vec3f position;
    Vec3f rotation;

    Matx44f model;
};

// A viz3d window and its objects.
class Window
{
public:
    Window(const String& name);

    Ptr<Object> get(const String& obj_name);
    void set(const String& obj_name, const Ptr<Object>& obj);

    void setSun(const Vec3f& direction, const Vec3f& ambient, const Vec3f& diffuse);
    void setSky(const Vec3f& color);
    void setGridVisible(bool visible);

    void draw();
    void onMouse(int event, int x, int y, int flags);

    inline View& getView() { return this->view; }

private:
    String name;
    Size size;

    Light sun;
    Vec3f sky_color;

    View view;
    int l_mouse_x;
    int l_mouse_y;

    Ptr<Object> crosshair;
    Ptr<Object> grid;
    std::map<String, Ptr<Object>> objects;
    std::map<String, ogl::Program> shaders;
};

// Mesh object.
class Mesh : public Object
{
public:
    Mesh(InputArray verts, InputArray indices);
    Mesh(InputArray verts);

    virtual void draw(const View& view, const Light& light) override;

    virtual String getShaderName() override;
    virtual ogl::Program buildShader() override;
    virtual void setShader(ogl::Program program) override;

private:
    void initVA(int width);

    ogl::Program program;
    ogl::VertexArray va;
    ogl::Buffer verts;
    ogl::Buffer indices;

    int index_type;

    int model_loc;
    int view_loc;
    int proj_loc;
    int sun_direction_loc;
    int sun_ambient_loc;
    int sun_diffuse_loc;
};

// Lines object.
class Lines : public Object
{
public:
    Lines(InputArray points, int count = -1);

    virtual void draw(const View& view, const Light& light) override;
    void update(InputArray points);

    virtual String getShaderName() override;
    virtual ogl::Program buildShader() override;
    virtual void setShader(ogl::Program program) override;

private:
    ogl::Program program;
    ogl::VertexArray va;
    ogl::Buffer points;

    int model_loc;
    int view_loc;
    int proj_loc;

    int count;
};

// Point-cloud object.
class PointCloud : public Object
{
public:
    PointCloud(InputArray points);

    virtual void draw(const View& view, const Light& light) override;

    virtual String getShaderName() override;
    virtual ogl::Program buildShader() override;
    virtual void setShader(ogl::Program program) override;

private:
    ogl::Program program;
    ogl::VertexArray va;
    ogl::Buffer points;

    int model_loc;
    int view_loc;
    int proj_loc;
};

// 3D Gaussian Splatting object.
class GaussianSplats : public Object
{
public:
    GaussianSplats(InputArray splats);

    virtual void draw(const View& view, const Light& light) override;

    virtual String getShaderName() override;
    virtual ogl::Program buildShader() override;
    virtual void setShader(ogl::Program program) override;

    virtual bool isTransparent() const override { return true; }

private:
    void reorder(const Vec3f& cam);

    ogl::Program program;
    ogl::VertexArray va;
    ogl::Buffer quad;
    ogl::Buffer quad_indices;
    ogl::Buffer data;
    ogl::Buffer order;
    ogl::TextureBuffer data_tex;
    ogl::TextureBuffer order_tex;

    Mat splats;
    std::vector<int> order_cpu;
    Vec3f last_cam;
    Matx44f last_model;
    bool sorted;
    int count;

    int model_loc;
    int view_loc;
    int proj_loc;
    int focal_loc;
    int viewport_loc;
    int data_loc;
    int order_loc;
};

}} // namespace cv::viz3d

#endif // HAVE_OPENGL

#endif
