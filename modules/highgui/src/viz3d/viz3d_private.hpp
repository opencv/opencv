// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef __OPENCV_HIGHGUI_VIZ3D_PRIVATE_HPP__
#define __OPENCV_HIGHGUI_VIZ3D_PRIVATE_HPP__

#include "../precomp.hpp"
#include "opencv2/core/opengl.hpp"

#include <map>

#ifdef HAVE_OPENGL

namespace cv { namespace viz3d {

// Stores a view's matrices
class View
{
public:
    View();

    void setAspect(float aspect);
    void setPerspective(float fov, float z_near, float z_far);

    void rotate(float dx, float dy); // Rotates the camera using mouse input
    void move(float dx, float dy);   // Moves the camera using mouse input
    void scaleDistance(float amount);

    inline Vec3f getOrigin() const { return this->origin; }
    inline Vec3f getPosition() const { return this->position; }
    inline float getDistance() const { return this->distance; }
    inline Matx44f getView() const { return this->view; }
    inline Matx44f getProj() const { return this->proj; }

private:
    void lookAt(const Vec3f& point, const Vec3f& up);

    float aspect;
    float fov;
    float z_near;
    float z_far;

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

    inline Matx44f getModel() const { return this->model; }

private:
    void updateModel();

    Vec3f position;
    Vec3f rotation;

    Matx44f model;
};

// Class which stores the viz3d data associated to a window.
class Window
{
public:
    Window(const String& name);
    ~Window();

    Object* get(const String& obj_name);
    void set(const String& obj_name, Object* obj);

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

    Object* crosshair;
    Object* grid;
    std::map<String, Object*> objects;
    std::map<String, ogl::Program> shaders;
};

// Class which stores the viz3d data associated to a mesh object.
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

// Class which stores the viz3d data associated to a textured mesh object.
class TexturedMesh : public Object
{
public:
    TexturedMesh(InputArray verts, Mat tex, int count = -1);

    virtual void draw(const View& view, const Light& light) override;
    void update(InputArray verts);

    virtual String getShaderName() override;
    virtual ogl::Program buildShader() override;
    virtual void setShader(ogl::Program program) override;

private:
    ogl::Program program;
    ogl::VertexArray va;
    ogl::Buffer verts;
    ogl::Texture2D tex;

    int count;

    int tex_loc;
    int model_loc;
    int view_loc;
    int proj_loc;
};

// Class which stores the viz3d data associated to a lines object.
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

	ogl::IndexType index_type;

	int model_loc;
	int view_loc;
	int proj_loc;

    int count;
};

// Data necessary for drawing a point cloud on a window.
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

}} // namespace cv::viz3d

#endif // HAVE_OPENGL

#endif
