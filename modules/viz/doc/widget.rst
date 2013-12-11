Widget
======

.. highlight:: cpp

In this section, the widget framework is explained. Widgets represent
2D or 3D objects, varying from simple ones such as lines to complex one such as
point clouds and meshes.

Widgets are **implicitly shared**. Therefore, one can add a widget to the scene,
and modify the widget without re-adding the widget.

.. code-block:: cpp

    ...
    /// Create a cloud widget
    viz::WCloud cw(cloud, viz::Color::red());
    /// Display it in a window
    myWindow.showWidget("CloudWidget1", cw);
    /// Modify it, and it will be modified in the window.
    cw.setColor(viz::Color::yellow());
    ...

viz::Widget
-----------
.. ocv:class:: Widget

Base class of all widgets. Widget is implicitly shared. ::

    class CV_EXPORTS Widget
    {
    public:
        Widget();
        Widget(const Widget& other);
        Widget& operator=(const Widget& other);
        ~Widget();

        //! Create a widget directly from ply file
        static Widget fromPlyFile(const String &file_name);

        //! Rendering properties of this particular widget
        void setRenderingProperty(int property, double value);
        double getRenderingProperty(int property) const;

        //! Casting between widgets
        template<typename _W> _W cast();
    private:
        /* hidden */
    };

viz::Widget::fromPlyFile
------------------------
Creates a widget from ply file.

.. ocv:function:: static Widget fromPlyFile(const String &file_name)

    :param file_name: Ply file name.

viz::Widget::setRenderingProperty
---------------------------------
Sets rendering property of the widget.

.. ocv:function:: void setRenderingProperty(int property, double value)

    :param property: Property that will be modified.
    :param value: The new value of the property.

    **Rendering property** can be one of the following:

    * **POINT_SIZE**
    * **OPACITY**
    * **LINE_WIDTH**
    * **FONT_SIZE**
    * **REPRESENTATION**: Expected values are
        * **REPRESENTATION_POINTS**
        * **REPRESENTATION_WIREFRAME**
        * **REPRESENTATION_SURFACE**
    * **IMMEDIATE_RENDERING**:
        * Turn on immediate rendering by setting the value to ``1``.
        * Turn off immediate rendering by setting the value to ``0``.
    * **SHADING**: Expected values are
        * **SHADING_FLAT**
        * **SHADING_GOURAUD**
        * **SHADING_PHONG**

viz::Widget::getRenderingProperty
---------------------------------
Returns rendering property of the widget.

.. ocv:function:: double getRenderingProperty(int property) const

    :param property: Property.

    **Rendering property** can be one of the following:

    * **POINT_SIZE**
    * **OPACITY**
    * **LINE_WIDTH**
    * **FONT_SIZE**
    * **REPRESENTATION**: Expected values are
        * **REPRESENTATION_POINTS**
        * **REPRESENTATION_WIREFRAME**
        * **REPRESENTATION_SURFACE**
    * **IMMEDIATE_RENDERING**:
        * Turn on immediate rendering by setting the value to ``1``.
        * Turn off immediate rendering by setting the value to ``0``.
    * **SHADING**: Expected values are
        * **SHADING_FLAT**
        * **SHADING_GOURAUD**
        * **SHADING_PHONG**

viz::Widget::cast
-----------------
Casts a widget to another.

.. ocv:function:: template<typename _W> _W cast()

.. code-block:: cpp

    // Create a sphere widget
    viz::WSphere sw(Point3f(0.0f,0.0f,0.0f), 0.5f);
    // Cast sphere widget to cloud widget
    viz::WCloud cw = sw.cast<viz::WCloud>();

.. note:: 3D Widgets can only be cast to 3D Widgets. 2D Widgets can only be cast to 2D Widgets.

viz::WidgetAccessor
-------------------
.. ocv:class:: WidgetAccessor

This class is for users who want to develop their own widgets using VTK library API. ::

    struct CV_EXPORTS WidgetAccessor
    {
        static vtkSmartPointer<vtkProp> getProp(const Widget &widget);
        static void setProp(Widget &widget, vtkSmartPointer<vtkProp> prop);
    };

viz::WidgetAccessor::getProp
----------------------------
Returns ``vtkProp`` of a given widget.

.. ocv:function:: static vtkSmartPointer<vtkProp> getProp(const Widget &widget)

    :param widget: Widget whose ``vtkProp`` is to be returned.

.. note:: vtkProp has to be down cast appropriately to be modified.

    .. code-block:: cpp

        vtkActor * actor = vtkActor::SafeDownCast(viz::WidgetAccessor::getProp(widget));

viz::WidgetAccessor::setProp
----------------------------
Sets ``vtkProp`` of a given widget.

.. ocv:function:: static void setProp(Widget &widget, vtkSmartPointer<vtkProp> prop)

    :param widget: Widget whose ``vtkProp`` is to be set.
    :param prop: A ``vtkProp``.

viz::Widget3D
-------------
.. ocv:class:: Widget3D

Base class of all 3D widgets. ::

    class CV_EXPORTS Widget3D : public Widget
    {
    public:
        Widget3D() {}

        void setPose(const Affine3f &pose);
        void updatePose(const Affine3f &pose);
        Affine3f getPose() const;

        void setColor(const Color &color);
    private:
        /* hidden */
    };

viz::Widget3D::setPose
----------------------
Sets pose of the widget.

.. ocv:function:: void setPose(const Affine3f &pose)

    :param pose: The new pose of the widget.

viz::Widget3D::updateWidgetPose
-------------------------------
Updates pose of the widget by pre-multiplying its current pose.

.. ocv:function:: void updateWidgetPose(const Affine3f &pose)

    :param pose: The pose that the current pose of the widget will be pre-multiplied by.

viz::Widget3D::getPose
----------------------
Returns the current pose of the widget.

.. ocv:function:: Affine3f getWidgetPose() const

viz::Widget3D::setColor
-----------------------
Sets the color of the widget.

.. ocv:function:: void setColor(const Color &color)

    :param color: color of type :ocv:class:`Color`

viz::Widget2D
-------------
.. ocv:class:: Widget2D

Base class of all 2D widgets. ::

    class CV_EXPORTS Widget2D : public Widget
    {
    public:
        Widget2D() {}

        void setColor(const Color &color);
    };

viz::Widget2D::setColor
-----------------------
Sets the color of the widget.

.. ocv:function:: void setColor(const Color &color)

    :param color: color of type :ocv:class:`Color`

viz::WLine
----------
.. ocv:class:: WLine

This 3D Widget defines a finite line. ::

    class CV_EXPORTS WLine : public Widget3D
    {
    public:
        WLine(const Point3f &pt1, const Point3f &pt2, const Color &color = Color::white());
    };

viz::WLine::WLine
-----------------
Constructs a WLine.

.. ocv:function:: WLine(const Point3f &pt1, const Point3f &pt2, const Color &color = Color::white())

    :param pt1: Start point of the line.
    :param pt2: End point of the line.
    :param color: :ocv:class:`Color` of the line.

viz::WPlane
-----------
.. ocv:class:: WPlane

This 3D Widget defines a finite plane. ::

    class CV_EXPORTS WPlane : public Widget3D
    {
    public:
        WPlane(const Vec4f& coefs, float size = 1.0, const Color &color = Color::white());
        WPlane(const Vec4f& coefs, const Point3f& pt, float size = 1.0, const Color &color = Color::white());
    private:
        /* hidden */
    };

viz::WPlane::WPlane
-------------------
Constructs a WPlane.

.. ocv:function:: WPlane(const Vec4f& coefs, float size = 1.0, const Color &color = Color::white())

    :param coefs: Plane coefficients as in (A,B,C,D) where Ax + By + Cz + D = 0.
    :param size: Size of the plane.
    :param color: :ocv:class:`Color` of the plane.

.. ocv:function:: WPlane(const Vec4f& coefs, const Point3f& pt, float size = 1.0, const Color &color = Color::white())

    :param coefs: Plane coefficients as in (A,B,C,D) where Ax + By + Cz + D = 0.
    :param pt: Position of the plane.
    :param size: Size of the plane.
    :param color: :ocv:class:`Color` of the plane.

viz::WSphere
------------
.. ocv:class:: WSphere

This 3D Widget defines a sphere. ::

    class CV_EXPORTS WSphere : public Widget3D
    {
    public:
        WSphere(const cv::Point3f &center, float radius, int sphere_resolution = 10, const Color &color = Color::white())
    };

viz::WSphere::WSphere
---------------------
Constructs a WSphere.

.. ocv:function:: WSphere(const cv::Point3f &center, float radius, int sphere_resolution = 10, const Color &color = Color::white())

    :param center: Center of the sphere.
    :param radius: Radius of the sphere.
    :param sphere_resolution: Resolution of the sphere.
    :param color: :ocv:class:`Color` of the sphere.

viz::WArrow
----------------
.. ocv:class:: WArrow

This 3D Widget defines an arrow. ::

    class CV_EXPORTS WArrow : public Widget3D
    {
    public:
        WArrow(const Point3f& pt1, const Point3f& pt2, float thickness = 0.03, const Color &color = Color::white());
    };

viz::WArrow::WArrow
-----------------------------
Constructs an WArrow.

.. ocv:function:: WArrow(const Point3f& pt1, const Point3f& pt2, float thickness = 0.03, const Color &color = Color::white())

    :param pt1: Start point of the arrow.
    :param pt2: End point of the arrow.
    :param thickness: Thickness of the arrow. Thickness of arrow head is also adjusted accordingly.
    :param color: :ocv:class:`Color` of the arrow.

Arrow head is located at the end point of the arrow.

viz::WCircle
-----------------
.. ocv:class:: WCircle

This 3D Widget defines a circle. ::

    class CV_EXPORTS WCircle : public Widget3D
    {
    public:
        WCircle(const Point3f& pt, float radius, float thickness = 0.01, const Color &color = Color::white());
    };

viz::WCircle::WCircle
-------------------------------
Constructs a WCircle.

.. ocv:function:: WCircle(const Point3f& pt, float radius, float thickness = 0.01, const Color &color = Color::white())

    :param pt: Center of the circle.
    :param radius: Radius of the circle.
    :param thickness: Thickness of the circle.
    :param color: :ocv:class:`Color` of the circle.

viz::WCylinder
--------------
.. ocv:class:: WCylinder

This 3D Widget defines a cylinder. ::

    class CV_EXPORTS WCylinder : public Widget3D
    {
    public:
        WCylinder(const Point3f& pt_on_axis, const Point3f& axis_direction, float radius, int numsides = 30, const Color &color = Color::white());
    };

viz::WCylinder::WCylinder
-----------------------------------
Constructs a WCylinder.

.. ocv:function:: WCylinder(const Point3f& pt_on_axis, const Point3f& axis_direction, float radius, int numsides = 30, const Color &color = Color::white())

    :param pt_on_axis: A point on the axis of the cylinder.
    :param axis_direction: Direction of the axis of the cylinder.
    :param radius: Radius of the cylinder.
    :param numsides: Resolution of the cylinder.
    :param color: :ocv:class:`Color` of the cylinder.

viz::WCube
----------
.. ocv:class:: WCube

This 3D Widget defines a cube. ::

    class CV_EXPORTS WCube : public Widget3D
    {
    public:
        WCube(const Point3f& pt_min, const Point3f& pt_max, bool wire_frame = true, const Color &color = Color::white());
    };

viz::WCube::WCube
---------------------------
Constructs a WCube.

.. ocv:function:: WCube(const Point3f& pt_min, const Point3f& pt_max, bool wire_frame = true, const Color &color = Color::white())

    :param pt_min: Specifies minimum point of the bounding box.
    :param pt_max: Specifies maximum point of the bounding box.
    :param wire_frame: If true, cube is represented as wireframe.
    :param color: :ocv:class:`Color` of the cube.

.. image:: images/cube_widget.png
    :alt: Cube Widget
    :align: center

viz::WCoordinateSystem
----------------------
.. ocv:class:: WCoordinateSystem

This 3D Widget represents a coordinate system. ::

    class CV_EXPORTS WCoordinateSystem : public Widget3D
    {
    public:
        WCoordinateSystem(float scale = 1.0);
    };

viz::WCoordinateSystem::WCoordinateSystem
---------------------------------------------------
Constructs a WCoordinateSystem.

.. ocv:function:: WCoordinateSystem(float scale = 1.0)

    :param scale: Determines the size of the axes.

viz::WPolyLine
--------------
.. ocv:class:: WPolyLine

This 3D Widget defines a poly line. ::

    class CV_EXPORTS WPolyLine : public Widget3D
    {
    public:
        WPolyLine(InputArray points, const Color &color = Color::white());

    private:
        /* hidden */
    };

viz::WPolyLine::WPolyLine
-----------------------------------
Constructs a WPolyLine.

.. ocv:function:: WPolyLine(InputArray points, const Color &color = Color::white())

    :param points: Point set.
    :param color: :ocv:class:`Color` of the poly line.

viz::WGrid
----------
.. ocv:class:: WGrid

This 3D Widget defines a grid. ::

    class CV_EXPORTS WGrid : public Widget3D
    {
    public:
        //! Creates grid at the origin
        WGrid(const Vec2i &dimensions, const Vec2d &spacing, const Color &color = Color::white());
        //! Creates grid based on the plane equation
        WGrid(const Vec4f &coeffs, const Vec2i &dimensions, const Vec2d &spacing, const Color &color = Color::white());
    private:
        /* hidden */
    };

viz::WGrid::WGrid
---------------------------
Constructs a WGrid.

.. ocv:function:: WGrid(const Vec2i &dimensions, const Vec2d &spacing, const Color &color = Color::white())

    :param dimensions: Number of columns and rows, respectively.
    :param spacing: Size of each column and row, respectively.
    :param color: :ocv:class:`Color` of the grid.

.. ocv:function:  WGrid(const Vec4f &coeffs, const Vec2i &dimensions, const Vec2d &spacing, const Color &color = Color::white())

    :param coeffs: Plane coefficients as in (A,B,C,D) where Ax + By + Cz + D = 0.
    :param dimensions: Number of columns and rows, respectively.
    :param spacing: Size of each column and row, respectively.
    :param color: :ocv:class:`Color` of the grid.

viz::WText3D
------------
.. ocv:class:: WText3D

This 3D Widget represents 3D text. The text always faces the camera. ::

    class CV_EXPORTS WText3D : public Widget3D
    {
    public:
        WText3D(const String &text, const Point3f &position, float text_scale = 1.0, bool face_camera = true, const Color &color = Color::white());

        void setText(const String &text);
        String getText() const;
    };

viz::WText3D::WText3D
-------------------------------
Constructs a WText3D.

.. ocv:function:: WText3D(const String &text, const Point3f &position, float text_scale = 1.0, bool face_camera = true, const Color &color = Color::white())

    :param text: Text content of the widget.
    :param position: Position of the text.
    :param text_scale: Size of the text.
    :param face_camera: If true, text always faces the camera.
    :param color: :ocv:class:`Color` of the text.

viz::WText3D::setText
---------------------
Sets the text content of the widget.

.. ocv:function:: void setText(const String &text)

    :param text: Text content of the widget.

viz::WText3D::getText
---------------------
Returns the current text content of the widget.

.. ocv:function:: String getText() const

viz::WText
----------
.. ocv:class:: WText

This 2D Widget represents text overlay. ::

    class CV_EXPORTS WText : public Widget2D
    {
    public:
        WText(const String &text, const Point2i &pos, int font_size = 10, const Color &color = Color::white());

        void setText(const String &text);
        String getText() const;
    };

viz::WText::WText
-----------------
Constructs a WText.

.. ocv:function:: WText(const String &text, const Point2i &pos, int font_size = 10, const Color &color = Color::white())

    :param text: Text content of the widget.
    :param pos: Position of the text.
    :param font_size: Font size.
    :param color: :ocv:class:`Color` of the text.

viz::WText::setText
-------------------
Sets the text content of the widget.

.. ocv:function:: void setText(const String &text)

    :param text: Text content of the widget.

viz::WText::getText
-------------------
Returns the current text content of the widget.

.. ocv:function:: String getText() const

viz::WImageOverlay
------------------
.. ocv:class:: WImageOverlay

This 2D Widget represents an image overlay. ::

    class CV_EXPORTS WImageOverlay : public Widget2D
    {
    public:
        WImageOverlay(const Mat &image, const Rect &rect);

        void setImage(const Mat &image);
    };

viz::WImageOverlay::WImageOverlay
---------------------------------
Constructs an WImageOverlay.

.. ocv:function:: WImageOverlay(const Mat &image, const Rect &rect)

    :param image: BGR or Gray-Scale image.
    :param rect: Image is scaled and positioned based on rect.

viz::WImageOverlay::setImage
----------------------------
Sets the image content of the widget.

.. ocv:function:: void setImage(const Mat &image)

    :param image: BGR or Gray-Scale image.

viz::WImage3D
-------------
.. ocv:class:: WImage3D

This 3D Widget represents an image in 3D space. ::

    class CV_EXPORTS WImage3D : public Widget3D
    {
    public:
        //! Creates 3D image at the origin
        WImage3D(const Mat &image, const Size &size);
        //! Creates 3D image at a given position, pointing in the direction of the normal, and having the up_vector orientation
        WImage3D(const Vec3f &position, const Vec3f &normal, const Vec3f &up_vector, const Mat &image, const Size &size);

        void setImage(const Mat &image);
    };

viz::WImage3D::WImage3D
-----------------------
Constructs an WImage3D.

.. ocv:function:: WImage3D(const Mat &image, const Size &size)

    :param image: BGR or Gray-Scale image.
    :param size: Size of the image.

.. ocv:function:: WImage3D(const Vec3f &position, const Vec3f &normal, const Vec3f &up_vector, const Mat &image, const Size &size)

    :param position: Position of the image.
    :param normal: Normal of the plane that represents the image.
    :param up_vector: Determines orientation of the image.
    :param image: BGR or Gray-Scale image.
    :param size: Size of the image.

viz::WImage3D::setImage
-----------------------
Sets the image content of the widget.

.. ocv:function:: void setImage(const Mat &image)

    :param image: BGR or Gray-Scale image.

viz::WCameraPosition
--------------------
.. ocv:class:: WCameraPosition

This 3D Widget represents camera position in a scene by its axes or viewing frustum. ::

    class CV_EXPORTS WCameraPosition : public Widget3D
    {
    public:
        //! Creates camera coordinate frame (axes) at the origin
        WCameraPosition(float scale = 1.0);
        //! Creates frustum based on the intrinsic marix K at the origin
        WCameraPosition(const Matx33f &K, float scale = 1.0, const Color &color = Color::white());
        //! Creates frustum based on the field of view at the origin
        WCameraPosition(const Vec2f &fov, float scale = 1.0, const Color &color = Color::white());
        //! Creates frustum and display given image at the far plane
        WCameraPosition(const Matx33f &K, const Mat &img, float scale = 1.0, const Color &color = Color::white());
        //! Creates frustum and display given image at the far plane
        WCameraPosition(const Vec2f &fov, const Mat &img, float scale = 1.0, const Color &color = Color::white());
    };

viz::WCameraPosition::WCameraPosition
-------------------------------------
Constructs a WCameraPosition.

- **Display camera coordinate frame.**

    .. ocv:function:: WCameraPosition(float scale = 1.0)

        Creates camera coordinate frame at the origin.

    .. image:: images/cpw1.png
        :alt: Camera coordinate frame
        :align: center

- **Display the viewing frustum.**

    .. ocv:function:: WCameraPosition(const Matx33f &K, float scale = 1.0, const Color &color = Color::white())

        :param K: Intrinsic matrix of the camera.
        :param scale: Scale of the frustum.
        :param color: :ocv:class:`Color` of the frustum.

        Creates viewing frustum of the camera based on its intrinsic matrix K.

    .. ocv:function:: WCameraPosition(const Vec2f &fov, float scale = 1.0, const Color &color = Color::white())

        :param fov: Field of view of the camera (horizontal, vertical).
        :param scale: Scale of the frustum.
        :param color: :ocv:class:`Color` of the frustum.

        Creates viewing frustum of the camera based on its field of view fov.

    .. image:: images/cpw2.png
        :alt: Camera viewing frustum
        :align: center

- **Display image on the far plane of the viewing frustum.**

    .. ocv:function:: WCameraPosition(const Matx33f &K, const Mat &img, float scale = 1.0, const Color &color = Color::white())

        :param K: Intrinsic matrix of the camera.
        :param img: BGR or Gray-Scale image that is going to be displayed on the far plane of the frustum.
        :param scale: Scale of the frustum and image.
        :param color: :ocv:class:`Color` of the frustum.

        Creates viewing frustum of the camera based on its intrinsic matrix K, and displays image on the far end plane.

    .. ocv:function:: WCameraPosition(const Vec2f &fov, const Mat &img, float scale = 1.0, const Color &color = Color::white())

        :param fov: Field of view of the camera (horizontal, vertical).
        :param img: BGR or Gray-Scale image that is going to be displayed on the far plane of the frustum.
        :param scale: Scale of the frustum and image.
        :param color: :ocv:class:`Color` of the frustum.

        Creates viewing frustum of the camera based on its intrinsic matrix K, and displays image on the far end plane.

    .. image:: images/cpw3.png
        :alt: Camera viewing frustum with image
        :align: center

viz::WTrajectory
----------------
.. ocv:class:: WTrajectory

This 3D Widget represents a trajectory. ::

    class CV_EXPORTS WTrajectory : public Widget3D
    {
    public:
        enum {DISPLAY_FRAMES = 1, DISPLAY_PATH = 2};

        //! Displays trajectory of the given path either by coordinate frames or polyline
        WTrajectory(const std::vector<Affine3f> &path, int display_mode = WTrajectory::DISPLAY_PATH, const Color &color = Color::white(), float scale = 1.0);
        //! Displays trajectory of the given path by frustums
        WTrajectory(const std::vector<Affine3f> &path, const Matx33f &K, float scale = 1.0, const Color &color = Color::white());
        //! Displays trajectory of the given path by frustums
        WTrajectory(const std::vector<Affine3f> &path, const Vec2f &fov, float scale = 1.0, const Color &color = Color::white());

    private:
        /* hidden */
    };

viz::WTrajectory::WTrajectory
-----------------------------
Constructs a WTrajectory.

.. ocv:function:: WTrajectory(const std::vector<Affine3f> &path, int display_mode = WTrajectory::DISPLAY_PATH, const Color &color = Color::white(), float scale = 1.0)

    :param path: List of poses on a trajectory.
    :param display_mode: Display mode. This can be DISPLAY_PATH, DISPLAY_FRAMES, DISPLAY_PATH & DISPLAY_FRAMES.
    :param color: :ocv:class:`Color` of the polyline that represents path. Frames are not affected.
    :param scale: Scale of the frames. Polyline is not affected.

    Displays trajectory of the given path as follows:

    * DISPLAY_PATH : Displays a poly line that represents the path.
    * DISPLAY_FRAMES : Displays coordinate frames at each pose.
    * DISPLAY_PATH & DISPLAY_FRAMES : Displays both poly line and coordinate frames.

.. ocv:function:: WTrajectory(const std::vector<Affine3f> &path, const Matx33f &K, float scale = 1.0, const Color &color = Color::white())

    :param path: List of poses on a trajectory.
    :param K: Intrinsic matrix of the camera.
    :param scale: Scale of the frustums.
    :param color: :ocv:class:`Color` of the frustums.

    Displays frustums at each pose of the trajectory.

.. ocv:function:: WTrajectory(const std::vector<Affine3f> &path, const Vec2f &fov, float scale = 1.0, const Color &color = Color::white())

    :param path: List of poses on a trajectory.
    :param fov: Field of view of the camera (horizontal, vertical).
    :param scale: Scale of the frustums.
    :param color: :ocv:class:`Color` of the frustums.

    Displays frustums at each pose of the trajectory.

viz::WSpheresTrajectory
-----------------------
.. ocv:class:: WSpheresTrajectory

This 3D Widget represents a trajectory using spheres and lines, where spheres represent the positions of the camera, and lines
represent the direction from previous position to the current. ::

    class CV_EXPORTS WSpheresTrajectory : public Widget3D
    {
    public:
        WSpheresTrajectory(const std::vector<Affine3f> &path, float line_length = 0.05f,
                    float init_sphere_radius = 0.021, sphere_radius = 0.007,
                    Color &line_color = Color::white(), const Color &sphere_color = Color::white());
    };

viz::WSpheresTrajectory::WSpheresTrajectory
-------------------------------------------
Constructs a WSpheresTrajectory.

.. ocv:function:: WSpheresTrajectory(const std::vector<Affine3f> &path, float line_length = 0.05f, float init_sphere_radius = 0.021, float sphere_radius = 0.007, const Color &line_color = Color::white(), const Color &sphere_color = Color::white())

    :param path: List of poses on a trajectory.
    :param line_length: Length of the lines.
    :param init_sphere_radius: Radius of the first sphere which represents the initial position of the camera.
    :param sphere_radius: Radius of the rest of the spheres.
    :param line_color: :ocv:class:`Color` of the lines.
    :param sphere_color: :ocv:class:`Color` of the spheres.

viz::WCloud
-----------
.. ocv:class:: WCloud

This 3D Widget defines a point cloud. ::

    class CV_EXPORTS WCloud : public Widget3D
    {
    public:
        //! Each point in cloud is mapped to a color in colors
        WCloud(InputArray cloud, InputArray colors);
        //! All points in cloud have the same color
        WCloud(InputArray cloud, const Color &color = Color::white());

    private:
        /* hidden */
    };

viz::WCloud::WCloud
-------------------
Constructs a WCloud.

.. ocv:function:: WCloud(InputArray cloud, InputArray colors)

    :param cloud: Set of points which can be of type: ``CV_32FC3``, ``CV_32FC4``, ``CV_64FC3``, ``CV_64FC4``.
    :param colors: Set of colors. It has to be of the same size with cloud.

    Points in the cloud belong to mask when they are set to (NaN, NaN, NaN).

.. ocv:function:: WCloud(InputArray cloud, const Color &color = Color::white())

    :param cloud: Set of points which can be of type: ``CV_32FC3``, ``CV_32FC4``, ``CV_64FC3``, ``CV_64FC4``.
    :param color: A single :ocv:class:`Color` for the whole cloud.

    Points in the cloud belong to mask when they are set to (NaN, NaN, NaN).

.. note:: In case there are four channels in the cloud, fourth channel is ignored.

viz::WCloudCollection
---------------------
.. ocv:class:: WCloudCollection

This 3D Widget defines a collection of clouds. ::

    class CV_EXPORTS WCloudCollection : public Widget3D
    {
    public:
        WCloudCollection();

        //! Each point in cloud is mapped to a color in colors
        void addCloud(InputArray cloud, InputArray colors, const Affine3f &pose = Affine3f::Identity());
        //! All points in cloud have the same color
        void addCloud(InputArray cloud, const Color &color = Color::white(), Affine3f &pose = Affine3f::Identity());

    private:
        /* hidden */
    };

viz::WCloudCollection::WCloudCollection
---------------------------------------
Constructs a WCloudCollection.

.. ocv:function:: WCloudCollection()

viz::WCloudCollection::addCloud
-------------------------------
Adds a cloud to the collection.

.. ocv:function:: void addCloud(InputArray cloud, InputArray colors, const Affine3f &pose = Affine3f::Identity())

    :param cloud: Point set which can be of type: ``CV_32FC3``, ``CV_32FC4``, ``CV_64FC3``, ``CV_64FC4``.
    :param colors: Set of colors. It has to be of the same size with cloud.
    :param pose: Pose of the cloud.

    Points in the cloud belong to mask when they are set to (NaN, NaN, NaN).

.. ocv:function:: void addCloud(InputArray cloud, const Color &color = Color::white(), const Affine3f &pose = Affine3f::Identity())

    :param cloud: Point set which can be of type: ``CV_32FC3``, ``CV_32FC4``, ``CV_64FC3``, ``CV_64FC4``.
    :param colors: A single :ocv:class:`Color` for the whole cloud.
    :param pose: Pose of the cloud.

    Points in the cloud belong to mask when they are set to (NaN, NaN, NaN).

.. note:: In case there are four channels in the cloud, fourth channel is ignored.

viz::WCloudNormals
------------------
.. ocv:class:: WCloudNormals

This 3D Widget represents normals of a point cloud. ::

    class CV_EXPORTS WCloudNormals : public Widget3D
    {
    public:
        WCloudNormals(InputArray cloud, InputArray normals, int level = 100, float scale = 0.02f, const Color &color = Color::white());

    private:
        /* hidden */
    };

viz::WCloudNormals::WCloudNormals
---------------------------------
Constructs a WCloudNormals.

.. ocv:function:: WCloudNormals(InputArray cloud, InputArray normals, int level = 100, float scale = 0.02f, const Color &color = Color::white())

    :param cloud: Point set which can be of type: ``CV_32FC3``, ``CV_32FC4``, ``CV_64FC3``, ``CV_64FC4``.
    :param normals: A set of normals that has to be of same type with cloud.
    :param level: Display only every ``level`` th normal.
    :param scale: Scale of the arrows that represent normals.
    :param color: :ocv:class:`Color` of the arrows that represent normals.

.. note:: In case there are four channels in the cloud, fourth channel is ignored.

viz::WMesh
----------
.. ocv:class:: WMesh

This 3D Widget defines a mesh. ::

    class CV_EXPORTS WMesh : public Widget3D
    {
    public:
        WMesh(const Mesh3d &mesh);

    private:
        /* hidden */
    };

viz::WMesh::WMesh
-----------------
Constructs a WMesh.

.. ocv:function:: WMesh(const Mesh3d &mesh)

    :param mesh: :ocv:class:`Mesh3d` object that will be displayed.
