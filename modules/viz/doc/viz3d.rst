Viz
===

.. highlight:: cpp

This section describes 3D visualization window as well as classes and methods
that are used to interact with it.

3D visualization window (see :ocv:class:`Viz3d`) is used to display widgets (see :ocv:class:`Widget`), and it provides
several methods to interact with scene and widgets.

viz::makeTransformToGlobal
--------------------------
Takes coordinate frame data and builds transform to global coordinate frame.

.. ocv:function:: Affine3f viz::makeTransformToGlobal(const Vec3f& axis_x, const Vec3f& axis_y, const Vec3f& axis_z, const Vec3f& origin = Vec3f::all(0))

    :param axis_x: X axis vector in global coordinate frame.
    :param axis_y: Y axis vector in global coordinate frame.
    :param axis_z: Z axis vector in global coordinate frame.
    :param origin: Origin of the coordinate frame in global coordinate frame.

This function returns affine transform that describes transformation between global coordinate frame and a given coordinate frame.

viz::makeCameraPose
-------------------
Constructs camera pose from position, focal_point and up_vector (see gluLookAt() for more infromation).

.. ocv:function:: Affine3f makeCameraPose(const Vec3f& position, const Vec3f& focal_point, const Vec3f& y_dir)

    :param position: Position of the camera in global coordinate frame.
    :param focal_point: Focal point of the camera in global coordinate frame.
    :param y_dir: Up vector of the camera in global coordinate frame.

This function returns pose of the camera in global coordinate frame.

viz::get
--------
Retrieves a window by its name.

.. ocv:function:: Viz3d get(const String &window_name)

    :param window_name: Name of the window that is to be retrieved.

This function returns a :ocv:class:`Viz3d` object with the given name.

.. note:: If the window with that name already exists, that window is returned. Otherwise, new window is created with the given name, and it is returned.

.. note:: Window names are automatically prefixed by "Viz - " if it is not done by the user.

          .. code-block:: cpp

                /// window and window_2 are the same windows.
                viz::Viz3d window   = viz::get("myWindow");
                viz::Viz3d window_2 = viz::get("Viz - myWindow");

viz::isNan
----------
Checks **float/double** value for nan.

    .. ocv:function:: bool isNan(float x)

    .. ocv:function:: bool isNan(double x)

        :param x: return true if nan.

Checks **vector** for nan.

    .. ocv:function:: bool isNan(const Vec<_Tp, cn>& v)

        :param v: return true if **any** of the elements of the vector is *nan*.

Checks **point** for nan

    .. ocv:function:: bool isNan(const Point3_<_Tp>& p)

        :param p: return true if **any** of the elements of the point is *nan*.

viz::Viz3d
----------
.. ocv:class:: Viz3d

The Viz3d class represents a 3D visualizer window. This class is implicitly shared.    ::

    class CV_EXPORTS Viz3d
    {
    public:
        typedef cv::Ptr<Viz3d> Ptr;
        typedef void (*KeyboardCallback)(const KeyboardEvent&, void*);
        typedef void (*MouseCallback)(const MouseEvent&, void*);

        Viz3d(const String& window_name = String());
        Viz3d(const Viz3d&);
        Viz3d& operator=(const Viz3d&);
        ~Viz3d();

        void showWidget(const String &id, const Widget &widget, const Affine3f &pose = Affine3f::Identity());
        void removeWidget(const String &id);
        Widget getWidget(const String &id) const;
        void removeAllWidgets();

        void setWidgetPose(const String &id, const Affine3f &pose);
        void updateWidgetPose(const String &id, const Affine3f &pose);
        Affine3f getWidgetPose(const String &id) const;

        void setCamera(const Camera &camera);
        Camera getCamera() const;
        Affine3f getViewerPose();
        void setViewerPose(const Affine3f &pose);

        void resetCameraViewpoint (const String &id);
        void resetCamera();

        void convertToWindowCoordinates(const Point3d &pt, Point3d &window_coord);
        void converTo3DRay(const Point3d &window_coord, Point3d &origin, Vec3d &direction);

        Size getWindowSize() const;
        void setWindowSize(const Size &window_size);
        String getWindowName() const;
        void saveScreenshot (const String &file);
        void setWindowPosition (int x, int y);
        void setFullScreen (bool mode);
        void setBackgroundColor(const Color& color = Color::black());

        void spin();
        void spinOnce(int time = 1, bool force_redraw = false);
        bool wasStopped() const;

        void registerKeyboardCallback(KeyboardCallback callback, void* cookie = 0);
        void registerMouseCallback(MouseCallback callback, void* cookie = 0);

        void setRenderingProperty(const String &id, int property, double value);
        double getRenderingProperty(const String &id, int property);

        void setDesiredUpdateRate(double rate);
        double getDesiredUpdateRate();

        void setRepresentation(int representation);
    private:
        /* hidden */
    };

viz::Viz3d::Viz3d
-----------------
The constructors.

.. ocv:function:: Viz3d::Viz3d(const String& window_name = String())

    :param window_name: Name of the window.

viz::Viz3d::showWidget
----------------------
Shows a widget in the window.

.. ocv:function:: void Viz3d::showWidget(const String &id, const Widget &widget, const Affine3f &pose = Affine3f::Identity())

    :param id: A unique id for the widget.
    :param widget: The widget to be displayed in the window.
    :param pose: Pose of the widget.

viz::Viz3d::removeWidget
------------------------
Removes a widget from the window.

.. ocv:function:: void removeWidget(const String &id)

    :param id: The id of the widget that will be removed.

viz::Viz3d::getWidget
---------------------
Retrieves a widget from the window. A widget is implicitly shared;
that is, if the returned widget is modified, the changes will be
immediately visible in the window.

.. ocv:function:: Widget getWidget(const String &id) const

    :param id: The id of the widget that will be returned.

viz::Viz3d::removeAllWidgets
----------------------------
Removes all widgets from the window.

.. ocv:function:: void removeAllWidgets()

viz::Viz3d::setWidgetPose
-------------------------
Sets pose of a widget in the window.

.. ocv:function:: void setWidgetPose(const String &id, const Affine3f &pose)

    :param id: The id of the widget whose pose will be set.
    :param pose: The new pose of the widget.

viz::Viz3d::updateWidgetPose
----------------------------
Updates pose of a widget in the window by pre-multiplying its current pose.

.. ocv:function:: void updateWidgetPose(const String &id, const Affine3f &pose)

    :param id: The id of the widget whose pose will be updated.
    :param pose: The pose that the current pose of the widget will be pre-multiplied by.

viz::Viz3d::getWidgetPose
-------------------------
Returns the current pose of a widget in the window.

.. ocv:function:: Affine3f getWidgetPose(const String &id) const

    :param id: The id of the widget whose pose will be returned.

viz::Viz3d::setCamera
---------------------
Sets the intrinsic parameters of the viewer using Camera.

.. ocv:function:: void setCamera(const Camera &camera)

    :param camera: Camera object wrapping intrinsinc parameters.

viz::Viz3d::getCamera
---------------------
Returns a camera object that contains intrinsic parameters of the current viewer.

.. ocv:function:: Camera getCamera() const

viz::Viz3d::getViewerPose
-------------------------
Returns the current pose of the viewer.

..ocv:function:: Affine3f getViewerPose()

viz::Viz3d::setViewerPose
-------------------------
Sets pose of the viewer.

.. ocv:function:: void setViewerPose(const Affine3f &pose)

    :param pose: The new pose of the viewer.

viz::Viz3d::resetCameraViewpoint
--------------------------------
Resets camera viewpoint to a 3D widget in the scene.

.. ocv:function:: void resetCameraViewpoint (const String &id)

    :param pose: Id of a 3D widget.

viz::Viz3d::resetCamera
-----------------------
Resets camera.

.. ocv:function:: void resetCamera()

viz::Viz3d::convertToWindowCoordinates
--------------------------------------
Transforms a point in world coordinate system to window coordinate system.

.. ocv:function:: void convertToWindowCoordinates(const Point3d &pt, Point3d &window_coord)

    :param pt: Point in world coordinate system.
    :param window_coord: Output point in window coordinate system.

viz::Viz3d::converTo3DRay
-------------------------
Transforms a point in window coordinate system to a 3D ray in world coordinate system.

.. ocv:function:: void converTo3DRay(const Point3d &window_coord, Point3d &origin, Vec3d &direction)

    :param window_coord: Point in window coordinate system.
    :param origin: Output origin of the ray.
    :param direction: Output direction of the ray.

viz::Viz3d::getWindowSize
-------------------------
Returns the current size of the window.

.. ocv:function:: Size getWindowSize() const

viz::Viz3d::setWindowSize
-------------------------
Sets the size of the window.

.. ocv:function:: void setWindowSize(const Size &window_size)

    :param window_size: New size of the window.

viz::Viz3d::getWindowName
-------------------------
Returns the name of the window which has been set in the constructor.

.. ocv:function:: String getWindowName() const

viz::Viz3d::saveScreenshot
--------------------------
Saves screenshot of the current scene.

.. ocv:function:: void saveScreenshot(const String &file)

    :param file: Name of the file.

viz::Viz3d::setWindowPosition
-----------------------------
Sets the position of the window in the screen.

.. ocv:function:: void setWindowPosition(int x, int y)

    :param x: x coordinate of the window
    :param y: y coordinate of the window

viz::Viz3d::setFullScreen
-------------------------
Sets or unsets full-screen rendering mode.

.. ocv:function:: void setFullScreen(bool mode)

    :param mode: If true, window will use full-screen mode.

viz::Viz3d::setBackgroundColor
------------------------------
Sets background color.

.. ocv:function:: void setBackgroundColor(const Color& color = Color::black())

viz::Viz3d::spin
----------------
The window renders and starts the event loop.

.. ocv:function:: void spin()

viz::Viz3d::spinOnce
--------------------
Starts the event loop for a given time.

.. ocv:function:: void spinOnce(int time = 1, bool force_redraw = false)

    :param time: Amount of time in milliseconds for the event loop to keep running.
    :param force_draw: If true, window renders.

viz::Viz3d::wasStopped
----------------------
Returns whether the event loop has been stopped.

.. ocv:function:: bool wasStopped()

viz::Viz3d::registerKeyboardCallback
------------------------------------
Sets keyboard handler.

.. ocv:function:: void registerKeyboardCallback(KeyboardCallback callback, void* cookie = 0)

    :param callback: Keyboard callback ``(void (*KeyboardCallbackFunction(const KeyboardEvent&, void*))``.
    :param cookie: The optional parameter passed to the callback.

viz::Viz3d::registerMouseCallback
---------------------------------
Sets mouse handler.

.. ocv:function:: void registerMouseCallback(MouseCallback callback, void* cookie = 0)

    :param callback: Mouse callback ``(void (*MouseCallback)(const MouseEvent&, void*))``.
    :param cookie: The optional parameter passed to the callback.

viz::Viz3d::setRenderingProperty
--------------------------------
Sets rendering property of a widget.

.. ocv:function:: void setRenderingProperty(const String &id, int property, double value)

    :param id: Id of the widget.
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

viz::Viz3d::getRenderingProperty
--------------------------------
Returns rendering property of a widget.

.. ocv:function:: double getRenderingProperty(const String &id, int property)

    :param id: Id of the widget.
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

viz::Viz3d::setDesiredUpdateRate
--------------------------------
Sets desired update rate of the window.

.. ocv:function:: void setDesiredUpdateRate(double rate)

    :param rate: Desired update rate. The default is 30.

viz::Viz3d::getDesiredUpdateRate
--------------------------------
Returns desired update rate of the window.

.. ocv:function:: double getDesiredUpdateRate()

viz::Viz3d::setRepresentation
-----------------------------
Sets geometry representation of the widgets to surface, wireframe or points.

.. ocv:function:: void setRepresentation(int representation)

    :param representation: Geometry representation which can be one of the following:

        * **REPRESENTATION_POINTS**
        * **REPRESENTATION_WIREFRAME**
        * **REPRESENTATION_SURFACE**

viz::Color
----------
.. ocv:class:: Color

This class a represents BGR color. ::

    class CV_EXPORTS Color : public Scalar
    {
    public:
        Color();
        Color(double gray);
        Color(double blue, double green, double red);

        Color(const Scalar& color);

        static Color black();
        static Color blue();
        static Color green();
        static Color cyan();

        static Color red();
        static Color magenta();
        static Color yellow();
        static Color white();

        static Color gray();
    };

viz::Mesh3d
-----------
.. ocv:class:: Mesh3d

This class wraps mesh attributes, and it can load a mesh from a ``ply`` file. ::

    class CV_EXPORTS Mesh3d
    {
    public:

        Mat cloud, colors;
        Mat polygons;

        //! Loads mesh from a given ply file
        static Mesh3d loadMesh(const String& file);

    private:
        /* hidden */
    };

viz::Mesh3d::loadMesh
---------------------
Loads a mesh from a ``ply`` file.

.. ocv:function:: static Mesh3d loadMesh(const String& file)

    :param file: File name.


viz::KeyboardEvent
------------------
.. ocv:class:: KeyboardEvent

This class represents a keyboard event. ::

    class CV_EXPORTS KeyboardEvent
    {
    public:
        static const unsigned int Alt   = 1;
        static const unsigned int Ctrl  = 2;
        static const unsigned int Shift = 4;

        //! Create a keyboard event
        //! - Note that action is true if key is pressed, false if released
        KeyboardEvent (bool action, const std::string& key_sym, unsigned char key, bool alt, bool ctrl, bool shift);

        bool isAltPressed () const;
        bool isCtrlPressed () const;
        bool isShiftPressed () const;

        unsigned char getKeyCode () const;

        const String& getKeySym () const;
        bool keyDown () const;
        bool keyUp () const;

    protected:
        /* hidden */
    };

viz::KeyboardEvent::KeyboardEvent
---------------------------------
Constructs a KeyboardEvent.

.. ocv:function:: KeyboardEvent (bool action, const std::string& key_sym, unsigned char key, bool alt, bool ctrl, bool shift)

    :param action: If true, key is pressed. If false, key is released.
    :param key_sym: Name of the key.
    :param key: Code of the key.
    :param alt: If true, ``alt`` is pressed.
    :param ctrl: If true, ``ctrl`` is pressed.
    :param shift: If true, ``shift`` is pressed.

viz::MouseEvent
---------------
.. ocv:class:: MouseEvent

This class represents a mouse event. ::

    class CV_EXPORTS MouseEvent
    {
    public:
        enum Type { MouseMove = 1, MouseButtonPress, MouseButtonRelease, MouseScrollDown, MouseScrollUp, MouseDblClick } ;
        enum MouseButton { NoButton = 0, LeftButton, MiddleButton, RightButton, VScroll } ;

        MouseEvent (const Type& type, const MouseButton& button, const Point& p, bool alt, bool ctrl, bool shift);

        Type type;
        MouseButton button;
        Point pointer;
        unsigned int key_state;
    };

viz::MouseEvent::MouseEvent
---------------------------
Constructs a MouseEvent.

.. ocv:function:: MouseEvent (const Type& type, const MouseButton& button, const Point& p, bool alt, bool ctrl, bool shift)

    :param type: Type of the event. This can be **MouseMove**, **MouseButtonPress**, **MouseButtonRelease**, **MouseScrollDown**, **MouseScrollUp**, **MouseDblClick**.
    :param button: Mouse button. This can be **NoButton**, **LeftButton**, **MiddleButton**, **RightButton**, **VScroll**.
    :param p: Position of the event.
    :param alt: If true, ``alt`` is pressed.
    :param ctrl: If true, ``ctrl`` is pressed.
    :param shift: If true, ``shift`` is pressed.

viz::Camera
-----------
.. ocv:class:: Camera

This class wraps intrinsic parameters of a camera. It provides several constructors
that can extract the intrinsic parameters from ``field of view``, ``intrinsic matrix`` and
``projection matrix``. ::

    class CV_EXPORTS Camera
    {
    public:
        Camera(float f_x, float f_y, float c_x, float c_y, const Size &window_size);
        Camera(const Vec2f &fov, const Size &window_size);
        Camera(const cv::Matx33f &K, const Size &window_size);
        Camera(const cv::Matx44f &proj, const Size &window_size);

        inline const Vec2d & getClip() const { return clip_; }
        inline void setClip(const Vec2d &clip) { clip_ = clip; }

        inline const Size & getWindowSize() const { return window_size_; }
        void setWindowSize(const Size &window_size);

        inline const Vec2f & getFov() const { return fov_; }
        inline void setFov(const Vec2f & fov) { fov_ = fov; }

        inline const Vec2f & getPrincipalPoint() const { return principal_point_; }
        inline const Vec2f & getFocalLength() const { return focal_; }

        void computeProjectionMatrix(Matx44f &proj) const;

        static Camera KinectCamera(const Size &window_size);

    private:
        /* hidden */
    };

viz::Camera::Camera
-------------------
Constructs a Camera.

.. ocv:function:: Camera(float f_x, float f_y, float c_x, float c_y, const Size &window_size)

    :param f_x: Horizontal focal length.
    :param f_y: Vertical focal length.
    :param c_x: x coordinate of the principal point.
    :param c_y: y coordinate of the principal point.
    :param window_size: Size of the window. This together with focal length and principal point determines the field of view.

.. ocv:function:: Camera(const Vec2f &fov, const Size &window_size)

    :param fov: Field of view (horizontal, vertical)
    :param window_size: Size of the window.

    Principal point is at the center of the window by default.

.. ocv:function:: Camera(const cv::Matx33f &K, const Size &window_size)

    :param K: Intrinsic matrix of the camera.
    :param window_size: Size of the window. This together with intrinsic matrix determines the field of view.

.. ocv:function:: Camera(const cv::Matx44f &proj, const Size &window_size)

    :param proj: Projection matrix of the camera.
    :param window_size: Size of the window. This together with projection matrix determines the field of view.

viz::Camera::computeProjectionMatrix
------------------------------------
Computes projection matrix using intrinsic parameters of the camera.

.. ocv:function:: void computeProjectionMatrix(Matx44f &proj) const

    :param proj: Output projection matrix.

viz::Camera::KinectCamera
-------------------------
Creates a Kinect Camera.

.. ocv:function:: static Camera KinectCamera(const Size &window_size)

    :param window_size: Size of the window. This together with intrinsic matrix of a Kinect Camera determines the field of view.
