Widget
======
	
.. highlight:: cpp

In this section, the built-in widgets are presented. 

Widget
------
.. ocv:class:: Widget

Base class of all widgets. Widget is implicitly shared.::

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

Widget::fromPlyFile
-------------------
Creates a widget from ply file.

.. ocv:function:: static Widget fromPlyFile(const String &file_name)

	:param file_name: Ply file name.
	
Widget::setRenderingProperty
----------------------------
Sets rendering property of the widget.

.. ocv:function:: void setRenderingProperty(int property, double value)

	:param property: Property that will be modified.
	:param value: The new value of the property.
	
Widget::getRenderingProperty
----------------------------
Returns rendering property of the widget.

.. ocv:function:: double getRenderingProperty(int property) const

	:param property: Property.
	
Widget::cast
------------
Casts a widget to another.

.. ocv:function:: template<typename _W> _W cast()

Widget3D
--------
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

Widget3D::setPose
-----------------
Sets pose of the widget.

.. ocv:function:: void setPose(const Affine3f &pose)

	:param pose: The new pose of the widget.
	
Widget3D::updateWidgetPose
--------------------------
Updates pose of the widget by pre-multiplying its current pose.

.. ocv:function:: void updateWidgetPose(const Affine3f &pose)

	:param pose: The pose that the current pose of the widget will be pre-multiplied by.

Widget3D::getPose
-----------------
Returns the current pose of the widget.

.. ocv:function:: Affine3f getWidgetPose() const

Widget3D::setColor
------------------
Sets the color of the widget.

.. ocv:function:: void setColor(const Color &color)

	:param color: Color
	
Widget2D
--------
.. ocv:class:: Widget2D

Base class of all 2D widgets. ::

	class CV_EXPORTS Widget2D : public Widget
	{
	public:
		Widget2D() {}

		void setColor(const Color &color);
	};
	
Widget2D::setColor
------------------
Sets the color of the widget.

.. ocv:function:: void setColor(const Color &color)

	:param color: Color

LineWidget
----------
.. ocv:class:: LineWidget

This 3D Widget defines a finite line. ::

	class CV_EXPORTS LineWidget : public Widget3D
	{
	public:
		LineWidget(const Point3f &pt1, const Point3f &pt2, const Color &color = Color::white());
	};
	
LineWidget::LineWidget
----------------------
Constructs a LineWidget.

.. ocv:function:: LineWidget(const Point3f &pt1, const Point3f &pt2, const Color &color = Color::white())

	:param pt1: Start point of the line.
	:param pt2: End point of the line.
	:param color: Color of the line.
	
PlaneWidget
-----------
.. ocv:class:: PlaneWidget

This 3D Widget defines a finite plane. ::

	class CV_EXPORTS PlaneWidget : public Widget3D
	{
	public:
		PlaneWidget(const Vec4f& coefs, double size = 1.0, const Color &color = Color::white());
		PlaneWidget(const Vec4f& coefs, const Point3f& pt, double size = 1.0, const Color &color = Color::white());
	private:
		/* hidden */
	};
	
PlaneWidget::PlaneWidget
------------------------
Constructs a PlaneWidget.

.. ocv:function:: PlaneWidget(const Vec4f& coefs, double size = 1.0, const Color &color = Color::white())
	
	:param coefs: Plane coefficients as in (A,B,C,D) where Ax + By + Cz + D = 0.
	:param size: Size of the plane.
	:param color: Color of the plane.

.. ocv:function:: PlaneWidget(const Vec4f& coefs, const Point3f& pt, double size = 1.0, const Color &color = Color::white())

	:param coefs: Plane coefficients as in (A,B,C,D) where Ax + By + Cz + D = 0.
	:param pt: Position of the plane.
	:param color: Color of the plane.
	
SphereWidget
------------
.. ocv:class:: SphereWidget

This 3D Widget defines a sphere. ::

	class CV_EXPORTS SphereWidget : public Widget3D
	{
	public:
		SphereWidget(const cv::Point3f &center, float radius, int sphere_resolution = 10, const Color &color = Color::white())
	};

SphereWidget::SphereWidget
--------------------------
Constructs a SphereWidget.

.. ocv:function:: SphereWidget(const cv::Point3f &center, float radius, int sphere_resolution = 10, const Color &color = Color::white())

	:param center: Center of the sphere.
	:param radius: Radius of the sphere.
	:param sphere_resolution: Resolution of the sphere.
	:param color: Color of the sphere.

ArrowWidget
-----------
.. ocv:class:: ArrowWidget

This 3D Widget defines an arrow. ::

	class CV_EXPORTS ArrowWidget : public Widget3D
	{
	public:
		ArrowWidget(const Point3f& pt1, const Point3f& pt2, double thickness = 0.03, const Color &color = Color::white());
	};
	
ArrowWidget::ArrowWidget
------------------------
Constructs an ArrowWidget.

.. ocv:function:: ArrowWidget(const Point3f& pt1, const Point3f& pt2, double thickness = 0.03, const Color &color = Color::white())

	:param pt1: Start point of the arrow.
	:param pt2: End point of the arrow.
	:param thickness: Thickness of the arrow. Thickness of arrow head is also adjusted accordingly.
	:param color: Color of the arrow.
	
Arrow head is located at the end point of the arrow.
	
CircleWidget
------------
.. ocv:class:: CircleWidget

This 3D Widget defines a circle. ::

	class CV_EXPORTS CircleWidget : public Widget3D
	{
	public:
		CircleWidget(const Point3f& pt, double radius, double thickness = 0.01, const Color &color = Color::white());
	};
	
CircleWidget::CircleWidget
--------------------------
Constructs a CircleWidget.

.. ocv:function:: CircleWidget(const Point3f& pt, double radius, double thickness = 0.01, const Color &color = Color::white())

	:param pt: Center of the circle.
	:param radius: Radius of the circle.
	:param thickness: Thickness of the circle.
	:param color: Color of the circle.
	
CylinderWidget
--------------
.. ocv:class:: CylinderWidget

This 3D Widget defines a cylinder. ::

	class CV_EXPORTS CylinderWidget : public Widget3D
	{
	public:
		CylinderWidget(const Point3f& pt_on_axis, const Point3f& axis_direction, double radius, int numsides = 30, const Color &color = Color::white());
	};

CylinderWidget::CylinderWidget
------------------------------
Constructs a CylinderWidget.

.. ocv:function:: CylinderWidget(const Point3f& pt_on_axis, const Point3f& axis_direction, double radius, int numsides = 30, const Color &color = Color::white())

	:param pt_on_axis: A point on the axis of the cylinder.
	:param axis_direction: Direction of the axis of the cylinder.
	:param radius: Radius of the cylinder.
	:param numsides: Resolution of the cylinder.
	:param color: Color of the cylinder.
	
CubeWidget
----------
.. ocv:class:: CubeWidget

This 3D Widget defines a cube. ::

	class CV_EXPORTS CubeWidget : public Widget3D
	{
	public:
		CubeWidget(const Point3f& pt_min, const Point3f& pt_max, bool wire_frame = true, const Color &color = Color::white());
	};
	
CubeWidget::CubeWidget
----------------------
Constructs a CudeWidget.

.. ocv:function:: CubeWidget(const Point3f& pt_min, const Point3f& pt_max, bool wire_frame = true, const Color &color = Color::white())

	:param pt_min: Specifies minimum point of the bounding box.
	:param pt_max: Specifies maximum point of the bounding box.
	:param wire_frame: If true, cube is represented as wireframe.
	:param color: Color of the cube.
	
CoordinateSystemWidget
----------------------
.. ocv:class:: CoordinateSystemWidget

This 3D Widget represents a coordinate system. ::

	class CV_EXPORTS CoordinateSystemWidget : public Widget3D
	{
	public:
		CoordinateSystemWidget(double scale = 1.0);
	};
	
CoordinateSystemWidget::CoordinateSystemWidget
----------------------------------------------
Constructs a CoordinateSystemWidget.

.. ocv:function:: CoordinateSystemWidget(double scale = 1.0)

	:param scale: Determines the size of the axes.
	
PolyLineWidget
--------------
.. ocv:class:: PolyLineWidget

This 3D Widget defines a poly line. ::

	class CV_EXPORTS PolyLineWidget : public Widget3D
	{
	public:
		PolyLineWidget(InputArray points, const Color &color = Color::white());

	private:
		/* hidden */
	};

PolyLineWidget::PolyLineWidget
------------------------------
Constructs a PolyLineWidget.

.. ocv:function:: PolyLineWidget(InputArray points, const Color &color = Color::white())
	
	:param points: Point set.
	:param color: Color of the poly line.
	
GridWidget
----------
.. ocv:class:: GridWidget

This 3D Widget defines a grid. ::

	class CV_EXPORTS GridWidget : public Widget3D
	{
	public:
		//! Creates grid at the origin
		GridWidget(const Vec2i &dimensions, const Vec2d &spacing, const Color &color = Color::white());
		//! Creates grid based on the plane equation
		GridWidget(const Vec4f &coeffs, const Vec2i &dimensions, const Vec2d &spacing, const Color &color = Color::white());
	private:
		/* hidden */
	};
	
GridWidget::GridWidget
----------------------
Constructs a GridWidget.

.. ocv:function:: GridWidget(const Vec2i &dimensions, const Vec2d &spacing, const Color &color = Color::white())

	:param dimensions: Number of columns and rows, respectively.
	:param spacing: Size of each column and row, respectively.
	:param color: Color of the grid.
	
.. ocv:function:  GridWidget(const Vec4f &coeffs, const Vec2i &dimensions, const Vec2d &spacing, const Color &color = Color::white())
	
	:param coeffs: Plane coefficients as in (A,B,C,D) where Ax + By + Cz + D = 0.
	:param dimensions: Number of columns and rows, respectively.
	:param spacing: Size of each column and row, respectively.
	:param color: Color of the grid.
	
Text3DWidget
------------
.. ocv:class:: Text3DWidget

This 3D Widget represents 3D text. The text always faces the camera. ::

	class CV_EXPORTS Text3DWidget : public Widget3D
	{
	public:
		Text3DWidget(const String &text, const Point3f &position, double text_scale = 1.0, const Color &color = Color::white());

		void setText(const String &text);
		String getText() const;
	};
	
Text3DWidget::Text3DWidget
--------------------------
Constructs a Text3DWidget.

.. ocv:function:: Text3DWidget(const String &text, const Point3f &position, double text_scale = 1.0, const Color &color = Color::white())

	:param text: Text content of the widget.
	:param position: Position of the text.
	:param text_scale: Size of the text.
	:param color: Color of the text.
	
Text3DWidget::setText
---------------------
Sets the text content of the widget.

.. ocv:function:: void setText(const String &text)

	:param text: Text content of the widget.

Text3DWidget::getText
---------------------
Returns the current text content of the widget.

.. ocv:function:: String getText() const

TextWidget
----------
.. ocv:class:: TextWidget

This 2D Widget represents text overlay. ::

	class CV_EXPORTS TextWidget : public Widget2D
	{
	public:
		TextWidget(const String &text, const Point2i &pos, int font_size = 10, const Color &color = Color::white());

		void setText(const String &text);
		String getText() const;
	};
	
TextWidget::TextWidget
----------------------
Constructs a TextWidget.

.. ocv:function:: TextWidget(const String &text, const Point2i &pos, int font_size = 10, const Color &color = Color::white())

	:param text: Text content of the widget.
	:param pos: Position of the text.
	:param font_size: Font size.
	:param color: Color of the text.
	
TextWidget::setText
---------------------
Sets the text content of the widget.

.. ocv:function:: void setText(const String &text)

	:param text: Text content of the widget.

TextWidget::getText
---------------------
Returns the current text content of the widget.

.. ocv:function:: String getText() const

ImageOverlayWidget
------------------
.. ocv:class:: ImageOverlayWidget

This 2D Widget represents an image overlay. ::

	class CV_EXPORTS ImageOverlayWidget : public Widget2D
	{
	public:
		ImageOverlayWidget(const Mat &image, const Rect &rect);
		
		void setImage(const Mat &image);
	};
	
ImageOverlayWidget::ImageOverlayWidget
--------------------------------------
Constructs a ImageOverlayWidget.

.. ocv:function:: ImageOverlayWidget(const Mat &image, const Rect &rect)

	:param image: BGR or Gray-Scale image.
	:param rect: Image is scaled and positioned based on rect.
	
ImageOverlayWidget::setImage
----------------------------
Sets the image content of the widget.

.. ocv:function:: void setImage(const Mat &image)

	:param image: BGR or Gray-Scale image.
	
Image3DWidget
-------------
.. ocv:class:: Image3DWidget

This 3D Widget represents 3D image. ::

	class CV_EXPORTS Image3DWidget : public Widget3D
	{
	public:
		//! Creates 3D image at the origin
		Image3DWidget(const Mat &image, const Size &size);
		//! Creates 3D image at a given position, pointing in the direction of the normal, and having the up_vector orientation
		Image3DWidget(const Vec3f &position, const Vec3f &normal, const Vec3f &up_vector, const Mat &image, const Size &size);
		
		void setImage(const Mat &image);
	};

Image3DWidget::Image3DWidget
----------------------------
Constructs a Image3DWidget.

.. ocv:function:: Image3DWidget(const Mat &image, const Size &size)
	
	:param image: BGR or Gray-Scale image.
	:param size: Size of the image.
	
.. ocv:function:: Image3DWidget(const Vec3f &position, const Vec3f &normal, const Vec3f &up_vector, const Mat &image, const Size &size)

	:param position: Position of the image.
	:param normal: Normal of the plane that represents the image.
	:param up_vector: Determines orientation of the image.
	:param image: BGR or Gray-Scale image.
	:param size: Size of the image.
	
Image3DWidget::setImage
-----------------------
Sets the image content of the widget.

.. ocv:function:: void setImage(const Mat &image)

	:param image: BGR or Gray-Scale image.
	
CameraPositionWidget
--------------------
.. ocv:class:: CameraPositionWidget

This 3D Widget represents camera position. ::

	class CV_EXPORTS CameraPositionWidget : public Widget3D
	{
	public:
		//! Creates camera coordinate frame (axes) at the origin
		CameraPositionWidget(double scale = 1.0);
		//! Creates frustum based on the intrinsic marix K at the origin
		CameraPositionWidget(const Matx33f &K, double scale = 1.0, const Color &color = Color::white());
		//! Creates frustum based on the field of view at the origin
		CameraPositionWidget(const Vec2f &fov, double scale = 1.0, const Color &color = Color::white());
		//! Creates frustum and display given image at the far plane
		CameraPositionWidget(const Matx33f &K, const Mat &img, double scale = 1.0, const Color &color = Color::white());
	};
	
CameraPositionWidget::CameraPositionWidget
------------------------------------------
Constructs a CameraPositionWidget.

.. ocv:function:: CameraPositionWidget(double scale = 1.0)

	Creates camera coordinate frame at the origin.
	
.. ocv:function:: CameraPositionWidget(const Matx33f &K, double scale = 1.0, const Color &color = Color::white())

	:param K: Intrinsic matrix of the camera.
	:param scale: Scale of the frustum.
	:param color: Color of the frustum.
	
	Creates viewing frustum of the camera based on its intrinsic matrix K.
	
.. ocv:function:: CameraPositionWidget(const Vec2f &fov, double scale = 1.0, const Color &color = Color::white())

	:param fov: Field of view of the camera (horizontal, vertical).
	:param scale: Scale of the frustum.
	:param color: Color of the frustum.
	
	Creates viewing frustum of the camera based on its field of view fov.

.. ocv:function:: CameraPositionWidget(const Matx33f &K, const Mat &img, double scale = 1.0, const Color &color = Color::white())

	:param K: Intrinsic matrix of the camera.
	:param img: BGR or Gray-Scale image that is going to be displayed at the far plane of the frustum.
	:param scale: Scale of the frustum and image.
	:param color: Color of the frustum.
	
	Creates viewing frustum of the camera based on its intrinsic matrix K, and displays image on the far end plane.
	
TrajectoryWidget
----------------
.. ocv:class:: TrajectoryWidget

This 3D Widget represents a trajectory. ::

	class CV_EXPORTS TrajectoryWidget : public Widget3D
	{
	public:
		enum {DISPLAY_FRAMES = 1, DISPLAY_PATH = 2};
		
		//! Displays trajectory of the given path either by coordinate frames or polyline
		TrajectoryWidget(const std::vector<Affine3f> &path, int display_mode = TrajectoryWidget::DISPLAY_PATH, const Color &color = Color::white(), double scale = 1.0);
		//! Displays trajectory of the given path by frustums
		TrajectoryWidget(const std::vector<Affine3f> &path, const Matx33f &K, double scale = 1.0, const Color &color = Color::white());
		//! Displays trajectory of the given path by frustums
		TrajectoryWidget(const std::vector<Affine3f> &path, const Vec2f &fov, double scale = 1.0, const Color &color = Color::white());
		
	private:
		/* hidden */
	};
	
TrajectoryWidget::TrajectoryWidget
----------------------------------
Constructs a TrajectoryWidget.

.. ocv:function:: TrajectoryWidget(const std::vector<Affine3f> &path, int display_mode = TrajectoryWidget::DISPLAY_PATH, const Color &color = Color::white(), double scale = 1.0)

	:param path: List of poses on a trajectory.
	:param display_mode: Display mode. This can be DISPLAY_PATH, DISPLAY_FRAMES, DISPLAY_PATH & DISPLAY_FRAMES.
	:param color: Color of the polyline that represents path. Frames are not affected.
	:param scale: Scale of the frames. Polyline is not affected.
	
	Displays trajectory of the given path as follows:
	
	* DISPLAY_PATH : Displays a poly line that represents the path.
	* DISPLAY_FRAMES : Displays coordinate frames at each pose.
	* DISPLAY_PATH & DISPLAY_FRAMES : Displays both poly line and coordinate frames.
	
.. ocv:function:: TrajectoryWidget(const std::vector<Affine3f> &path, const Matx33f &K, double scale = 1.0, const Color &color = Color::white())

	:param path: List of poses on a trajectory.
	:param K: Intrinsic matrix of the camera.
	:param scale: Scale of the frustums.
	:param color: Color of the frustums.
	
	Displays frustums at each pose of the trajectory.
	
.. ocv:function:: TrajectoryWidget(const std::vector<Affine3f> &path, const Vec2f &fov, double scale = 1.0, const Color &color = Color::white())

	:param path: List of poses on a trajectory.
	:param fov: Field of view of the camera (horizontal, vertical).
	:param scale: Scale of the frustums.
	:param color: Color of the frustums.
	
	Displays frustums at each pose of the trajectory.

SpheresTrajectoryWidget
-----------------------
.. ocv:class:: SpheresTrajectoryWidget

This 3D Widget represents a trajectory using spheres and lines, where spheres represent the positions of the camera, and lines
represent the direction from previous position to the current. ::

	class CV_EXPORTS SpheresTrajectoryWidget : public Widget3D
	{
	public:
		SpheresTrajectoryWidget(const std::vector<Affine3f> &path, float line_length = 0.05f, 
					double init_sphere_radius = 0.021, sphere_radius = 0.007, 
					Color &line_color = Color::white(), const Color &sphere_color = Color::white());
	};
	
SpheresTrajectoryWidget::SpheresTrajectoryWidget
------------------------------------------------
Constructs a SpheresTrajectoryWidget.

.. ocv:function:: SpheresTrajectoryWidget(const std::vector<Affine3f> &path, float line_length = 0.05f, double init_sphere_radius = 0.021, double sphere_radius = 0.007, const Color &line_color = Color::white(), const Color &sphere_color = Color::white())
	
	:param path: List of poses on a trajectory.
	:param line_length: Length of the lines.
	:param init_sphere_radius: Radius of the first sphere which represents the initial position of the camera.
	:param sphere_radius: Radius of the rest of the spheres.
	:param line_color: Color of the lines.
	:param sphere_color: Color of the spheres.
	
CloudWidget
-----------
.. ocv:class:: CloudWidget

This 3D Widget defines a point cloud. ::

	class CV_EXPORTS CloudWidget : public Widget3D
	{
	public:
		//! Each point in cloud is mapped to a color in colors
		CloudWidget(InputArray cloud, InputArray colors);
		//! All points in cloud have the same color
		CloudWidget(InputArray cloud, const Color &color = Color::white());

	private:
		/* hidden */
	};
	
CloudWidget::CloudWidget
------------------------
Constructs a CloudWidget.

.. ocv:function:: CloudWidget(InputArray cloud, InputArray colors)

	:param cloud: Point set which can be of type: CV_32FC3, CV_32FC4, CV_64FC3, CV_64FC4.
	:param colors: Set of colors. It has to be of the same size with cloud.
	
	Points in the cloud belong to mask when they are set to (NaN, NaN, NaN). 

.. ocv:function:: CloudWidget(InputArray cloud, const Color &color = Color::white())
	
	:param cloud: Point set which can be of type: CV_32FC3, CV_32FC4, CV_64FC3, CV_64FC4.
	:param color: A single color for the whole cloud.

	Points in the cloud belong to mask when they are set to (NaN, NaN, NaN). 

CloudCollectionWidget
---------------------
.. ocv:class:: CloudCollectionWidget

This 3D Widget defines a collection of clouds. ::

	class CV_EXPORTS CloudCollectionWidget : public Widget3D
	{
	public:
		CloudCollectionWidget();
		
		//! Each point in cloud is mapped to a color in colors
		void addCloud(InputArray cloud, InputArray colors, const Affine3f &pose = Affine3f::Identity());
		//! All points in cloud have the same color
		void addCloud(InputArray cloud, const Color &color = Color::white(), Affine3f &pose = Affine3f::Identity());
		
	private:
		/* hidden */
	};
	
CloudCollectionWidget::CloudCollectionWidget
--------------------------------------------
Constructs a CloudCollectionWidget.

.. ocv:function:: CloudCollectionWidget()

CloudCollectionWidget::addCloud
-------------------------------
Adds a cloud to the collection.

.. ocv:function:: void addCloud(InputArray cloud, InputArray colors, const Affine3f &pose = Affine3f::Identity())

	:param cloud: Point set which can be of type: CV_32FC3, CV_32FC4, CV_64FC3, CV_64FC4.
	:param colors: Set of colors. It has to be of the same size with cloud.
	:param pose: Pose of the cloud.
	
	Points in the cloud belong to mask when they are set to (NaN, NaN, NaN). 
	
.. ocv:function:: void addCloud(InputArray cloud, const Color &color = Color::white(), const Affine3f &pose = Affine3f::Identity())

	:param cloud: Point set which can be of type: CV_32FC3, CV_32FC4, CV_64FC3, CV_64FC4.
	:param colors: A single color for the whole cloud.
	:param pose: Pose of the cloud.
	
	Points in the cloud belong to mask when they are set to (NaN, NaN, NaN). 
	
CloudNormalsWidget
------------------
.. ocv:class:: CloudNormalsWidget

This 3D Widget represents normals of a point cloud. ::

	class CV_EXPORTS CloudNormalsWidget : public Widget3D
	{
	public:
		CloudNormalsWidget(InputArray cloud, InputArray normals, int level = 100, float scale = 0.02f, const Color &color = Color::white());

	private:
		/* hidden */
	};
	
CloudNormalsWidget::CloudNormalsWidget
--------------------------------------
Constructs a CloudNormalsWidget.

.. ocv:function:: CloudNormalsWidget(InputArray cloud, InputArray normals, int level = 100, float scale = 0.02f, const Color &color = Color::white())
	
	:param cloud: Point set which can be of type: CV_32FC3, CV_32FC4, CV_64FC3, CV_64FC4.
	:param normals: A set of normals that has to be of same type with cloud.
	:param level: Display only every levelth normal.
	:param scale: Scale of the arrows that represent normals.
	:param color: Color of the arrows that represent normals.
	
MeshWidget
----------
.. ocv:class:: MeshWidget

This 3D Widget defines a mesh. ::
	
	class CV_EXPORTS MeshWidget : public Widget3D
	{
	public:
		MeshWidget(const Mesh3d &mesh);
		
	private:
		/* hidden */
	};
	
MeshWidget::MeshWidget
----------------------
Constructs a MeshWidget.

.. ocv:function:: MeshWidget(const Mesh3d &mesh)

	:param mesh: Mesh object that will be displayed.




