OpenGL interoperability
=======================

.. highlight:: cpp



General Information
-------------------
This section describes OpenGL interoperability.

To enable OpenGL support, configure OpenCV using ``CMake`` with ``WITH_OPENGL=ON`` .
Currently OpenGL is supported only with WIN32, GTK and Qt backends on Windows and Linux (MacOS and Android are not supported).
For GTK backend ``gtkglext-1.0`` library is required.

To use OpenGL functionality you should first create OpenGL context (window or frame buffer).
You can do this with :ocv:func:`namedWindow` function or with other OpenGL toolkit (GLUT, for example).



ogl::Buffer
-----------
Smart pointer for OpenGL buffer object with reference counting.

.. ocv:class:: ogl::Buffer

Buffer Objects are OpenGL objects that store an array of unformatted memory allocated by the OpenGL context.
These can be used to store vertex data, pixel data retrieved from images or the framebuffer, and a variety of other things.

``ogl::Buffer`` has interface similar with :ocv:class:`Mat` interface and represents 2D array memory.

``ogl::Buffer`` supports memory transfers between host and device and also can be mapped to CUDA memory.



ogl::Buffer::Target
-------------------
The target defines how you intend to use the buffer object.

.. ocv:enum:: ogl::Buffer::Target

    .. ocv:emember:: ARRAY_BUFFER

        The buffer will be used as a source for vertex data.

    .. ocv:emember:: ELEMENT_ARRAY_BUFFER

        The buffer will be used for indices (in ``glDrawElements`` or :ocv:func:`ogl::render`, for example).

    .. ocv:emember:: PIXEL_PACK_BUFFER

        The buffer will be used for reading from OpenGL textures.

    .. ocv:emember:: PIXEL_UNPACK_BUFFER

        The buffer will be used for writing to OpenGL textures.



ogl::Buffer::Buffer
-------------------
The constructors.

.. ocv:function:: ogl::Buffer::Buffer()

.. ocv:function:: ogl::Buffer::Buffer(int arows, int acols, int atype, unsigned int abufId, bool autoRelease = false)

.. ocv:function:: ogl::Buffer::Buffer(Size asize, int atype, unsigned int abufId, bool autoRelease = false)

.. ocv:function:: ogl::Buffer::Buffer(int arows, int acols, int atype, Target target = ARRAY_BUFFER, bool autoRelease = false)

.. ocv:function:: ogl::Buffer::Buffer(Size asize, int atype, Target target = ARRAY_BUFFER, bool autoRelease = false)

.. ocv:function:: ogl::Buffer::Buffer(InputArray arr, Target target = ARRAY_BUFFER, bool autoRelease = false)

    :param arows: Number of rows in a 2D array.

    :param acols: Number of columns in a 2D array.

    :param asize: 2D array size.

    :param atype: Array type ( ``CV_8UC1, ..., CV_64FC4`` ). See :ocv:class:`Mat` for details.

    :param abufId: Buffer object name.

    :param arr: Input array (host or device memory, it can be :ocv:class:`Mat` , :ocv:class:`cuda::GpuMat` or ``std::vector`` ).

    :param target: Buffer usage. See :ocv:enum:`ogl::Buffer::Target` .

    :param autoRelease: Auto release mode (if true, release will be called in object's destructor).

Creates empty ``ogl::Buffer`` object, creates ``ogl::Buffer`` object from existed buffer ( ``abufId`` parameter),
allocates memory for ``ogl::Buffer`` object or copies from host/device memory.



ogl::Buffer::create
-------------------
Allocates memory for ``ogl::Buffer`` object.

.. ocv:function:: void ogl::Buffer::create(int arows, int acols, int atype, Target target = ARRAY_BUFFER, bool autoRelease = false)

.. ocv:function:: void ogl::Buffer::create(Size asize, int atype, Target target = ARRAY_BUFFER, bool autoRelease = false)

    :param arows: Number of rows in a 2D array.

    :param acols: Number of columns in a 2D array.

    :param asize: 2D array size.

    :param atype: Array type ( ``CV_8UC1, ..., CV_64FC4`` ). See :ocv:class:`Mat` for details.

    :param target: Buffer usage. See :ocv:enum:`ogl::Buffer::Target` .

    :param autoRelease: Auto release mode (if true, release will be called in object's destructor).



ogl::Buffer::release
--------------------
Decrements the reference counter and destroys the buffer object if needed.

.. ocv:function:: void ogl::Buffer::release()

The function will call `setAutoRelease(true)` .



ogl::Buffer::setAutoRelease
---------------------------
Sets auto release mode.

.. ocv:function:: void ogl::Buffer::setAutoRelease(bool flag)

    :param flag: Auto release mode (if true, release will be called in object's destructor).

The lifetime of the OpenGL object is tied to the lifetime of the context.
If OpenGL context was bound to a window it could be released at any time (user can close a window).
If object's destructor is called after destruction of the context it will cause an error.
Thus ``ogl::Buffer`` doesn't destroy OpenGL object in destructor by default (all OpenGL resources will be released with OpenGL context).
This function can force ``ogl::Buffer`` destructor to destroy OpenGL object.



ogl::Buffer::copyFrom
---------------------
Copies from host/device memory to OpenGL buffer.

.. ocv:function:: void ogl::Buffer::copyFrom(InputArray arr, Target target = ARRAY_BUFFER, bool autoRelease = false)

    :param arr: Input array (host or device memory, it can be :ocv:class:`Mat` , :ocv:class:`cuda::GpuMat` or ``std::vector`` ).

    :param target: Buffer usage. See :ocv:enum:`ogl::Buffer::Target` .

    :param autoRelease: Auto release mode (if true, release will be called in object's destructor).



ogl::Buffer::copyTo
-------------------
Copies from OpenGL buffer to host/device memory or another OpenGL buffer object.

.. ocv:function:: void ogl::Buffer::copyTo(OutputArray arr) const

    :param arr: Destination array (host or device memory, can be :ocv:class:`Mat` , :ocv:class:`cuda::GpuMat` , ``std::vector`` or ``ogl::Buffer`` ).



ogl::Buffer::clone
------------------
Creates a full copy of the buffer object and the underlying data.

.. ocv:function:: Buffer ogl::Buffer::clone(Target target = ARRAY_BUFFER, bool autoRelease = false) const

    :param target: Buffer usage for destination buffer.

    :param autoRelease: Auto release mode for destination buffer.



ogl::Buffer::bind
-----------------
Binds OpenGL buffer to the specified buffer binding point.

.. ocv:function:: void ogl::Buffer::bind(Target target) const

    :param target: Binding point. See :ocv:enum:`ogl::Buffer::Target` .



ogl::Buffer::unbind
-------------------
Unbind any buffers from the specified binding point.

.. ocv:function:: static void ogl::Buffer::unbind(Target target)

    :param target: Binding point. See :ocv:enum:`ogl::Buffer::Target` .



ogl::Buffer::mapHost
--------------------
Maps OpenGL buffer to host memory.

.. ocv:function:: Mat ogl::Buffer::mapHost(Access access)

    :param access: Access policy, indicating whether it will be possible to read from, write to, or both read from and write to the buffer object's mapped data store. The symbolic constant must be ``ogl::Buffer::READ_ONLY`` , ``ogl::Buffer::WRITE_ONLY`` or ``ogl::Buffer::READ_WRITE`` .

``mapHost`` maps to the client's address space the entire data store of the buffer object.
The data can then be directly read and/or written relative to the returned pointer, depending on the specified ``access`` policy.

A mapped data store must be unmapped with :ocv:func:`ogl::Buffer::unmapHost` before its buffer object is used.

This operation can lead to memory transfers between host and device.

Only one buffer object can be mapped at a time.



ogl::Buffer::unmapHost
----------------------
Unmaps OpenGL buffer.

.. ocv:function:: void ogl::Buffer::unmapHost()



ogl::Buffer::mapDevice
----------------------
Maps OpenGL buffer to CUDA device memory.

.. ocv:function:: cuda::GpuMat ogl::Buffer::mapDevice()

This operatation doesn't copy data.
Several buffer objects can be mapped to CUDA memory at a time.

A mapped data store must be unmapped with :ocv:func:`ogl::Buffer::unmapDevice` before its buffer object is used.



ogl::Buffer::unmapDevice
------------------------
Unmaps OpenGL buffer.

.. ocv:function:: void ogl::Buffer::unmapDevice()



ogl::Texture2D
--------------
Smart pointer for OpenGL 2D texture memory with reference counting.

.. ocv:class:: ogl::Texture2D



ogl::Texture2D::Format
----------------------
An Image Format describes the way that the images in Textures store their data.

.. ocv:enum:: ogl::Texture2D::Format

    .. ocv:emember:: NONE
    .. ocv:emember:: DEPTH_COMPONENT
    .. ocv:emember:: RGB
    .. ocv:emember:: RGBA



ogl::Texture2D::Texture2D
-------------------------
The constructors.

.. ocv:function:: ogl::Texture2D::Texture2D()

.. ocv:function:: ogl::Texture2D::Texture2D(int arows, int acols, Format aformat, unsigned int atexId, bool autoRelease = false)

.. ocv:function:: ogl::Texture2D::Texture2D(Size asize, Format aformat, unsigned int atexId, bool autoRelease = false)

.. ocv:function:: ogl::Texture2D::Texture2D(int arows, int acols, Format aformat, bool autoRelease = false)

.. ocv:function:: ogl::Texture2D::Texture2D(Size asize, Format aformat, bool autoRelease = false)

.. ocv:function:: ogl::Texture2D::Texture2D(InputArray arr, bool autoRelease = false)

    :param arows: Number of rows.

    :param acols: Number of columns.

    :param asize: 2D array size.

    :param aformat: Image format. See :ocv:enum:`ogl::Texture2D::Format` .

    :param arr: Input array (host or device memory, it can be :ocv:class:`Mat` , :ocv:class:`cuda::GpuMat` or :ocv:class:`ogl::Buffer` ).

    :param autoRelease: Auto release mode (if true, release will be called in object's destructor).

Creates empty ``ogl::Texture2D`` object, allocates memory for ``ogl::Texture2D`` object or copies from host/device memory.



ogl::Texture2D::create
----------------------
Allocates memory for ``ogl::Texture2D`` object.

.. ocv:function:: void ogl::Texture2D::create(int arows, int acols, Format aformat, bool autoRelease = false)

.. ocv:function:: void ogl::Texture2D::create(Size asize, Format aformat, bool autoRelease = false)

    :param arows: Number of rows.

    :param acols: Number of columns.

    :param asize: 2D array size.

    :param aformat: Image format. See :ocv:enum:`ogl::Texture2D::Format` .

    :param autoRelease: Auto release mode (if true, release will be called in object's destructor).



ogl::Texture2D::release
-----------------------
Decrements the reference counter and destroys the texture object if needed.

.. ocv:function:: void ogl::Texture2D::release()

The function will call `setAutoRelease(true)` .



ogl::Texture2D::setAutoRelease
------------------------------
Sets auto release mode.

.. ocv:function:: void ogl::Texture2D::setAutoRelease(bool flag)

    :param flag: Auto release mode (if true, release will be called in object's destructor).

The lifetime of the OpenGL object is tied to the lifetime of the context.
If OpenGL context was bound to a window it could be released at any time (user can close a window).
If object's destructor is called after destruction of the context it will cause an error.
Thus ``ogl::Texture2D`` doesn't destroy OpenGL object in destructor by default (all OpenGL resources will be released with OpenGL context).
This function can force ``ogl::Texture2D`` destructor to destroy OpenGL object.



ogl::Texture2D::copyFrom
------------------------
Copies from host/device memory to OpenGL texture.

.. ocv:function:: void ogl::Texture2D::copyFrom(InputArray arr, bool autoRelease = false)

    :param arr: Input array (host or device memory, it can be :ocv:class:`Mat` , :ocv:class:`cuda::GpuMat` or :ocv:class:`ogl::Buffer` ).

    :param autoRelease: Auto release mode (if true, release will be called in object's destructor).



ogl::Texture2D::copyTo
----------------------
Copies from OpenGL texture to host/device memory or another OpenGL texture object.

.. ocv:function:: void ogl::Texture2D::copyTo(OutputArray arr, int ddepth = CV_32F, bool autoRelease = false) const

    :param arr: Destination array (host or device memory, can be :ocv:class:`Mat` , :ocv:class:`cuda::GpuMat` , :ocv:class:`ogl::Buffer` or ``ogl::Texture2D`` ).

    :param ddepth: Destination depth.

    :param autoRelease: Auto release mode for destination buffer (if ``arr`` is OpenGL buffer or texture).



ogl::Texture2D::bind
--------------------
Binds texture to current active texture unit for ``GL_TEXTURE_2D`` target.

.. ocv:function:: void ogl::Texture2D::bind() const



ogl::Arrays
-----------
Wrapper for OpenGL Client-Side Vertex arrays.

.. ocv:class:: ogl::Arrays

``ogl::Arrays`` stores vertex data in :ocv:class:`ogl::Buffer` objects.



ogl::Arrays::setVertexArray
---------------------------
Sets an array of vertex coordinates.

.. ocv:function:: void ogl::Arrays::setVertexArray(InputArray vertex)

    :param vertex: array with vertex coordinates, can be both host and device memory.



ogl::Arrays::resetVertexArray
-----------------------------
Resets vertex coordinates.

.. ocv:function:: void ogl::Arrays::resetVertexArray()



ogl::Arrays::setColorArray
--------------------------
Sets an array of vertex colors.

.. ocv:function:: void ogl::Arrays::setColorArray(InputArray color)

    :param color: array with vertex colors, can be both host and device memory.



ogl::Arrays::resetColorArray
----------------------------
Resets vertex colors.

.. ocv:function:: void ogl::Arrays::resetColorArray()



ogl::Arrays::setNormalArray
---------------------------
Sets an array of vertex normals.

.. ocv:function:: void ogl::Arrays::setNormalArray(InputArray normal)

    :param normal: array with vertex normals, can be both host and device memory.



ogl::Arrays::resetNormalArray
-----------------------------
Resets vertex normals.

.. ocv:function:: void ogl::Arrays::resetNormalArray()



ogl::Arrays::setTexCoordArray
-----------------------------
Sets an array of vertex texture coordinates.

.. ocv:function:: void ogl::Arrays::setTexCoordArray(InputArray texCoord)

    :param texCoord: array with vertex texture coordinates, can be both host and device memory.



ogl::Arrays::resetTexCoordArray
-------------------------------
Resets vertex texture coordinates.

.. ocv:function:: void ogl::Arrays::resetTexCoordArray()



ogl::Arrays::release
--------------------
Releases all inner buffers.

.. ocv:function:: void ogl::Arrays::release()



ogl::Arrays::setAutoRelease
---------------------------
Sets auto release mode all inner buffers.

.. ocv:function:: void ogl::Arrays::setAutoRelease(bool flag)

    :param flag: Auto release mode.



ogl::Arrays::bind
-----------------
Binds all vertex arrays.

.. ocv:function:: void ogl::Arrays::bind() const



ogl::Arrays::size
-----------------
Returns the vertex count.

.. ocv:function:: int ogl::Arrays::size() const



ogl::render
-----------
Render OpenGL texture or primitives.

.. ocv:function:: void ogl::render(const Texture2D& tex, Rect_<double> wndRect = Rect_<double>(0.0, 0.0, 1.0, 1.0), Rect_<double> texRect = Rect_<double>(0.0, 0.0, 1.0, 1.0))

.. ocv:function:: void ogl::render(const Arrays& arr, int mode = POINTS, Scalar color = Scalar::all(255))

.. ocv:function:: void ogl::render(const Arrays& arr, InputArray indices, int mode = POINTS, Scalar color = Scalar::all(255))

    :param tex: Texture to draw.

    :param wndRect: Region of window, where to draw a texture (normalized coordinates).

    :param texRect: Region of texture to draw (normalized coordinates).

    :param arr: Array of privitives vertices.

    :param indices: Array of vertices indices (host or device memory).

    :param mode: Render mode. Available options:

        * **POINTS**
        * **LINES**
        * **LINE_LOOP**
        * **LINE_STRIP**
        * **TRIANGLES**
        * **TRIANGLE_STRIP**
        * **TRIANGLE_FAN**
        * **QUADS**
        * **QUAD_STRIP**
        * **POLYGON**

    :param color: Color for all vertices. Will be used if ``arr`` doesn't contain color array.



cuda::setGlDevice
-----------------
Sets a CUDA device and initializes it for the current thread with OpenGL interoperability.

.. ocv:function:: void cuda::setGlDevice( int device = 0 )

    :param device: System index of a CUDA device starting with 0.

This function should be explicitly called after OpenGL context creation and before any CUDA calls.
