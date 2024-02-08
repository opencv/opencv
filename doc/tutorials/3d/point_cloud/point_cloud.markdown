Point cloud visualisation {#tutorial_point_cloud}
==============================

|    |    |
| -: | :- |
| Original author | Dmitrii Klepikov |
| Compatibility | OpenCV >= 5.0 |

Goal
----

In this tutorial you will:

-   Load and save point cloud data
-   Visualise your data

Requirements
------------

For visualisations you need to compile OpenCV library with OpenGL support.
For this you should set WITH_OPENGL flag ON in CMake while building OpenCV from source.

Practice
-------

Loading and saving of point cloud can be done using `cv::loadPointCloud` and `cv::savePointCloud` accordingly.

Currently supported formats are:
- [.OBJ](https://en.wikipedia.org/wiki/Wavefront_.obj_file) (supported keys are v(which is responsible for point position), vn(normal coordinates) and f(faces of a mesh), other keys are ignored)
- [.PLY](https://en.wikipedia.org/wiki/PLY_(file_format)) (all encoding types(ascii and byte) are supported with limitation to only float type for data)

@code{.py}
vertices, normals = cv2.loadPointCloud("teapot.obj")
@endcode

Function `cv::loadPointCloud` returns vector of points of float (`cv::Point3f`) and vector of their normals(if specified in source file).
To visualize it you can use functions from viz3d module and it is needed to reinterpret data into another format

@code{.py}
vertices = np.squeeze(vertices, axis=1)

color = [1.0, 1.0, 0.0]
colors = np.tile(color, (vertices.shape[0], 1))
obj_pts = np.concatenate((vertices, colors), axis=1).astype(np.float32)

cv2.viz3d.showPoints("Window", "Points", obj_pts)

cv2.waitKey(0)
@endcode

In presented code sample we add a colour attribute to every point
Result will be:

![](tutorial_point_cloud_teapot.jpg)

For additional info grid can be added

@code{.py}
vertices, normals = cv2.loadPointCloud("teapot.obj")
@endcode

![](teapot_grid.jpg)

Other possible way to draw 3d objects can be a mesh.
For that we use special functions to load mesh data and display it.
Here for now only .OBJ files are supported and they should be triangulated before processing (triangulation - process of breaking faces into triangles).

@code{.py}
vertices, indices = cv2.loadMesh("../data/teapot.obj")
vertices = np.squeeze(vertices, axis=1)

cv2.viz3d.showMesh("window", "mesh", vertices, indices)
@endcode

![](teapot_mesh.jpg)
