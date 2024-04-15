Aruco module FAQ {#tutorial_aruco_faq}
================

@prev_tutorial{tutorial_aruco_calibration}

This is a compilation of questions that can be useful for those that want to use the aruco module.

- I only want to label some objects, what should I use?

In this case, you only need single ArUco markers. You can place one or several markers with different
ids in each of the object you want to identify.


- Which algorithm is used for marker detection?

The aruco module is based on the original ArUco library. A full description of the detection process
can be found in:

> S. Garrido-Jurado, R. Muñoz-Salinas, F. J. Madrid-Cuevas, and M. J. Marín-Jiménez. 2014.
> "Automatic generation and detection of highly reliable fiducial markers under occlusion".
> Pattern Recogn. 47, 6 (June 2014), 2280-2292. DOI=10.1016/j.patcog.2014.01.005


- My markers are not being detected correctly, what can I do?

There can be many factors that avoid the correct detection of markers. You probably need to adjust
some of the parameters in the `cv::aruco::DetectorParameters` object. The first thing you can do is
checking if your markers are returned as rejected candidates by the `cv::aruco::ArucoDetector::detectMarkers()`
function. Depending on this, you should try to modify different parameters.

If you are using a ArUco board, you can also try the `cv::aruco::ArucoDetector::refineDetectedMarkers()` function.
If you are [using big markers](https://github.com/opencv/opencv_contrib/issues/2811) (400x400 pixels and more), try
increasing `cv::aruco::DetectorParameters::adaptiveThreshWinSizeMax` value.
Also avoid [narrow borders around the ArUco marker](https://github.com/opencv/opencv_contrib/issues/2492)
(5% or less of the marker perimeter, adjusted by `cv::aruco::DetectorParameters::minMarkerDistanceRate`)
around markers.


- What are the benefits of ArUco boards? What are the drawbacks?

Using a board of markers you can obtain the camera pose from a set of markers, instead of a single one.
This way, the detection is able to handle occlusion of partial views of the Board, since only one
marker is necessary to obtain the pose.

Furthermore, as in most cases you are using more corners for pose estimation, it will be more
accurate than using a single marker.

The main drawback is that a Board is not as versatile as a single marker.



- What are the benefits of ChArUco boards over ArUco boards? And the drawbacks?

ChArUco boards combines chessboards with ArUco boards. Thanks to this, the corners provided by
ChArUco boards are more accurate than those provided by ArUco Boards (or single markers).

The main drawback is that ChArUco boards are not as versatile as ArUco board. For instance,
a ChArUco board is a planar board with a specific marker layout while the ArUco boards can have
any layout, even in 3d. Furthermore, the markers in the ChArUco board are usually smaller and
more difficult to detect.


- I do not need pose estimation, should I use ChArUco boards?

No. The main goal of ChArUco boards is provide high accurate corners for pose estimation or camera
calibration.


- Should all the markers in an ArUco board be placed in the same plane?

No, the marker corners in a ArUco board can be placed anywhere in its 3d coordinate system.


- Should all the markers in an ChArUco board be placed in the same plane?

Yes, all the markers in a ChArUco board need to be in the same plane and their layout is fixed by
the chessboard shape.


- What is the difference between a `cv::aruco::Board` object and a `cv::aruco::GridBoard` object?

The `cv::aruco::GridBoard` class is a specific type of board that inherits from `cv::aruco::Board` class.
A `cv::aruco::GridBoard` object is a board whose markers are placed in the same plane and in a grid layout.


- What are Diamond markers?

Diamond markers are very similar to a ChArUco board of 3x3 squares. However, contrary to ChArUco boards,
the detection of diamonds is based on the relative position of the markers.
They are useful when you want to provide a conceptual meaning to any (or all) of the markers in
the diamond. An example is using one of the marker to provide the diamond scale.


- Do I need to detect marker before board detection, ChArUco board detection or Diamond detection?

Yes, the detection of single markers is a basic tool in the aruco module. It is done using the
`cv::aruco::DetectorParameters::detectMarkers()` function. The rest of functionalities receives
a list of detected markers from this function.


- I want to calibrate my camera, can I use this module?

Yes, the aruco module provides functionalities to calibrate the camera using both, ArUco boards and
ChArUco boards.


- Should I calibrate using a ChArUco board or an ArUco board?

It is highly recommended the calibration using ChArUco board due to the high accuracy.


- Should I use a predefined dictionary or generate my own dictionary?

In general, it is easier to use one of the predefined dictionaries. However, if you need a bigger
dictionary (in terms of number of markers or number of bits) you should generate your own dictionary.
Dictionary generation is also useful if you want to maximize the inter-marker distance to achieve
a better error correction during the identification step.

- I am generating my own dictionary but it takes too long

Dictionary generation should only be done once at the beginning of your application and it should take
some seconds. If you are generating the dictionary on each iteration of your detection loop, you are
doing it wrong.

Furthermore, it is recommendable to save the dictionary to a file with `cv::aruco::Dictionary::writeDictionary()`
and read it with `cv::aruco::Dictionary::readDictionary()` on every execution, so you don't need
to generate it.


- I would like to use some markers of the original ArUco library that I have already printed, can I use them?

Yes, one of the predefined dictionary is `cv::aruco::DICT_ARUCO_ORIGINAL`, which detects the marker
of the original ArUco library with the same identifiers.


- Can I use the Board configuration file of the original ArUco library in this module?

Not directly, you will need to adapt the information of the ArUco file to the aruco module Board format.


- Can I use this module to detect the markers of other libraries based on binary fiducial markers?

Probably yes, however you will need to port the dictionary of the original library to the aruco module format.


- Do I need to store the Dictionary information in a file so I can use it in different executions?

If you are using one of the predefined dictionaries, it is not necessary. Otherwise, it is recommendable
that you save it to file.


- Do I need to store the Board information in a file so I can use it in different executions?

If you are using a `cv::aruco::GridBoard` or a `cv::aruco::CharucoBoard` you only need to store
the board measurements that are provided to the `cv::aruco::GridBoard::GridBoard()` constructor or
in or `cv::aruco::CharucoBoard` constructor. If you manually modify the marker ids of the boards,
or if you use a different type of board, you should save your board object to file.

- Does the aruco module provide functions to save the Dictionary or Board to file?

You can use `cv::aruco::Dictionary::writeDictionary()` and `cv::aruco::Dictionary::readDictionary()`
for `cv::aruco::Dictionary`. The data member of board classes are public and can be easily stored.


- Alright, but how can I render a 3d model to create an augmented reality application?

To do so, you will need to use an external rendering engine library, such as OpenGL. The aruco module
only provides the functionality to obtain the camera pose, i.e. the rotation and traslation vectors,
which is necessary to create the augmented reality effect. However, you will need to adapt the rotation
and traslation vectors from the OpenCV format to the format accepted by your 3d rendering library.
The original ArUco library contains examples of how to do it for OpenGL and Ogre3D.


- I have use this module in my research work, how can I cite it?

You can cite the original ArUco library:

> S. Garrido-Jurado, R. Muñoz-Salinas, F. J. Madrid-Cuevas, and M. J. Marín-Jiménez. 2014.
> "Automatic generation and detection of highly reliable fiducial markers under occlusion".
> Pattern Recogn. 47, 6 (June 2014), 2280-2292. DOI=10.1016/j.patcog.2014.01.005

- Pose estimation markers are not being detected correctly, what can I do?

It is important to remark that the estimation of the pose using only 4 coplanar points is subject to ambiguity.
In general, the ambiguity can be solved, if the camera is near to the marker.
However, as the marker becomes small, the errors in the corner estimation grows and ambiguity comes
as a problem. Try increasing the size of the marker you're using, and you can also try non-symmetrical
(aruco_dict_utils.cpp) markers to avoid collisions. Use multiple markers (ArUco/ChArUco/Diamonds boards)
and pose estimation with solvePnP() with the `cv::SOLVEPNP_IPPE_SQUARE` option.
More in [this issue](https://github.com/opencv/opencv/issues/8813).
