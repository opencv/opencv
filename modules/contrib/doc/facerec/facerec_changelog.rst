Changelog
=========

Release 0.05
------------

This library is now included in the official OpenCV distribution (from 2.4 on).
The :ocv:class`FaceRecognizer` is now an :ocv:class:`Algorithm`, which better fits into the overall
OpenCV API.

To reduce the confusion on user side and minimize my work, libfacerec and OpenCV
have been synchronized and are now based on the same interfaces and implementation.

The library now has an extensive documentation:

* The API is explained in detail and with a lot of code examples.
* The face recognition guide I had written for Python and GNU Octave/MATLAB has been adapted to the new OpenCV C++ ``cv::FaceRecognizer``.
* A tutorial for gender classification with Fisherfaces.
* A tutorial for face recognition in videos (e.g. webcam).


Release highlights
++++++++++++++++++

* There are no single highlights to pick from, this release is a highlight itself.

Release 0.04
------------

This version is fully Windows-compatible and works with OpenCV 2.3.1. Several
bugfixes, but none influenced the recognition rate.

Release highlights
++++++++++++++++++

* A whole lot of exceptions with meaningful error messages.
* A tutorial for Windows users: `http://bytefish.de/blog/opencv_visual_studio_and_libfacerec <http://bytefish.de/blog/opencv_visual_studio_and_libfacerec>`_


Release 0.03
------------

Reworked the library to provide separate implementations in cpp files, because
it's the preferred way of contributing OpenCV libraries. This means the library
is not header-only anymore. Slight API changes were done, please see the
documentation for details.

Release highlights
++++++++++++++++++

* New Unit Tests (for LBP Histograms) make the library more robust.
* Added more documentation.


Release 0.02
------------

Reworked the library to provide separate implementations in cpp files, because
it's the preferred way of contributing OpenCV libraries. This means the library
is not header-only anymore. Slight API changes were done, please see the
documentation for details.

Release highlights
++++++++++++++++++

* New Unit Tests (for LBP Histograms) make the library more robust.
* Added a documentation and changelog in reStructuredText.

Release 0.01
------------

Initial release as header-only library.

Release highlights
++++++++++++++++++

* Colormaps for OpenCV to enhance the visualization.
* Face Recognition algorithms implemented:

  * Eigenfaces [TP91]_
  * Fisherfaces [BHK97]_
  * Local Binary Patterns Histograms [AHP04]_

* Added persistence facilities to store the models with a common API.
* Unit Tests (using `gtest <http://code.google.com/p/googletest/>`_).
* Providing a CMakeLists.txt to enable easy cross-platform building.
