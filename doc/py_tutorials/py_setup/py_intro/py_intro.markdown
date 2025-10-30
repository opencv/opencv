Introduction to OpenCV-Python Tutorials {#tutorial_py_intro}
=======================================

OpenCV
------

OpenCV was started at Intel in 1999 by **Gary Bradsky**, and the first release came out in 2000.
**Vadim Pisarevsky** joined Gary Bradsky to manage Intel's Russian software OpenCV team. In 2005,
OpenCV was used on Stanley, the vehicle that won the 2005 DARPA Grand Challenge. Later, its active
development continued under the support of Willow Garage with Gary Bradsky and Vadim Pisarevsky
leading the project. OpenCV now supports a multitude of algorithms related to Computer Vision and
Machine Learning and is expanding day by day.

OpenCV supports a wide variety of programming languages such as C++, Python, Java, etc., and is
available on different platforms including Windows, Linux, OS X, Android, and iOS. Interfaces for
high-speed GPU operations based on CUDA and OpenCL are also under active development.

OpenCV-Python is the Python API for OpenCV, combining the best qualities of the OpenCV C++ API and
the Python language.

OpenCV-Python
-------------

OpenCV-Python is a library of Python bindings designed to solve computer vision problems.

Python is a general purpose programming language started by **Guido van Rossum** that became very
popular very quickly, mainly because of its simplicity and code readability. It enables the
programmer to express ideas in fewer lines of code without reducing readability.

Compared to languages like C/C++, Python is slower. That said, Python can be easily extended with
C/C++, which allows us to write computationally intensive code in C/C++ and create Python wrappers
that can be used as Python modules. This gives us two advantages: first, the code is as fast as the
original C/C++ code (since it is the actual C++ code working in background) and second, it is easier to
code in Python than C/C++. OpenCV-Python is a Python wrapper for the original OpenCV C++
implementation.

OpenCV-Python makes use of **Numpy**, which is a highly optimized library for numerical operations
with a MATLAB-style syntax. All the OpenCV array structures are converted to and from Numpy arrays.
This also makes it easier to integrate with other libraries that use Numpy such as SciPy and
Matplotlib.

OpenCV-Python Tutorials
-----------------------

OpenCV introduces a new set of tutorials which will guide you through various functions available in
OpenCV-Python. **This guide is mainly focused on OpenCV 3.x version** (although most of the
tutorials will also work with OpenCV 2.x).

Prior knowledge of Python and Numpy is recommended as they won't be covered in this guide.
**Proficiency with Numpy is a must in order to write optimized code using OpenCV-Python.**

This tutorial was originally started by *Abid Rahman K.* as part of the Google Summer of Code 2013
program under the guidance of *Alexander Mordvintsev*.

OpenCV Needs You !!!
--------------------

Since OpenCV is an open source initiative, all are welcome to make contributions to the library,
documentation, and tutorials. If you find any mistake in this tutorial (from a small spelling
mistake to an egregious error in code or concept), feel free to correct it by cloning OpenCV in
[GitHub](https://github.com/opencv/opencv) and submitting a pull request. OpenCV developers will
check your pull request, give you important feedback and (once it passes the approval of the
reviewer) it will be merged into OpenCV. You will then become an open source contributor :-)

As new modules are added to OpenCV-Python, this tutorial will have to be expanded. If you are
familiar with a particular algorithm and can write up a tutorial including basic theory of the
algorithm and code showing example usage, please do so.

Remember,**together** we can make this project a great success !!!

Contributors
------------

Below is the list of contributors who submitted tutorials to OpenCV-Python.

-#  Alexander Mordvintsev (GSoC-2013 mentor)
2.  Abid Rahman K. (GSoC-2013 intern)

Additional Resources
--------------------

-#  A Quick guide to Python - [A Byte of Python](https://python.swaroopch.com/)
1.  [A Quick guide to Python](https://www.freecodecamp.org/news/the-python-guide-for-beginners/)
2.  [NumPy Quickstart tutorial](https://numpy.org/doc/stable/user/quickstart.html)
3.  [NumPy Reference](https://numpy.org/doc/stable/reference/index.html)
4.  [OpenCV Documentation](https://docs.opencv.org/)
5.  [OpenCV Forum](https://forum.opencv.org/)
