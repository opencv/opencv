.. _iOS-Installation:

Installation in iOS
*******************

Required Packages
=================

  * CMake 2.8.8 or higher
  * Xcode 4.2 or higher

Getting the Cutting-edge OpenCV from Git Repository
---------------------------------------------------

Launch GIT client and clone OpenCV repository from `here <http://github.com/itseez/opencv>`_

In MacOS it can be done using the following command in Terminal:

.. code-block:: bash

   cd ~/<my_working _directory>
   git clone https://github.com/Itseez/opencv.git


Building OpenCV from Source, using CMake and Command Line
=========================================================

#. Make symbolic link for Xcode to let OpenCV build scripts find the compiler, header files etc.

    .. code-block:: bash

       cd /
       sudo ln -s /Applications/Xcode.app/Contents/Developer Developer

#. Build OpenCV framework:

    .. code-block:: bash

       cd ~/<my_working_directory>
       python opencv/platforms/ios/build_framework.py ios

If everything's fine, a few minutes later you will get ~/<my_working_directory>/ios/opencv2.framework. You can add this framework to your Xcode projects.

Further Reading
===============
You can find several OpenCV+iOS tutorials here :ref:`Table-Of-Content-iOS`.
