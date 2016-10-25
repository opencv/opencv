Using OpenCV with biicode dependency manager {#tutorial_biicode}
============================================

Goals
-----
In this tutorial you will learn how to:

  * Get started with OpenCV using biicode.
  * Develop your own application in OpenCV with biicode.
  * Switching between OpenCV versions.

What is biicode?
----------------

![](images/biicode.png)
[biicode](http://opencv.org/biicode.html) resolves and keeps track of dependencies and version compatibilities in C/C++ projects.
Using biicode *hooks feature*, **getting started with OpenCV in C++ and C** is pretty straight-forward. **Just write an include to OpenCV headers** and biicode will retrieve and install OpenCV in your computer and configure your project.

Prerequisites
-------------

  * biicode. Here is a [link to install it at any OS](http://www.biicode.com/downloads).
  * Windows users: Any Visual Studio version (Visual Studio 12 preferred).

Explanation
-----------

### Example: Detect faces in images using the Objdetect module from OpenCV

Once biicode is installed, execute in your terminal/console:

@code{.bash}
$ bii init mycvproject
$ cd mycvproject
$ bii open diego/opencvex
@endcode

Windows users also execute:

@code{.bash}
$ bii cpp:configure -G "Visual Studio 12"
@endcode

Now execute ``bii cpp:build`` to build the project. @note This can take a while, until it downloads and builds OpenCV. However, this is downloaded just once in your machine to your "user/.biicode" folder. If the OpenCV installation process fails, you might simply go there, delete OpenCV files inside "user/.biicode" and repeat.

@code{.bash}
$ bii cpp:build
@endcode

Find your binaries in the bin folder:

@code{.bash}
$ cd bin
$ ./diego_opencvex_main
@endcode

![](images/biiapp.png)

@code{.bash}
$ ./diego_opencvex_mainfaces
@endcode

![](images/bii_lena.png)

###Developing your own application

**biicode works with include headers in your source-code files**, it reads them and retrieves all the dependencies in its database. So it is as simple as typing:

@code{.cpp}
    #include "diego/opencv/opencv/cv.h"
@endcode

in the headers of your ``.cpp`` file.

To start a new project using OpenCV, execute:

@code{.bash}
$ bii init mycvproject
$ cd mycvproject
@endcode

The next line just creates a *myuser/myblock* folder inside "blocks" with a simple "Hello World" *main.cpp* into it. You can also do it manually:

@code{.bash}
$ bii new myuser/myblock --hello=cpp
@endcode

Now replace your *main.cpp* contents inside *blocks/myuser/myblock* with **your app code**.
Put the includes as:

@code{.cpp}
    #include "diego/opencv/opencv/cv.h
@endcode

If you type:

@code{.bash}
$ bii deps
@endcode

You will check that ``opencv/cv.h`` is an "unresolved" dependency. You can find it with:

@code{.bash}
$ bii find
@endcode

Now, you can just `bii cpp:configure` and `bii cpp:build` your project as described above.

**To use regular include directives**, configure them in your **biicode.conf** file. Let your includes be:

@code{.cpp}
    #include "opencv/cv.h"
@endcode

And write in your **biicode.conf**:

@code{.cpp}
    [includes]
        opencv/cv.h: diego/opencv
    [requirements]
        diego/opencv: 0
@endcode

###Switching OpenCV versions

If you want to try or develop your application against **OpenCV 2.4.10** and also against **3.0-beta**, change it in your **biicode.conf** file, simply alternating track in your `[requirements]`:

@code{.cpp}
    [requirements]
        diego/opencv: 0
@endcode

replace with:

@code{.cpp}
    [requirements]
        diego/opencv(beta): 0
@endcode

@note The first time you switch to 3.0-beta, it will also take a while to download and build the 3.0-beta release. From that point on you can change back and forth between versions just by modifying your *biicode.conf requirements*.

Find the hooks and examples:
* [OpenCV 2.4.10](http://www.biicode.com/diego/opencv)
* [OpenCV 3.0 beta](http://www.biicode.com/diego/diego/opencv/beta)
* [objdetect module from OpenCV](@ref tutorial_table_of_content_objdetect)

This is just an example of how can it be done with biicode python hooks. Probably now that CMake files reuse is possible with biicode, it could be better to implement it with CMake, in order to get more control over the build of OpenCV.

Results and conclusion
----------------------

Installing OpenCV with biicode is straight forward for any OS.

Run any example like you just did with *objdetect module* from OpenCV, or develop your own application. It only needs a *biicode.conf* file to get OpenCV library working in your computer.

Switching between OpenCV versions is available too and effortless.

For any doubts or further information regarding biicode, suit yourselves at [Stackoverflow](http://stackoverflow.com/questions/tagged/biicode?sort=newest), biicode’s [forum](http://forum.biicode.com/) or [ask biicode](http://web.biicode.com/contact-us/), we will be glad to help you.