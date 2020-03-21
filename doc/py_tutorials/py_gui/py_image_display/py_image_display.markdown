Getting Started with Images {#tutorial_py_image_display}
===========================

Goals
-----

-   Here, you will learn how to read an image, how to display it, and how to save it back
-   You will learn these functions : **cv.imread()**, **cv.imshow()** , **cv.imwrite()**
-   Optionally, you will learn how to display images with Matplotlib

Using OpenCV
------------

### Read an image

Use the function **cv.imread()** to read an image. The image should be in the working directory or
a full path of image should be given.

Second argument is a flag which specifies the way image should be read.

-   cv.IMREAD_COLOR : Loads a color image. Any transparency of image will be neglected. It is the
    default flag.
-   cv.IMREAD_GRAYSCALE : Loads image in grayscale mode
-   cv.IMREAD_UNCHANGED : Loads image as such including alpha channel

@note Instead of these three flags, you can simply pass integers 1, 0 or -1 respectively.

See the code below:
@code{.py}
import numpy as np
import cv2 as cv

# Load a color image in grayscale
img = cv.imread('messi5.jpg',0)
@endcode

**warning**

Even if the image path is wrong, it won't throw any error, but `print img` will give you `None`

### Display an image

Use the function **cv.imshow()** to display an image in a window. The window automatically fits to
the image size.

First argument is a window name which is a string. Second argument is our image. You can create as
many windows as you wish, but with different window names.
@code{.py}
cv.imshow('image',img)
cv.waitKey(0)
cv.destroyAllWindows()
@endcode
A screenshot of the window will look like this (in Fedora-Gnome machine):

![image](images/opencv_screenshot.jpg)

**cv.waitKey()** is a keyboard binding function. Its argument is the time in milliseconds. The
function waits for specified milliseconds for any keyboard event. If you press any key in that time,
the program continues. If **0** is passed, it waits indefinitely for a key stroke. It can also be
set to detect specific key strokes like, if key a is pressed etc which we will discuss below.

@note Besides binding keyboard events this function also processes many other GUI events, so you
MUST use it to actually display the image.

**cv.destroyAllWindows()** simply destroys all the windows we created. If you want to destroy any
specific window, use the function **cv.destroyWindow()** where you pass the exact window name as
the argument.

@note There is a special case where you can create an empty window and load an image to it later. In
that case, you can specify whether the window is resizable or not. It is done with the function
**cv.namedWindow()**. By default, the flag is cv.WINDOW_AUTOSIZE. But if you specify the flag to be
cv.WINDOW_NORMAL, you can resize window. It will be helpful when an image is too large in dimension
and when adding track bars to windows.

See the code below:
@code{.py}
cv.namedWindow('image', cv.WINDOW_NORMAL)
cv.imshow('image',img)
cv.waitKey(0)
cv.destroyAllWindows()
@endcode
### Write an image

Use the function **cv.imwrite()** to save an image.

First argument is the file name, second argument is the image you want to save.
@code{.py}
cv.imwrite('messigray.png',img)
@endcode
This will save the image in PNG format in the working directory.

### Sum it up

Below program loads an image in grayscale, displays it, saves the image if you press 's' and exit, or
simply exits without saving if you press ESC key.
@code{.py}
import numpy as np
import cv2 as cv

img = cv.imread('messi5.jpg',0)
cv.imshow('image',img)
k = cv.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv.destroyAllWindows()
elif k == ord('s'): # wait for 's' key to save and exit
    cv.imwrite('messigray.png',img)
    cv.destroyAllWindows()
@endcode

**warning**

If you are using a 64-bit machine, you will have to modify `k = cv.waitKey(0)` line as follows :
`k = cv.waitKey(0) & 0xFF`

Using Matplotlib
----------------

Matplotlib is a plotting library for Python which gives you wide variety of plotting methods. You
will see them in coming articles. Here, you will learn how to display image with Matplotlib. You can
zoom images, save them, etc, using Matplotlib.
@code{.py}
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('messi5.jpg',0)
plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()
@endcode
A screen-shot of the window will look like this :

![image](images/matplotlib_screenshot.jpg)

@note Plenty of plotting options are available in Matplotlib. Please refer to Matplotlib docs for more
details. Some, we will see on the way.

__warning__

Color image loaded by OpenCV is in BGR mode. But Matplotlib displays in RGB mode. So color images
will not be displayed correctly in Matplotlib if image is read with OpenCV. Please see the exercises
for more details.

Additional Resources
--------------------

-#  [Matplotlib Plotting Styles and Features](http://matplotlib.org/api/pyplot_api.html)

Exercises
---------

-#  There is some problem when you try to load color image in OpenCV and display it in Matplotlib.
    Read [this discussion](http://stackoverflow.com/a/15074748/1134940) and understand it.
