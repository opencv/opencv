Getting Started with Videos {#tutorial_py_video_display}
===========================

Goal
----

-   Learn to read video, display video and save video.
-   Learn to capture from Camera and display it.
-   You will learn these functions : **cv2.VideoCapture()**, **cv2.VideoWriter()**

Capture Video from Camera
-------------------------

Often, we have to capture live stream with camera. OpenCV provides a very simple interface to this.
Let's capture a video from the camera (I am using the in-built webcam of my laptop), convert it into
grayscale video and display it. Just a simple task to get started.

To capture a video, you need to create a **VideoCapture** object. Its argument can be either the
device index or the name of a video file. Device index is just the number to specify which camera.
Normally one camera will be connected (as in my case). So I simply pass 0 (or -1). You can select
the second camera by passing 1 and so on. After that, you can capture frame-by-frame. But at the
end, don't forget to release the capture.
@code{.py}
import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
@endcode
cap.read() returns a bool (True/False). If frame is read correctly, it will be True. So you can
check end of the video by checking this return value.

Sometimes, cap may not have initialized the capture. In that case, this code shows error. You can
check whether it is initialized or not by the method **cap.isOpened()**. If it is True, OK.
Otherwise open it using **cap.open()**.

You can also access some of the features of this video using **cap.get(propId)** method where propId
is a number from 0 to 18. Each number denotes a property of the video (if it is applicable to that
video) and full details can be seen here: cv::VideoCapture::get() .
Some of these values can be modified using **cap.set(propId, value)**. Value is the new value you
want.

For example, I can check the frame width and height by cap.get(3) and cap.get(4). It gives me
640x480 by default. But I want to modify it to 320x240. Just use ret = cap.set(3,320) and
ret = cap.set(4,240).

@note If you are getting error, make sure camera is working fine using any other camera application
(like Cheese in Linux).

Playing Video from file
-----------------------

It is same as capturing from Camera, just change camera index with video file name. Also while
displaying the frame, use appropriate time for cv2.waitKey(). If it is too less, video will be very
fast and if it is too high, video will be slow (Well, that is how you can display videos in slow
motion). 25 milliseconds will be OK in normal cases.
@code{.py}
import numpy as np
import cv2

cap = cv2.VideoCapture('vtest.avi')

while(cap.isOpened()):
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
@endcode

@note Make sure proper versions of ffmpeg or gstreamer is installed. Sometimes, it is a headache to
work with Video Capture mostly due to wrong installation of ffmpeg/gstreamer.

Saving a Video
--------------

So we capture a video, process it frame-by-frame and we want to save that video. For images, it is
very simple, just use cv2.imwrite(). Here a little more work is required.

This time we create a **VideoWriter** object. We should specify the output file name (eg:
output.avi). Then we should specify the **FourCC** code (details in next paragraph). Then number of
frames per second (fps) and frame size should be passed. And last one is **isColor** flag. If it is
True, encoder expect color frame, otherwise it works with grayscale frame.

[FourCC](http://en.wikipedia.org/wiki/FourCC) is a 4-byte code used to specify the video codec. The
list of available codes can be found in [fourcc.org](http://www.fourcc.org/codecs.php). It is
platform dependent. Following codecs works fine for me.

-   In Fedora: DIVX, XVID, MJPG, X264, WMV1, WMV2. (XVID is more preferable. MJPG results in high
    size video. X264 gives very small size video)
-   In Windows: DIVX (More to be tested and added)
-   In OSX : *(I don't have access to OSX. Can some one fill this?)*

FourCC code is passed as cv2.VideoWriter_fourcc('M','J','P','G') or
cv2.VideoWriter_fourcc(\*'MJPG') for MJPG.

Below code capture from a Camera, flip every frame in vertical direction and saves it.
@code{.py}
import numpy as np
import cv2

cap = cv2.VideoCapture(0)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        frame = cv2.flip(frame,0)

        # write the flipped frame
        out.write(frame)

        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
@endcode

Additional Resources
--------------------

Exercises
---------
