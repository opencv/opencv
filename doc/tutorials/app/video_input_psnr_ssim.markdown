Video Input with OpenCV and similarity measurement {#tutorial_video_input_psnr_ssim}
==================================================

@tableofcontents

@prev_tutorial{tutorial_raster_io_gdal}
@next_tutorial{tutorial_video_write}

|    |    |
| -: | :- |
| Original author | Bernát Gábor |
| Compatibility | OpenCV >= 3.0 |

Goal
----

Today it is common to have a digital video recording system at your disposal. Therefore, you will
eventually come to the situation that you no longer process a batch of images, but video streams.
These may be of two kinds: real-time image feed (in the case of a webcam) or prerecorded and hard
disk drive stored files. Luckily OpenCV treats these two in the same manner, with the same C++
class. So here's what you'll learn in this tutorial:

-   How to open and read video streams
-   Two ways for checking image similarity: PSNR and SSIM

The source code
---------------

As a test case where to show off these using OpenCV I've created a small program that reads in two
video files and performs a similarity check between them. This is something you could use to check
just how well a new video compressing algorithms works. Let there be a reference (original) video
like [this small Megamind clip
](https://github.com/opencv/opencv/tree/4.x/samples/data/Megamind.avi) and [a compressed
version of it ](https://github.com/opencv/opencv/tree/4.x/samples/data/Megamind_bugy.avi).
You may also find the source code and these video file in the
`samples/data` folder of the OpenCV source library.

@add_toggle_cpp
@include cpp/tutorial_code/videoio/video-input-psnr-ssim/video-input-psnr-ssim.cpp
@end_toggle

@add_toggle_python
@include samples/python/tutorial_code/videoio/video-input-psnr-ssim.py
@end_toggle

How to read a video stream (online-camera or offline-file)?
-----------------------------------------------------------

Essentially, all the functionalities required for video manipulation is integrated in the @ref cv::VideoCapture
C++ class. This on itself builds on the FFmpeg open source library. This is a basic
dependency of OpenCV so you shouldn't need to worry about this. A video is composed of a succession
of images, we refer to these in the literature as frames. In case of a video file there is a *frame
rate* specifying just how long is between two frames. While for the video cameras usually there is a
limit of just how many frames they can digitize per second, this property is less important as at
any time the camera sees the current snapshot of the world.

The first task you need to do is to assign to a @ref cv::VideoCapture class its source. You can do
this either via the @ref cv::VideoCapture::VideoCapture or its @ref cv::VideoCapture::open function. If this argument is an
integer then you will bind the class to a camera, a device. The number passed here is the ID of the
device, assigned by the operating system. If you have a single camera attached to your system its ID
will probably be zero and further ones increasing from there. If the parameter passed to these is a
string it will refer to a video file, and the string points to the location and name of the file.
For example, to the upper source code a valid command line is:
@code{.bash}
video/Megamind.avi video/Megamind_bug.avi  35 10
@endcode
We do a similarity check. This requires a reference and a test case video file. The first two
arguments refer to this. Here we use a relative address. This means that the application will look
into its current working directory and open the video folder and try to find inside this the
*Megamind.avi* and the *Megamind_bug.avi*.
@code{.cpp}
const string sourceReference = argv[1],sourceCompareWith = argv[2];

VideoCapture captRefrnc(sourceReference);
// or
VideoCapture captUndTst;
captUndTst.open(sourceCompareWith);
@endcode
To check if the binding of the class to a video source was successful or not use the @ref cv::VideoCapture::isOpened
function:
@code{.cpp}
if ( !captRefrnc.isOpened())
  {
  cout  << "Could not open reference " << sourceReference << endl;
  return -1;
  }
@endcode
Closing the video is automatic when the objects destructor is called. However, if you want to close
it before this you need to call its @ref cv::VideoCapture::release function. The frames of the video are just
simple images. Therefore, we just need to extract them from the @ref cv::VideoCapture object and put
them inside a *Mat* one. The video streams are sequential. You may get the frames one after another
by the @ref cv::VideoCapture::read or the overloaded \>\> operator:
@code{.cpp}
Mat frameReference, frameUnderTest;
captRefrnc >> frameReference;
captUndTst.read(frameUnderTest);
@endcode
The upper read operations will leave empty the *Mat* objects if no frame could be acquired (either
cause the video stream was closed or you got to the end of the video file). We can check this with a
simple if:
@code{.cpp}
if( frameReference.empty()  || frameUnderTest.empty())
{
 // exit the program
}
@endcode
A read method is made of a frame grab and a decoding applied on that. You may call explicitly these
two by using the @ref cv::VideoCapture::grab and then the @ref cv::VideoCapture::retrieve functions.

Videos have many-many information attached to them besides the content of the frames. These are
usually numbers, however in some case it may be short character sequences (4 bytes or less). Due to
this to acquire these information there is a general function named @ref cv::VideoCapture::get that returns double
values containing these properties. Use bitwise operations to decode the characters from a double
type and conversions where valid values are only integers. Its single argument is the ID of the
queried property. For example, here we get the size of the frames in the reference and test case
video file; plus the number of frames inside the reference.
@code{.cpp}
Size refS = Size((int) captRefrnc.get(CAP_PROP_FRAME_WIDTH),
                 (int) captRefrnc.get(CAP_PROP_FRAME_HEIGHT)),

cout << "Reference frame resolution: Width=" << refS.width << "  Height=" << refS.height
     << " of nr#: " << captRefrnc.get(CAP_PROP_FRAME_COUNT) << endl;
@endcode
When you are working with videos you may often want to control these values yourself. To do this
there is a @ref cv::VideoCapture::set function. Its first argument remains the name of the property you want to
change and there is a second of double type containing the value to be set. It will return true if
it succeeds and false otherwise. Good examples for this is seeking in a video file to a given time
or frame:
@code{.cpp}
captRefrnc.set(CAP_PROP_POS_MSEC, 1.2);  // go to the 1.2 second in the video
captRefrnc.set(CAP_PROP_POS_FRAMES, 10); // go to the 10th frame of the video
// now a read operation would read the frame at the set position
@endcode
For properties you can read and change look into the documentation of the @ref cv::VideoCapture::get and
@ref cv::VideoCapture::set functions.

### Image similarity - PSNR and SSIM

We want to check just how imperceptible our video converting operation went, therefore we need a
system to check frame by frame the similarity or differences. The most common algorithm used for
this is the PSNR (aka **Peak signal-to-noise ratio**). The simplest definition of this starts out
from the *mean squared error*. Let there be two images: I1 and I2; with a two dimensional size i and
j, composed of c number of channels.

\f[MSE = \frac{1}{c*i*j} \sum{(I_1-I_2)^2}\f]

Then the PSNR is expressed as:

\f[PSNR = 10 \cdot \log_{10} \left( \frac{MAX_I^2}{MSE} \right)\f]

Here the \f$MAX_I\f$ is the maximum valid value for a pixel. In case of the simple single byte image
per pixel per channel this is 255. When two images are the same the MSE will give zero, resulting in
an invalid divide by zero operation in the PSNR formula. In this case the PSNR is undefined and as
we'll need to handle this case separately. The transition to a logarithmic scale is made because the
pixel values have a very wide dynamic range. All this translated to OpenCV and a function looks
like:

@add_toggle_cpp
@snippet cpp/tutorial_code/videoio/video-input-psnr-ssim/video-input-psnr-ssim.cpp get-psnr
@end_toggle

@add_toggle_python
@snippet samples/python/tutorial_code/videoio/video-input-psnr-ssim.py get-psnr
@end_toggle

Typically result values are anywhere between 30 and 50 for video compression, where higher is
better. If the images significantly differ you'll get much lower ones like 15 and so. This
similarity check is easy and fast to calculate, however in practice it may turn out somewhat
inconsistent with human eye perception. The **structural similarity** algorithm aims to correct
this.

Describing the methods goes well beyond the purpose of this tutorial. For that I invite you to read
the article introducing it. Nevertheless, you can get a good image of it by looking at the OpenCV
implementation below.

@note
    SSIM is described more in-depth in the: "Z. Wang, A. C. Bovik, H. R. Sheikh and E. P.
    Simoncelli, "Image quality assessment: From error visibility to structural similarity," IEEE
    Transactions on Image Processing, vol. 13, no. 4, pp. 600-612, Apr. 2004." article.

@add_toggle_cpp
@snippet samples/cpp/tutorial_code/videoio/video-input-psnr-ssim/video-input-psnr-ssim.cpp get-mssim
@end_toggle

@add_toggle_python
@snippet samples/python/tutorial_code/videoio/video-input-psnr-ssim.py get-mssim
@end_toggle

This will return a similarity index for each channel of the image. This value is between zero and
one, where one corresponds to perfect fit. Unfortunately, the many Gaussian blurring is quite
costly, so while the PSNR may work in a real time like environment (24 frames per second) this will
take significantly more than to accomplish similar performance results.

Therefore, the source code presented at the start of the tutorial will perform the PSNR measurement
for each frame, and the SSIM only for the frames where the PSNR falls below an input value. For
visualization purpose we show both images in an OpenCV window and print the PSNR and MSSIM values to
the console. Expect to see something like:

![](images/outputVideoInput.png)

You may observe a runtime instance of this on the [YouTube here](https://www.youtube.com/watch?v=iOcNljutOgg).

@youtube{iOcNljutOgg}
