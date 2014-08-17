Reading and Writing Images
==========================

.. highlight:: cpp

imdecode
--------
Reads an image from a buffer in memory.

.. ocv:function:: Mat imdecode( InputArray buf,  int flags )

.. ocv:function:: Mat imdecode( InputArray buf,  int flags, Mat* dst )

.. ocv:cfunction:: IplImage* cvDecodeImage( const CvMat* buf, int iscolor=CV_LOAD_IMAGE_COLOR)

.. ocv:cfunction:: CvMat* cvDecodeImageM( const CvMat* buf, int iscolor=CV_LOAD_IMAGE_COLOR)

.. ocv:pyfunction:: cv2.imdecode(buf, flags) -> retval

    :param buf: Input array or vector of bytes.

    :param flags: The same flags as in :ocv:func:`imread` .

    :param dst: The optional output placeholder for the decoded matrix. It can save the image reallocations when the function is called repeatedly for images of the same size.

The function reads an image from the specified buffer in the memory.
If the buffer is too short or contains invalid data, the empty matrix/image is returned.

See
:ocv:func:`imread` for the list of supported formats and flags description.

.. note:: In the case of color images, the decoded images will have the channels stored in ``B G R`` order.

imencode
--------
Encodes an image into a memory buffer.

.. ocv:function:: bool imencode( const String& ext, InputArray img, vector<uchar>& buf, const vector<int>& params=vector<int>())

.. ocv:cfunction:: CvMat* cvEncodeImage( const char* ext, const CvArr* image, const int* params=0 )

.. ocv:pyfunction:: cv2.imencode(ext, img[, params]) -> retval, buf

    :param ext: File extension that defines the output format.

    :param img: Image to be written.

    :param buf: Output buffer resized to fit the compressed image.

    :param params: Format-specific parameters. See  :ocv:func:`imwrite` .

The function compresses the image and stores it in the memory buffer that is resized to fit the result.
See
:ocv:func:`imwrite` for the list of supported formats and flags description.

.. note:: ``cvEncodeImage`` returns single-row matrix of type ``CV_8UC1`` that contains encoded image as array of bytes.

imread
------
Loads an image from a file.

.. ocv:function:: Mat imread( const String& filename, int flags=IMREAD_COLOR )

.. ocv:pyfunction:: cv2.imread(filename[, flags]) -> retval

.. ocv:cfunction:: IplImage* cvLoadImage( const char* filename, int iscolor=CV_LOAD_IMAGE_COLOR )

.. ocv:cfunction:: CvMat* cvLoadImageM( const char* filename, int iscolor=CV_LOAD_IMAGE_COLOR )

    :param filename: Name of file to be loaded.

    :param flags: Flags specifying the color type of a loaded image:

        * CV_LOAD_IMAGE_ANYDEPTH - If set, return 16-bit/32-bit image when the input has the corresponding depth, otherwise convert it to 8-bit.

        * CV_LOAD_IMAGE_COLOR - If set, always convert image to the color one

        * CV_LOAD_IMAGE_GRAYSCALE - If set, always convert image to the grayscale one

        * **>0**  Return a 3-channel color image.
            .. note:: In the current implementation the alpha channel, if any, is stripped from the output image. Use negative value if you need the alpha channel.

        * **=0**  Return a grayscale image.

        * **<0**  Return the loaded image as is (with alpha channel).

The function ``imread`` loads an image from the specified file and returns it. If the image cannot be read (because of missing file, improper permissions, unsupported or invalid format), the function returns an empty matrix ( ``Mat::data==NULL`` ). Currently, the following file formats are supported:

 * Windows bitmaps - ``*.bmp, *.dib`` (always supported)

 * JPEG files - ``*.jpeg, *.jpg, *.jpe`` (see the *Notes* section)

 * JPEG 2000 files - ``*.jp2`` (see the *Notes* section)

 * Portable Network Graphics - ``*.png`` (see the *Notes* section)

 * WebP - ``*.webp`` (see the *Notes* section)

 * Portable image format - ``*.pbm, *.pgm, *.ppm``     (always supported)

 * Sun rasters - ``*.sr, *.ras``     (always supported)

 * TIFF files - ``*.tiff, *.tif`` (see the *Notes* section)

.. note::

    * The function determines the type of an image by the content, not by the file extension.

    * On Microsoft Windows* OS and MacOSX*, the codecs shipped with an OpenCV image (libjpeg, libpng, libtiff, and libjasper) are used by default. So, OpenCV can always read JPEGs, PNGs, and TIFFs. On MacOSX, there is also an option to use native MacOSX image readers. But beware that currently these native image loaders give images with different pixel values because of the color management embedded into MacOSX.

    * On Linux*, BSD flavors and other Unix-like open-source operating systems, OpenCV looks for codecs supplied with an OS image. Install the relevant packages (do not forget the development files, for example, "libjpeg-dev", in Debian* and Ubuntu*) to get the codec support or turn on the ``OPENCV_BUILD_3RDPARTY_LIBS`` flag in CMake.

.. note:: In the case of color images, the decoded images will have the channels stored in ``B G R`` order.

imwrite
-----------
Saves an image to a specified file.

.. ocv:function:: bool imwrite( const String& filename, InputArray img, const vector<int>& params=vector<int>() )

.. ocv:pyfunction:: cv2.imwrite(filename, img[, params]) -> retval

.. ocv:cfunction:: int cvSaveImage( const char* filename, const CvArr* image, const int* params=0 )

    :param filename: Name of the file.

    :param image: Image to be saved.

    :param params: Format-specific save parameters encoded as pairs  ``paramId_1, paramValue_1, paramId_2, paramValue_2, ...`` . The following parameters are currently supported:

        *  For JPEG, it can be a quality ( ``CV_IMWRITE_JPEG_QUALITY`` ) from 0 to 100 (the higher is the better). Default value is 95.

        *  For WEBP, it can be a quality ( CV_IMWRITE_WEBP_QUALITY ) from 1 to 100 (the higher is the better).
           By default (without any parameter) and for quality above 100 the lossless compression is used.

        *  For PNG, it can be the compression level ( ``CV_IMWRITE_PNG_COMPRESSION`` ) from 0 to 9. A higher value means a smaller size and longer compression time. Default value is 3.

        *  For PPM, PGM, or PBM, it can be a binary format flag ( ``CV_IMWRITE_PXM_BINARY`` ), 0 or 1. Default value is 1.

The function ``imwrite`` saves the image to the specified file. The image format is chosen based on the ``filename`` extension (see
:ocv:func:`imread` for the list of extensions). Only 8-bit (or 16-bit unsigned (``CV_16U``) in case of PNG, JPEG 2000, and TIFF) single-channel or 3-channel (with 'BGR' channel order) images can be saved using this function. If the format, depth or channel order is different, use
:ocv:func:`Mat::convertTo` , and
:ocv:func:`cvtColor` to convert it before saving. Or, use the universal :ocv:class:`FileStorage` I/O functions to save the image to XML or YAML format.

It is possible to store PNG images with an alpha channel using this function. To do this, create 8-bit (or 16-bit) 4-channel image BGRA, where the alpha channel goes last. Fully transparent pixels should have alpha set to 0, fully opaque pixels should have alpha set to 255/65535. The sample below shows how to create such a BGRA image and store to PNG file. It also demonstrates how to set custom compression parameters ::

    #include <vector>
    #include <stdio.h>
    #include <opencv2/opencv.hpp>

    using namespace cv;
    using namespace std;

    void createAlphaMat(Mat &mat)
    {
        for (int i = 0; i < mat.rows; ++i) {
            for (int j = 0; j < mat.cols; ++j) {
                Vec4b& rgba = mat.at<Vec4b>(i, j);
                rgba[0] = UCHAR_MAX;
                rgba[1] = saturate_cast<uchar>((float (mat.cols - j)) / ((float)mat.cols) * UCHAR_MAX);
                rgba[2] = saturate_cast<uchar>((float (mat.rows - i)) / ((float)mat.rows) * UCHAR_MAX);
                rgba[3] = saturate_cast<uchar>(0.5 * (rgba[1] + rgba[2]));
            }
        }
    }

    int main(int argv, char **argc)
    {
        // Create mat with alpha channel
        Mat mat(480, 640, CV_8UC4);
        createAlphaMat(mat);

        vector<int> compression_params;
        compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
        compression_params.push_back(9);

        try {
            imwrite("alpha.png", mat, compression_params);
        }
        catch (runtime_error& ex) {
            fprintf(stderr, "Exception converting image to PNG format: %s\n", ex.what());
            return 1;
        }

        fprintf(stdout, "Saved PNG file with alpha data.\n");
        return 0;
    }
