// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// This code is also subject to the license terms in the LICENSE_WillowGarage.md file found in this module's directory

#include "precomp.hpp"

namespace cv
{
namespace rgbd
{

 ///////////////////////////////////////////////////////////////////////////////////

    // Our three input types have a different value for a depth pixel with no depth
    template<typename DepthDepth>
    inline DepthDepth
    noDepthSentinelValue()
    {
        return 0;
    }

    template<>
    inline float
    noDepthSentinelValue<float>()
    {
        return std::numeric_limits<float>::quiet_NaN();
    }

    template<>
    inline double
    noDepthSentinelValue<double>()
    {
        return std::numeric_limits<double>::quiet_NaN();
    }

///////////////////////////////////////////////////////////////////////////////////

    // Testing for depth pixels with no depth isn't straightforward for NaN values. We
    // need to specialize the equality check for floats and doubles.
    template<typename DepthDepth>
    inline bool
    isEqualToNoDepthSentinelValue(const DepthDepth &value)
    {
        return value == noDepthSentinelValue<DepthDepth>();
    }

    template<>
    inline bool
    isEqualToNoDepthSentinelValue<float>(const float &value)
    {
        return cvIsNaN(value) != 0;
    }

    template<>
    inline bool
    isEqualToNoDepthSentinelValue<double>(const double &value)
    {
        return cvIsNaN(value) != 0;
    }

 ///////////////////////////////////////////////////////////////////////////////////


    // When using the unsigned short representation, we'd like to round the values to the nearest
    // integer value. The float/double representations don't need to be rounded
    template<typename DepthDepth>
    inline DepthDepth
    floatToInputDepth(const float &value)
    {
        return (DepthDepth)value;
    }

    template<>
    inline unsigned short
    floatToInputDepth<unsigned short>(const float &value)
    {
        return (unsigned short)(value+0.5);
    }

 ///////////////////////////////////////////////////////////////////////////////////


    /** Computes a registered depth image from an unregistered image.
     *
     * @param unregisteredDepth the input depth data
     * @param unregisteredCameraMatrix the camera matrix of the depth camera
     * @param registeredCameraMatrix the camera matrix of the external camera
     * @param registeredDistCoeffs the distortion coefficients of the external camera
     * @param rbtRgb2Depth the rigid body transform between the cameras.
     * @param outputImagePlaneSize the image plane dimensions of the external camera (width, height)
     * @param depthDilation whether or not the depth is dilated to avoid holes and occlusion errors
     * @param inputDepthToMetersScale the scale needed to transform the input depth units to meters
     * @param registeredDepth the result of transforming the depth into the external camera
     */
    template<typename DepthDepth>
    void
    performRegistration(const Mat_<DepthDepth> &unregisteredDepth,
                        const Matx33f &unregisteredCameraMatrix,
                        const Matx33f &registeredCameraMatrix,
                        const Mat_<float> &registeredDistCoeffs,
                        const Matx44f &rbtRgb2Depth,
                        const Size outputImagePlaneSize,
                        const bool depthDilation,
                        const float inputDepthToMetersScale,
                        Mat &registeredDepth)
    {

        // Create output Mat of the correct type, filled with an initial value indicating no depth
        registeredDepth = Mat_<DepthDepth>(outputImagePlaneSize, noDepthSentinelValue<DepthDepth>());

        // Figure out whether we'll have to apply a distortion
        bool hasDistortion = (countNonZero(registeredDistCoeffs) > 0);

        // A point (i,j,1) will have to be converted to 3d first, by multiplying it by K.inv()
        // It will then be transformed by rbtRgb2Depth.
        // Finally, it will be projected into the external camera via registeredCameraMatrix and
        // its distortion coefficients. If there is no distortion in the external camera, we
        // can linearly chain all three operations together.

        Matx44f K = Matx44f::zeros();
        for(unsigned char j = 0; j < 3; ++j)
            for(unsigned char i = 0; i < 3; ++i)
                K(j, i) = unregisteredCameraMatrix(j, i);
        K(3, 3) = 1;

        Matx44f initialProjection;
        if (hasDistortion)
        {
            // The projection into the external camera will be done separately with distortion
            initialProjection = rbtRgb2Depth * K.inv();
        }
        else
        {
            // No distortion, so all operations can be chained
            initialProjection = Matx44f::zeros();
            for(unsigned char j = 0; j < 3; ++j)
                for(unsigned char i = 0; i < 3; ++i)
                    initialProjection(j, i) = registeredCameraMatrix(j, i);
            initialProjection(3, 3) = 1;

            initialProjection = initialProjection * rbtRgb2Depth * K.inv();
        }

        // Apply the initial projection to the input depth
        Mat_<Point3f> transformedCloud;
        {
            Mat_<Point3f> point_tmp(outputImagePlaneSize,Point3f(0.,0.,0.));

            for(int j = 0; j < unregisteredDepth.rows; ++j)
            {
                const DepthDepth *unregisteredDepthPtr = unregisteredDepth[j];

                Point3f *point = point_tmp[j];
                for(int i = 0; i < unregisteredDepth.cols; ++i, ++unregisteredDepthPtr, ++point)
                {
                    float rescaled_depth = float(*unregisteredDepthPtr) * inputDepthToMetersScale;

                    // If the DepthDepth is of type unsigned short, zero is a sentinel value to indicate
                    // no depth. CV_32F and CV_64F should already have NaN for no depth values.
                    if (rescaled_depth == 0)
                    {
                        rescaled_depth = std::numeric_limits<float>::quiet_NaN();
                    }

                    point->x = i * rescaled_depth;
                    point->y = j * rescaled_depth;
                    point->z = rescaled_depth;
                }
            }

            perspectiveTransform(point_tmp, transformedCloud, initialProjection);
        }

        std::vector<Point2f> transformedAndProjectedPoints(transformedCloud.cols);
        const float metersToInputUnitsScale = 1/inputDepthToMetersScale;
        const Rect registeredDepthBounds(Point(), outputImagePlaneSize);

        for( int y = 0; y < transformedCloud.rows; y++ )
        {
            if (hasDistortion)
            {

                // Project an entire row of points with distortion.
                // Doing this for the entire image at once would require more memory.
                projectPoints(transformedCloud.row(y),
                              Vec3f(0,0,0),
                              Vec3f(0,0,0),
                              registeredCameraMatrix,
                              registeredDistCoeffs,
                              transformedAndProjectedPoints);

            }
            else
            {

                // With no distortion, we just have to dehomogenize the point since all major transforms
                // already happened with initialProjection.
                Point2f *point2d = &transformedAndProjectedPoints[0];
                const Point2f *point2d_end = point2d + transformedAndProjectedPoints.size();
                const Point3f *point3d = transformedCloud[y];
                for( ; point2d < point2d_end; ++point2d, ++point3d )
                {
                    point2d->x = point3d->x / point3d->z;
                    point2d->y = point3d->y / point3d->z;
                }

            }

            const Point2f *outputProjectedPoint = &transformedAndProjectedPoints[0];
            const Point3f *p = transformedCloud[y], *p_end = p + transformedCloud.cols;


            for( ; p < p_end; ++outputProjectedPoint, ++p )
            {



                // Skip this one if there isn't a valid depth
                const Point2f projectedPixelFloatLocation = *outputProjectedPoint;
                if (cvIsNaN(projectedPixelFloatLocation.x))
                    continue;

                //Get integer pixel location
                const Point2i projectedPixelLocation = projectedPixelFloatLocation;

                // Ensure that the projected point is actually contained in our output image
                if (!registeredDepthBounds.contains(projectedPixelLocation))
                    continue;

                // Go back to our original scale, since that's what our output will be
                // The templated function is to ensure that integer values are rounded to the nearest integer
                const DepthDepth cloudDepth = floatToInputDepth<DepthDepth>(p->z*metersToInputUnitsScale);

                DepthDepth& outputDepth = registeredDepth.at<DepthDepth>(projectedPixelLocation.y, projectedPixelLocation.x);


                // Occlusion check
                if ( isEqualToNoDepthSentinelValue<DepthDepth>(outputDepth) || (outputDepth > cloudDepth) )
                    outputDepth = cloudDepth;


                // If desired, dilate this point to avoid holes in the final image
                if (depthDilation)
                {

                    // Choosing to dilate in a 2x2 region, where the original projected location is in the bottom right of this
                    // region. This is what's done on PrimeSense devices, but a more accurate scheme could be used.
                    const Point2i dilatedProjectedLocations[3] = {Point2i(projectedPixelLocation.x - 1, projectedPixelLocation.y    ),
                                                                  Point2i(projectedPixelLocation.x    , projectedPixelLocation.y - 1),
                                                                  Point2i(projectedPixelLocation.x - 1, projectedPixelLocation.y - 1)};

                    for (int i = 0; i < 3; i++) {

                        const Point2i& dilatedCoordinates = dilatedProjectedLocations[i];

                        if (!registeredDepthBounds.contains(dilatedCoordinates))
                            continue;

                        DepthDepth& outputDilatedDepth = registeredDepth.at<DepthDepth>(dilatedCoordinates.y, dilatedCoordinates.x);

                        // Occlusion check
                        if ( isEqualToNoDepthSentinelValue(outputDilatedDepth) || (outputDilatedDepth > cloudDepth) )
                            outputDilatedDepth = cloudDepth;

                    }

                } // depthDilation

            } // iterate cols
        } // iterate rows
    }



    void
    registerDepth(InputArray unregisteredCameraMatrix, InputArray registeredCameraMatrix,InputArray registeredDistCoeffs,
                  InputArray Rt, InputArray unregisteredDepth, const Size& outputImagePlaneSize,
                  OutputArray registeredDepth, bool depthDilation)
    {

        CV_Assert(unregisteredCameraMatrix.depth() == CV_64F || unregisteredCameraMatrix.depth() == CV_32F);

        CV_Assert(registeredCameraMatrix.depth() == CV_64F || registeredCameraMatrix.depth() == CV_32F);

        CV_Assert(registeredDistCoeffs.empty() || registeredDistCoeffs.depth() == CV_64F || registeredDistCoeffs.depth() == CV_32F);

        CV_Assert(Rt.depth() == CV_64F || Rt.depth() == CV_32F);

        CV_Assert(unregisteredDepth.cols() > 0 && unregisteredDepth.rows() > 0 &&
                  (unregisteredDepth.depth() == CV_32F || unregisteredDepth.depth() == CV_64F || unregisteredDepth.depth() == CV_16U));

        CV_Assert(outputImagePlaneSize.height > 0 && outputImagePlaneSize.width > 0);

        // Implicitly checking dimensions of the InputArrays
        Matx33f _unregisteredCameraMatrix = unregisteredCameraMatrix.getMat();
        Matx33f _registeredCameraMatrix = registeredCameraMatrix.getMat();
        Mat_<float> _registeredDistCoeffs = registeredDistCoeffs.getMat();
        Matx44f _rbtRgb2Depth = Rt.getMat();


        Mat &registeredDepthMat = registeredDepth.getMatRef();

        switch (unregisteredDepth.depth())
        {
            case CV_16U:
            {
                performRegistration<unsigned short>(unregisteredDepth.getMat(), _unregisteredCameraMatrix,
                                                    _registeredCameraMatrix, _registeredDistCoeffs,
                                                    _rbtRgb2Depth, outputImagePlaneSize, depthDilation,
                                                    .001f, registeredDepthMat);
                break;
            }
            case CV_32F:
            {
                performRegistration<float>(unregisteredDepth.getMat(), _unregisteredCameraMatrix,
                                           _registeredCameraMatrix, _registeredDistCoeffs,
                                           _rbtRgb2Depth, outputImagePlaneSize, depthDilation,
                                           1.0f, registeredDepthMat);
                break;
            }
            case CV_64F:
            {
                performRegistration<double>(unregisteredDepth.getMat(), _unregisteredCameraMatrix,
                                            _registeredCameraMatrix, _registeredDistCoeffs,
                                            _rbtRgb2Depth, outputImagePlaneSize, depthDilation,
                                            1.0f, registeredDepthMat);
                break;
            }
            default:
            {
                CV_Error(Error::StsUnsupportedFormat, "Input depth must be unsigned short, float, or double.");
            }
        }


    }

} /* namespace rgbd */
} /* namespace cv */
