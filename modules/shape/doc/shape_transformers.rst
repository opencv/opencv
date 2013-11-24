Shape Transformers and Interfaces
=================================

.. highlight:: cpp

A virtual interface that ease the use of transforming algorithms in some pipelines, such as
the Shape Context Matching Algorithm. Thus, all objects that implement shape transformation
techniques inherit the
:ocv:class:`ShapeTransformer` interface.

ShapeTransformer
----------------
.. ocv:class:: ShapeTransformer : public Algorithm

Abstract base class for shape transformation algorithms. ::

    class CV_EXPORTS_W ShapeTransformer : public Algorithm
    {
    public:
        CV_WRAP virtual void estimateTransformation(InputArray transformingShape, InputArray targetShape,
                                                     std::vector<DMatch>& matches) = 0;

        CV_WRAP virtual float applyTransformation(InputArray input, OutputArray output=noArray()) = 0;

        CV_WRAP virtual void warpImage(InputArray transformingImage, OutputArray output,
                                       int flags=INTER_LINEAR, int borderMode=BORDER_CONSTANT,
                                       const Scalar& borderValue=Scalar()) const = 0;
    };

ShapeTransformer::estimateTransformation
----------------------------------------
Estimate the transformation parameters of the current transformer algorithm, based on point matches.

.. ocv:function:: void estimateTransformation( InputArray transformingShape, InputArray targetShape, std::vector<DMatch>& matches )

    :param transformingShape: Contour defining first shape.

    :param targetShape: Contour defining second shape (Target).

    :param matches: Standard vector of Matches between points.

ShapeTransformer::applyTransformation
-------------------------------------
Apply a transformation, given a pre-estimated transformation parameters.

.. ocv:function:: float applyTransformation( InputArray input, OutputArray output=noArray() )

    :param input: Contour (set of points) to apply the transformation.

    :param output: Output contour.

ShapeTransformer::warpImage
---------------------------
Apply a transformation, given a pre-estimated transformation parameters, to an Image.

.. ocv:function:: void warpImage( InputArray transformingImage, OutputArray output, int flags=INTER_LINEAR, int borderMode=BORDER_CONSTANT, const Scalar& borderValue=Scalar() )

    :param transformingImage: Input image.

    :param output: Output image.

    :param flags: Image interpolation method.

    :param borderMode: border style.

    :param borderValue: border value.

ThinPlateSplineShapeTransformer
-------------------------------
.. ocv:class:: ThinPlateSplineShapeTransformer : public Algorithm

Definition of the transformation ocupied in the paper "Principal Warps: Thin-Plate Splines and Decomposition
of Deformations", by F.L. Bookstein (PAMI 1989). ::

    class CV_EXPORTS_W ThinPlateSplineShapeTransformer : public ShapeTransformer
    {
    public:
        CV_WRAP virtual void setRegularizationParameter(double beta) = 0;
        CV_WRAP virtual double getRegularizationParameter() const = 0;
    };

    /* Complete constructor */
    CV_EXPORTS_W Ptr<ThinPlateSplineShapeTransformer>
        createThinPlateSplineShapeTransformer(double regularizationParameter=0);

ThinPlateSplineShapeTransformer::setRegularizationParameter
-----------------------------------------------------------
Set the regularization parameter for relaxing the exact interpolation requirements of the TPS algorithm.

.. ocv:function:: void setRegularizationParameter( double beta )

    :param beta: value of the regularization parameter.

AffineTransformer
-----------------
.. ocv:class:: AffineTransformer : public Algorithm

Wrapper class for the OpenCV Affine Transformation algorithm. ::

    class CV_EXPORTS_W AffineTransformer : public ShapeTransformer
    {
    public:
        CV_WRAP virtual void setFullAffine(bool fullAffine) = 0;
        CV_WRAP virtual bool getFullAffine() const = 0;
    };

    /* Complete constructor */
    CV_EXPORTS_W Ptr<AffineTransformer> createAffineTransformer(bool fullAffine);
