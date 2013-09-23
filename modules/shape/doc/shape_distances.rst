Shape Distance and Common Interfaces
====================================

.. highlight:: cpp

Shape Distance algorithms in OpenCV are derivated from a common interface that allows you to
switch between them in a practical way for solving the same problem with different methods.
Thus, all objects that implement shape distance measures inherit the
:ocv:class:`ShapeDistanceExtractor` interface.


ShapeDistanceExtractor
----------------------
.. ocv:class:: ShapeDistanceExtractor : public Algorithm

Abstract base class for shape distance algorithms. ::

    class CV_EXPORTS_W ShapeDistanceExtractor : public Algorithm
    {
    public:
        CV_WRAP virtual float computeDistance(InputArray contour1, InputArray contour2) = 0;
    };

ShapeDistanceExtractor::computeDistance
---------------------------------------
Compute the shape distance between two shapes defined by its contours.

.. ocv:function:: float ShapeDistanceExtractor::computeDistance( InputArray contour1, InputArray contour2 )

    :param contour1: Contour defining first shape.

    :param contour2: Contour defining second shape.

ShapeContextDistanceExtractor
-----------------------------
.. ocv:class:: ShapeContextDistanceExtractor : public ShapeDistanceExtractor

Implementation of the Shape Context descriptor and matching algorithm proposed by Belongie et al. in
"Shape Matching and Object Recognition Using Shape Contexts" (PAMI 2002).
This implementation is packaged in a generic scheme, in order to allow you the implementation of the
common variations of the original pipeline. ::

    class CV_EXPORTS_W ShapeContextDistanceExtractor : public ShapeDistanceExtractor
    {
    public:
        CV_WRAP virtual void setAngularBins(int nAngularBins) = 0;
        CV_WRAP virtual int getAngularBins() const = 0;

        CV_WRAP virtual void setRadialBins(int nRadialBins) = 0;
        CV_WRAP virtual int getRadialBins() const = 0;

        CV_WRAP virtual void setInnerRadius(float innerRadius) = 0;
        CV_WRAP virtual float getInnerRadius() const = 0;

        CV_WRAP virtual void setOuterRadius(float outerRadius) = 0;
        CV_WRAP virtual float getOuterRadius() const = 0;

        CV_WRAP virtual void setRotationInvariant(bool rotationInvariant) = 0;
        CV_WRAP virtual bool getRotationInvariant() const = 0;

        CV_WRAP virtual void setShapeContextWeight(float shapeContextWeight) = 0;
        CV_WRAP virtual float getShapeContextWeight() const = 0;

        CV_WRAP virtual void setImageAppearanceWeight(float imageAppearanceWeight) = 0;
        CV_WRAP virtual float getImageAppearanceWeight() const = 0;

        CV_WRAP virtual void setBendingEnergyWeight(float bendingEnergyWeight) = 0;
        CV_WRAP virtual float getBendingEnergyWeight() const = 0;

        CV_WRAP virtual void setImages(InputArray image1, InputArray image2) = 0;
        CV_WRAP virtual void getImages(OutputArray image1, OutputArray image2) const = 0;

        CV_WRAP virtual void setIterations(int iterations) = 0;
        CV_WRAP virtual int getIterations() const = 0;

        CV_WRAP virtual void setCostExtractor(Ptr<HistogramCostExtractor> comparer) = 0;
        CV_WRAP virtual Ptr<HistogramCostExtractor> getCostExtractor() const = 0;

        CV_WRAP virtual void setTransformAlgorithm(Ptr<ShapeTransformer> transformer) = 0;
        CV_WRAP virtual Ptr<ShapeTransformer> getTransformAlgorithm() const = 0;
    };

    /* Complete constructor */
    CV_EXPORTS_W Ptr<ShapeContextDistanceExtractor>
        createShapeContextDistanceExtractor(int nAngularBins=12, int nRadialBins=4,
                                            float innerRadius=0.2, float outerRadius=2, int iterations=3,
                                            const Ptr<HistogramCostExtractor> &comparer = createChiHistogramCostExtractor(),
                                            const Ptr<ShapeTransformer> &transformer = createThinPlateSplineShapeTransformer());

ShapeContextDistanceExtractor::setAngularBins
---------------------------------------------
Establish the number of angular bins for the Shape Context Descriptor used in the shape matching pipeline.

.. ocv:function:: void setAngularBins( int nAngularBins )

    :param nAngularBins: The number of angular bins in the shape context descriptor.

ShapeContextDistanceExtractor::setRadialBins
--------------------------------------------
Establish the number of radial bins for the Shape Context Descriptor used in the shape matching pipeline.

.. ocv:function:: void setRadialBins( int nRadialBins )

    :param nRadialBins: The number of radial bins in the shape context descriptor.

ShapeContextDistanceExtractor::setInnerRadius
---------------------------------------------
Set the inner radius of the shape context descriptor.

.. ocv:function:: void setInnerRadius(float innerRadius)

    :param innerRadius: The value of the inner radius.

ShapeContextDistanceExtractor::setOuterRadius
---------------------------------------------
Set the outer radius of the shape context descriptor.

.. ocv:function:: void setOuterRadius(float outerRadius)

    :param outerRadius: The value of the outer radius.

ShapeContextDistanceExtractor::setShapeContextWeight
----------------------------------------------------
Set the weight of the shape context distance in the final value of the shape distance.
The shape context distance between two shapes is defined as the symmetric sum of shape
context matching costs over best matching points.
The final value of the shape distance is a user-defined linear combination of the shape
context distance, an image appearance distance, and a bending energy.

.. ocv:function:: void setShapeContextWeight( float shapeContextWeight )

    :param shapeContextWeight: The weight of the shape context distance in the final distance value.

ShapeContextDistanceExtractor::setImageAppearanceWeight
-------------------------------------------------------
Set the weight of the Image Appearance cost in the final value of the shape distance.
The image appearance cost is defined as the sum of squared brightness differences in
Gaussian windows around corresponding image points.
The final value of the shape distance is a user-defined linear combination of the shape
context distance, an image appearance distance, and a bending energy.
If this value is set to a number different from 0, is mandatory to set the images that
correspond to each shape.

.. ocv:function:: void setImageAppearanceWeight( float imageAppearanceWeight )

    :param imageAppearanceWeight: The weight of the appearance cost in the final distance value.

ShapeContextDistanceExtractor::setBendingEnergyWeight
-----------------------------------------------------
Set the weight of the Bending Energy in the final value of the shape distance.
The bending energy definition depends on what transformation is being used to align the
shapes.
The final value of the shape distance is a user-defined linear combination of the shape
context distance, an image appearance distance, and a bending energy.

.. ocv:function:: void setBendingEnergyWeight( float bendingEnergyWeight )

    :param bendingEnergyWeight: The weight of the Bending Energy in the final distance value.

ShapeContextDistanceExtractor::setImages
----------------------------------------
Set the images that correspond to each shape. This images are used in the calculation of the
Image Appearance cost.

.. ocv:function:: void setImages( InputArray image1, InputArray image2 )

    :param image1: Image corresponding to the shape defined by ``contours1``.

    :param image2: Image corresponding to the shape defined by ``contours2``.

ShapeContextDistanceExtractor::setCostExtractor
-----------------------------------------------
Set the algorithm used for building the shape context descriptor cost matrix.

.. ocv:function:: void setCostExtractor( Ptr<HistogramCostExtractor> comparer )

    :param comparer: Smart pointer to a HistogramCostExtractor, an algorithm that defines the cost matrix between descriptors.

ShapeContextDistanceExtractor::setStdDev
----------------------------------------
Set the value of the standard deviation for the Gaussian window for the image appearance cost.

.. ocv:function:: void setStdDev( float sigma )

    :param sigma: Standard Deviation.

ShapeContextDistanceExtractor::setTransformAlgorithm
----------------------------------------------------
Set the algorithm used for aligning the shapes.

.. ocv:function:: void setTransformAlgorithm( Ptr<ShapeTransformer> transformer )

    :param comparer: Smart pointer to a ShapeTransformer, an algorithm that defines the aligning transformation.

HausdorffDistanceExtractor
--------------------------
.. ocv:class:: HausdorffDistanceExtractor : public ShapeDistanceExtractor

A simple Hausdorff distance measure between shapes defined by contours,
according to the paper "Comparing Images using the Hausdorff distance." by
D.P. Huttenlocher, G.A. Klanderman, and W.J. Rucklidge. (PAMI 1993). ::

    class CV_EXPORTS_W HausdorffDistanceExtractor : public ShapeDistanceExtractor
    {
    public:
        CV_WRAP virtual void setDistanceFlag(int distanceFlag) = 0;
        CV_WRAP virtual int getDistanceFlag() const = 0;

        CV_WRAP virtual void setRankProportion(float rankProportion) = 0;
        CV_WRAP virtual float getRankProportion() const = 0;
    };

    /* Constructor */
    CV_EXPORTS_W Ptr<HausdorffDistanceExtractor> createHausdorffDistanceExtractor(int distanceFlag=cv::NORM_L2, float rankProp=0.6);

HausdorffDistanceExtractor::setDistanceFlag
-------------------------------------------
Set the norm used to compute the Hausdorff value between two shapes. It can be L1 or L2 norm.

.. ocv:function:: void setDistanceFlag( int distanceFlag )

    :param distanceFlag: Flag indicating which norm is used to compute the Hausdorff distance (NORM_L1, NORM_L2).

HausdorffDistanceExtractor::setRankProportion
---------------------------------------------
This method sets the rank proportion (or fractional value) that establish the Kth ranked value of the
partial Hausdorff distance. Experimentally had been shown that 0.6 is a good value to compare shapes.

.. ocv:function:: void setRankProportion( float rankProportion )

    :param rankProportion: fractional value (between 0 and 1).
