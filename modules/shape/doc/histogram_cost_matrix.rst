Cost Matrix for Histograms Common Interface
===========================================

.. highlight:: cpp

A common interface is defined to ease the implementation of some algorithms pipelines, such
as the Shape Context Matching Algorithm. A common class is defined, so any object that implements
a Cost Matrix builder inherits the
:ocv:class:`HistogramCostExtractor` interface.

HistogramCostExtractor
----------------------
.. ocv:class:: HistogramCostExtractor : public Algorithm

Abstract base class for histogram cost algorithms. ::

    class CV_EXPORTS_W HistogramCostExtractor : public Algorithm
    {
    public:
        CV_WRAP virtual void buildCostMatrix(InputArray descriptors1, InputArray descriptors2, OutputArray costMatrix) = 0;

        CV_WRAP virtual void setNDummies(int nDummies) = 0;
        CV_WRAP virtual int getNDummies() const = 0;

        CV_WRAP virtual void setDefaultCost(float defaultCost) = 0;
        CV_WRAP virtual float getDefaultCost() const = 0;
    };

NormHistogramCostExtractor
--------------------------
.. ocv:class:: NormHistogramCostExtractor : public HistogramCostExtractor

A norm based cost extraction. ::

    class CV_EXPORTS_W NormHistogramCostExtractor : public HistogramCostExtractor
    {
    public:
        CV_WRAP virtual void setNormFlag(int flag) = 0;
        CV_WRAP virtual int getNormFlag() const = 0;
    };

    CV_EXPORTS_W Ptr<HistogramCostExtractor>
        createNormHistogramCostExtractor(int flag=cv::DIST_L2, int nDummies=25, float defaultCost=0.2);

EMDHistogramCostExtractor
-------------------------
.. ocv:class:: EMDHistogramCostExtractor : public HistogramCostExtractor

An EMD based cost extraction. ::

    class CV_EXPORTS_W EMDHistogramCostExtractor : public HistogramCostExtractor
    {
    public:
        CV_WRAP virtual void setNormFlag(int flag) = 0;
        CV_WRAP virtual int getNormFlag() const = 0;
    };

    CV_EXPORTS_W Ptr<HistogramCostExtractor>
        createEMDHistogramCostExtractor(int flag=cv::DIST_L2, int nDummies=25, float defaultCost=0.2);

ChiHistogramCostExtractor
-------------------------
.. ocv:class:: ChiHistogramCostExtractor : public HistogramCostExtractor

An Chi based cost extraction. ::

    class CV_EXPORTS_W ChiHistogramCostExtractor : public HistogramCostExtractor
    {};

    CV_EXPORTS_W Ptr<HistogramCostExtractor> createChiHistogramCostExtractor(int nDummies=25, float defaultCost=0.2);

EMDL1HistogramCostExtractor
---------------------------
.. ocv:class:: EMDL1HistogramCostExtractor : public HistogramCostExtractor

An EMD-L1 based cost extraction. ::

    class CV_EXPORTS_W EMDL1HistogramCostExtractor : public HistogramCostExtractor
    {};

    CV_EXPORTS_W Ptr<HistogramCostExtractor>
        createEMDL1HistogramCostExtractor(int nDummies=25, float defaultCost=0.2);
