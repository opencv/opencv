#pragma once

#include "opencv2/core/ocl.hpp"

namespace cv
{

void clipObjects(Size sz, std::vector<Rect>& objects,
                 std::vector<int>* a, std::vector<double>* b);

class FeatureEvaluator
{
public:
    enum
    {
        HAAR = 0,
        LBP  = 1,
        HOG  = 2
    };

    struct ScaleData
    {
        ScaleData() { scale = 0.f; layer_ofs = ystep = 0; }
        Size getWorkingSize(Size winSize) const
        {
            return Size(std::max(szi.width - winSize.width, 0),
                        std::max(szi.height - winSize.height, 0));
        }

        float scale;
        Size szi;
        int layer_ofs, ystep;
    };

    virtual ~FeatureEvaluator();

    virtual bool read(const FileNode& node, Size origWinSize);
    virtual Ptr<FeatureEvaluator> clone() const;
    virtual int getFeatureType() const;
    int getNumChannels() const { return nchannels; }

    virtual bool setImage(InputArray img, const std::vector<float>& scales);
    virtual bool setWindow(Point p, int scaleIdx);
    const ScaleData& getScaleData(int scaleIdx) const
    {
        CV_Assert( 0 <= scaleIdx && scaleIdx < (int)scaleData->size());
        return scaleData->at(scaleIdx);
    }
    virtual void getUMats(std::vector<UMat>& bufs);
    virtual void getMats();

    Size getLocalSize() const { return localSize; }
    Size getLocalBufSize() const { return lbufSize; }

    virtual float calcOrd(int featureIdx) const;
    virtual int calcCat(int featureIdx) const;

    static Ptr<FeatureEvaluator> create(int type);

protected:
    enum { SBUF_VALID=1, USBUF_VALID=2 };
    int sbufFlag;

    bool updateScaleData( Size imgsz, const std::vector<float>& _scales );
    virtual void computeChannels( int, InputArray ) {}
    virtual void computeOptFeatures() {}

    Size origWinSize, sbufSize, localSize, lbufSize;
    int nchannels;
    Mat sbuf, rbuf;
    UMat urbuf, usbuf, ufbuf, uscaleData;

    Ptr<std::vector<ScaleData> > scaleData;
};


class CascadeClassifierImpl : public BaseCascadeClassifier
{
public:
    CascadeClassifierImpl();
    virtual ~CascadeClassifierImpl();

    bool empty() const;
    bool load( const String& filename );
    void read( const FileNode& node );
    bool read_( const FileNode& node );
    void detectMultiScale( InputArray image,
                          CV_OUT std::vector<Rect>& objects,
                          double scaleFactor = 1.1,
                          int minNeighbors = 3, int flags = 0,
                          Size minSize = Size(),
                          Size maxSize = Size() );

    void detectMultiScale( InputArray image,
                          CV_OUT std::vector<Rect>& objects,
                          CV_OUT std::vector<int>& numDetections,
                          double scaleFactor=1.1,
                          int minNeighbors=3, int flags=0,
                          Size minSize=Size(),
                          Size maxSize=Size() );

    void detectMultiScale( InputArray image,
                          CV_OUT std::vector<Rect>& objects,
                          CV_OUT std::vector<int>& rejectLevels,
                          CV_OUT std::vector<double>& levelWeights,
                          double scaleFactor = 1.1,
                          int minNeighbors = 3, int flags = 0,
                          Size minSize = Size(),
                          Size maxSize = Size(),
                          bool outputRejectLevels = false );


    bool isOldFormatCascade() const;
    Size getOriginalWindowSize() const;
    int getFeatureType() const;
    void* getOldCascade();

    void setMaskGenerator(const Ptr<MaskGenerator>& maskGenerator);
    Ptr<MaskGenerator> getMaskGenerator();

protected:
    enum { SUM_ALIGN = 64 };

    bool detectSingleScale( InputArray image, Size processingRectSize,
                            int yStep, double factor, std::vector<Rect>& candidates,
                            std::vector<int>& rejectLevels, std::vector<double>& levelWeights,
                            Size sumSize0, bool outputRejectLevels = false );
#ifdef HAVE_OPENCL
    bool ocl_detectMultiScaleNoGrouping( const std::vector<float>& scales,
                                         std::vector<Rect>& candidates );
#endif
    void detectMultiScaleNoGrouping( InputArray image, std::vector<Rect>& candidates,
                                    std::vector<int>& rejectLevels, std::vector<double>& levelWeights,
                                    double scaleFactor, Size minObjectSize, Size maxObjectSize,
                                    bool outputRejectLevels = false );

    enum { MAX_FACES = 10000 };
    enum { BOOST = 0 };
    enum { DO_CANNY_PRUNING    = CASCADE_DO_CANNY_PRUNING,
        SCALE_IMAGE         = CASCADE_SCALE_IMAGE,
        FIND_BIGGEST_OBJECT = CASCADE_FIND_BIGGEST_OBJECT,
        DO_ROUGH_SEARCH     = CASCADE_DO_ROUGH_SEARCH
    };

    friend class CascadeClassifierInvoker;
    friend class SparseCascadeClassifierInvoker;

    template<class FEval>
    friend int predictOrdered( CascadeClassifierImpl& cascade, Ptr<FeatureEvaluator> &featureEvaluator, double& weight);

    template<class FEval>
    friend int predictCategorical( CascadeClassifierImpl& cascade, Ptr<FeatureEvaluator> &featureEvaluator, double& weight);

    template<class FEval>
    friend int predictOrderedStump( CascadeClassifierImpl& cascade, Ptr<FeatureEvaluator> &featureEvaluator, double& weight);

    template<class FEval>
    friend int predictCategoricalStump( CascadeClassifierImpl& cascade, Ptr<FeatureEvaluator> &featureEvaluator, double& weight);

    int runAt( Ptr<FeatureEvaluator>& feval, Point pt, int scaleIdx, double& weight );

    class Data
    {
    public:
        struct DTreeNode
        {
            int featureIdx;
            float threshold; // for ordered features only
            int left;
            int right;
        };

        struct DTree
        {
            int nodeCount;
        };

        struct Stage
        {
            int first;
            int ntrees;
            float threshold;
        };

        struct Stump
        {
            Stump() { }
            Stump(int _featureIdx, float _threshold, float _left, float _right)
            : featureIdx(_featureIdx), threshold(_threshold), left(_left), right(_right) {}

            int featureIdx;
            float threshold;
            float left;
            float right;
        };

        Data();

        bool read(const FileNode &node);

        int stageType;
        int featureType;
        int ncategories;
        int minNodesPerTree, maxNodesPerTree;
        Size origWinSize;

        std::vector<Stage> stages;
        std::vector<DTree> classifiers;
        std::vector<DTreeNode> nodes;
        std::vector<float> leaves;
        std::vector<int> subsets;
        std::vector<Stump> stumps;
    };

    Data data;
    Ptr<FeatureEvaluator> featureEvaluator;
    Ptr<CvHaarClassifierCascade> oldCascade;

    Ptr<MaskGenerator> maskGenerator;
    UMat ugrayImage;
    UMat ufacepos, ustages, unodes, uleaves, usubsets;
#ifdef HAVE_OPENCL
    ocl::Kernel haarKernel, lbpKernel;
    bool tryOpenCL;
#endif

    Mutex mtx;
};

#define CC_CASCADE_PARAMS "cascadeParams"
#define CC_STAGE_TYPE     "stageType"
#define CC_FEATURE_TYPE   "featureType"
#define CC_HEIGHT         "height"
#define CC_WIDTH          "width"

#define CC_STAGE_NUM    "stageNum"
#define CC_STAGES       "stages"
#define CC_STAGE_PARAMS "stageParams"

#define CC_BOOST            "BOOST"
#define CC_MAX_DEPTH        "maxDepth"
#define CC_WEAK_COUNT       "maxWeakCount"
#define CC_STAGE_THRESHOLD  "stageThreshold"
#define CC_WEAK_CLASSIFIERS "weakClassifiers"
#define CC_INTERNAL_NODES   "internalNodes"
#define CC_LEAF_VALUES      "leafValues"

#define CC_FEATURES       "features"
#define CC_FEATURE_PARAMS "featureParams"
#define CC_MAX_CAT_COUNT  "maxCatCount"

#define CC_HAAR   "HAAR"
#define CC_RECTS  "rects"
#define CC_TILTED "tilted"

#define CC_LBP  "LBP"
#define CC_RECT "rect"

#define CC_HOG  "HOG"

#define CV_SUM_PTRS( p0, p1, p2, p3, sum, rect, step )                    \
    /* (x, y) */                                                          \
    (p0) = sum + (rect).x + (step) * (rect).y,                            \
    /* (x + w, y) */                                                      \
    (p1) = sum + (rect).x + (rect).width + (step) * (rect).y,             \
    /* (x + w, y) */                                                      \
    (p2) = sum + (rect).x + (step) * ((rect).y + (rect).height),          \
    /* (x + w, y + h) */                                                  \
    (p3) = sum + (rect).x + (rect).width + (step) * ((rect).y + (rect).height)

#define CV_TILTED_PTRS( p0, p1, p2, p3, tilted, rect, step )                        \
    /* (x, y) */                                                                    \
    (p0) = tilted + (rect).x + (step) * (rect).y,                                   \
    /* (x - h, y + h) */                                                            \
    (p1) = tilted + (rect).x - (rect).height + (step) * ((rect).y + (rect).height), \
    /* (x + w, y + w) */                                                            \
    (p2) = tilted + (rect).x + (rect).width + (step) * ((rect).y + (rect).width),   \
    /* (x + w - h, y + w + h) */                                                    \
    (p3) = tilted + (rect).x + (rect).width - (rect).height                         \
           + (step) * ((rect).y + (rect).width + (rect).height)

#define CALC_SUM_(p0, p1, p2, p3, offset) \
    ((p0)[offset] - (p1)[offset] - (p2)[offset] + (p3)[offset])

#define CALC_SUM(rect,offset) CALC_SUM_((rect)[0], (rect)[1], (rect)[2], (rect)[3], offset)

#define CV_SUM_OFS( p0, p1, p2, p3, sum, rect, step )                 \
/* (x, y) */                                                          \
(p0) = sum + (rect).x + (step) * (rect).y,                            \
/* (x + w, y) */                                                      \
(p1) = sum + (rect).x + (rect).width + (step) * (rect).y,             \
/* (x + w, y) */                                                      \
(p2) = sum + (rect).x + (step) * ((rect).y + (rect).height),          \
/* (x + w, y + h) */                                                  \
(p3) = sum + (rect).x + (rect).width + (step) * ((rect).y + (rect).height)

#define CV_TILTED_OFS( p0, p1, p2, p3, tilted, rect, step )                     \
/* (x, y) */                                                                    \
(p0) = tilted + (rect).x + (step) * (rect).y,                                   \
/* (x - h, y + h) */                                                            \
(p1) = tilted + (rect).x - (rect).height + (step) * ((rect).y + (rect).height), \
/* (x + w, y + w) */                                                            \
(p2) = tilted + (rect).x + (rect).width + (step) * ((rect).y + (rect).width),   \
/* (x + w - h, y + w + h) */                                                    \
(p3) = tilted + (rect).x + (rect).width - (rect).height                         \
+ (step) * ((rect).y + (rect).width + (rect).height)

#define CALC_SUM_(p0, p1, p2, p3, offset) \
((p0)[offset] - (p1)[offset] - (p2)[offset] + (p3)[offset])

#define CALC_SUM(rect,offset) CALC_SUM_((rect)[0], (rect)[1], (rect)[2], (rect)[3], offset)

#define CALC_SUM_OFS_(p0, p1, p2, p3, ptr) \
((ptr)[p0] - (ptr)[p1] - (ptr)[p2] + (ptr)[p3])

#define CALC_SUM_OFS(rect, ptr) CALC_SUM_OFS_((rect)[0], (rect)[1], (rect)[2], (rect)[3], ptr)

//----------------------------------------------  HaarEvaluator ---------------------------------------
class HaarEvaluator : public FeatureEvaluator
{
public:
    struct Feature
    {
        Feature();
        bool read( const FileNode& node );

        bool tilted;

        enum { RECT_NUM = 3 };
        struct
        {
            Rect r;
            float weight;
        } rect[RECT_NUM];
    };

    struct OptFeature
    {
        OptFeature();

        enum { RECT_NUM = Feature::RECT_NUM };
        float calc( const int* pwin ) const;
        void setOffsets( const Feature& _f, int step, int tofs );

        int ofs[RECT_NUM][4];
        float weight[4];
    };

    HaarEvaluator();
    virtual ~HaarEvaluator();

    virtual bool read( const FileNode& node, Size origWinSize);
    virtual Ptr<FeatureEvaluator> clone() const;
    virtual int getFeatureType() const { return FeatureEvaluator::HAAR; }

    virtual bool setWindow(Point p, int scaleIdx);
    Rect getNormRect() const;
    int getSquaresOffset() const;

    float operator()(int featureIdx) const
    { return optfeaturesPtr[featureIdx].calc(pwin) * varianceNormFactor; }
    virtual float calcOrd(int featureIdx) const
    { return (*this)(featureIdx); }

protected:
    virtual void computeChannels( int i, InputArray img );
    virtual void computeOptFeatures();

    Ptr<std::vector<Feature> > features;
    Ptr<std::vector<OptFeature> > optfeatures;
    Ptr<std::vector<OptFeature> > optfeatures_lbuf;
    bool hasTiltedFeatures;

    int tofs, sqofs;
    Vec4i nofs;
    Rect normrect;
    const int* pwin;
    OptFeature* optfeaturesPtr; // optimization
    float varianceNormFactor;
};

inline HaarEvaluator::Feature :: Feature()
{
    tilted = false;
    rect[0].r = rect[1].r = rect[2].r = Rect();
    rect[0].weight = rect[1].weight = rect[2].weight = 0;
}

inline HaarEvaluator::OptFeature :: OptFeature()
{
    weight[0] = weight[1] = weight[2] = 0.f;

    ofs[0][0] = ofs[0][1] = ofs[0][2] = ofs[0][3] =
    ofs[1][0] = ofs[1][1] = ofs[1][2] = ofs[1][3] =
    ofs[2][0] = ofs[2][1] = ofs[2][2] = ofs[2][3] = 0;
}

inline float HaarEvaluator::OptFeature :: calc( const int* ptr ) const
{
    float ret = weight[0] * CALC_SUM_OFS(ofs[0], ptr) +
                weight[1] * CALC_SUM_OFS(ofs[1], ptr);

    if( weight[2] != 0.0f )
        ret += weight[2] * CALC_SUM_OFS(ofs[2], ptr);

    return ret;
}

//----------------------------------------------  LBPEvaluator -------------------------------------

class LBPEvaluator : public FeatureEvaluator
{
public:
    struct Feature
    {
        Feature();
        Feature( int x, int y, int _block_w, int _block_h  ) :
                 rect(x, y, _block_w, _block_h) {}

        bool read(const FileNode& node );

        Rect rect; // weight and height for block
    };

    struct OptFeature
    {
        OptFeature();

        int calc( const int* pwin ) const;
        void setOffsets( const Feature& _f, int step );
        int ofs[16];
    };

    LBPEvaluator();
    virtual ~LBPEvaluator();

    virtual bool read( const FileNode& node, Size origWinSize );
    virtual Ptr<FeatureEvaluator> clone() const;
    virtual int getFeatureType() const { return FeatureEvaluator::LBP; }

    virtual bool setWindow(Point p, int scaleIdx);

    int operator()(int featureIdx) const
    { return optfeaturesPtr[featureIdx].calc(pwin); }
    virtual int calcCat(int featureIdx) const
    { return (*this)(featureIdx); }
protected:
    virtual void computeChannels( int i, InputArray img );
    virtual void computeOptFeatures();

    Ptr<std::vector<Feature> > features;
    Ptr<std::vector<OptFeature> > optfeatures;
    Ptr<std::vector<OptFeature> > optfeatures_lbuf;
    OptFeature* optfeaturesPtr; // optimization

    const int* pwin;
};


inline LBPEvaluator::Feature :: Feature()
{
    rect = Rect();
}

inline LBPEvaluator::OptFeature :: OptFeature()
{
    for( int i = 0; i < 16; i++ )
        ofs[i] = 0;
}

inline int LBPEvaluator::OptFeature :: calc( const int* p ) const
{
    int cval = CALC_SUM_OFS_( ofs[5], ofs[6], ofs[9], ofs[10], p );

    return (CALC_SUM_OFS_( ofs[0], ofs[1], ofs[4], ofs[5], p ) >= cval ? 128 : 0) |   // 0
           (CALC_SUM_OFS_( ofs[1], ofs[2], ofs[5], ofs[6], p ) >= cval ? 64 : 0) |    // 1
           (CALC_SUM_OFS_( ofs[2], ofs[3], ofs[6], ofs[7], p ) >= cval ? 32 : 0) |    // 2
           (CALC_SUM_OFS_( ofs[6], ofs[7], ofs[10], ofs[11], p ) >= cval ? 16 : 0) |  // 5
           (CALC_SUM_OFS_( ofs[10], ofs[11], ofs[14], ofs[15], p ) >= cval ? 8 : 0)|  // 8
           (CALC_SUM_OFS_( ofs[9], ofs[10], ofs[13], ofs[14], p ) >= cval ? 4 : 0)|   // 7
           (CALC_SUM_OFS_( ofs[8], ofs[9], ofs[12], ofs[13], p ) >= cval ? 2 : 0)|    // 6
           (CALC_SUM_OFS_( ofs[4], ofs[5], ofs[8], ofs[9], p ) >= cval ? 1 : 0);
}


//----------------------------------------------  predictor functions -------------------------------------

template<class FEval>
inline int predictOrdered( CascadeClassifierImpl& cascade,
                           Ptr<FeatureEvaluator> &_featureEvaluator, double& sum )
{
    int nstages = (int)cascade.data.stages.size();
    int nodeOfs = 0, leafOfs = 0;
    FEval& featureEvaluator = (FEval&)*_featureEvaluator;
    float* cascadeLeaves = &cascade.data.leaves[0];
    CascadeClassifierImpl::Data::DTreeNode* cascadeNodes = &cascade.data.nodes[0];
    CascadeClassifierImpl::Data::DTree* cascadeWeaks = &cascade.data.classifiers[0];
    CascadeClassifierImpl::Data::Stage* cascadeStages = &cascade.data.stages[0];

    for( int si = 0; si < nstages; si++ )
    {
        CascadeClassifierImpl::Data::Stage& stage = cascadeStages[si];
        int wi, ntrees = stage.ntrees;
        sum = 0;

        for( wi = 0; wi < ntrees; wi++ )
        {
            CascadeClassifierImpl::Data::DTree& weak = cascadeWeaks[stage.first + wi];
            int idx = 0, root = nodeOfs;

            do
            {
                CascadeClassifierImpl::Data::DTreeNode& node = cascadeNodes[root + idx];
                double val = featureEvaluator(node.featureIdx);
                idx = val < node.threshold ? node.left : node.right;
            }
            while( idx > 0 );
            sum += cascadeLeaves[leafOfs - idx];
            nodeOfs += weak.nodeCount;
            leafOfs += weak.nodeCount + 1;
        }
        if( sum < stage.threshold )
            return -si;
    }
    return 1;
}

template<class FEval>
inline int predictCategorical( CascadeClassifierImpl& cascade,
                               Ptr<FeatureEvaluator> &_featureEvaluator, double& sum )
{
    int nstages = (int)cascade.data.stages.size();
    int nodeOfs = 0, leafOfs = 0;
    FEval& featureEvaluator = (FEval&)*_featureEvaluator;
    size_t subsetSize = (cascade.data.ncategories + 31)/32;
    int* cascadeSubsets = &cascade.data.subsets[0];
    float* cascadeLeaves = &cascade.data.leaves[0];
    CascadeClassifierImpl::Data::DTreeNode* cascadeNodes = &cascade.data.nodes[0];
    CascadeClassifierImpl::Data::DTree* cascadeWeaks = &cascade.data.classifiers[0];
    CascadeClassifierImpl::Data::Stage* cascadeStages = &cascade.data.stages[0];

    for(int si = 0; si < nstages; si++ )
    {
        CascadeClassifierImpl::Data::Stage& stage = cascadeStages[si];
        int wi, ntrees = stage.ntrees;
        sum = 0;

        for( wi = 0; wi < ntrees; wi++ )
        {
            CascadeClassifierImpl::Data::DTree& weak = cascadeWeaks[stage.first + wi];
            int idx = 0, root = nodeOfs;
            do
            {
                CascadeClassifierImpl::Data::DTreeNode& node = cascadeNodes[root + idx];
                int c = featureEvaluator(node.featureIdx);
                const int* subset = &cascadeSubsets[(root + idx)*subsetSize];
                idx = (subset[c>>5] & (1 << (c & 31))) ? node.left : node.right;
            }
            while( idx > 0 );
            sum += cascadeLeaves[leafOfs - idx];
            nodeOfs += weak.nodeCount;
            leafOfs += weak.nodeCount + 1;
        }
        if( sum < stage.threshold )
            return -si;
    }
    return 1;
}

template<class FEval>
inline int predictOrderedStump( CascadeClassifierImpl& cascade,
                                Ptr<FeatureEvaluator> &_featureEvaluator, double& sum )
{
    CV_Assert(!cascade.data.stumps.empty());
    FEval& featureEvaluator = (FEval&)*_featureEvaluator;
    const CascadeClassifierImpl::Data::Stump* cascadeStumps = &cascade.data.stumps[0];
    const CascadeClassifierImpl::Data::Stage* cascadeStages = &cascade.data.stages[0];

    int nstages = (int)cascade.data.stages.size();
    double tmp = 0;

    for( int stageIdx = 0; stageIdx < nstages; stageIdx++ )
    {
        const CascadeClassifierImpl::Data::Stage& stage = cascadeStages[stageIdx];
        tmp = 0;

        int ntrees = stage.ntrees;
        for( int i = 0; i < ntrees; i++ )
        {
            const CascadeClassifierImpl::Data::Stump& stump = cascadeStumps[i];
            double value = featureEvaluator(stump.featureIdx);
            tmp += value < stump.threshold ? stump.left : stump.right;
        }

        if( tmp < stage.threshold )
        {
            sum = (double)tmp;
            return -stageIdx;
        }
        cascadeStumps += ntrees;
    }

    sum = (double)tmp;
    return 1;
}

template<class FEval>
inline int predictCategoricalStump( CascadeClassifierImpl& cascade,
                                    Ptr<FeatureEvaluator> &_featureEvaluator, double& sum )
{
    CV_Assert(!cascade.data.stumps.empty());
    int nstages = (int)cascade.data.stages.size();
    FEval& featureEvaluator = (FEval&)*_featureEvaluator;
    size_t subsetSize = (cascade.data.ncategories + 31)/32;
    const int* cascadeSubsets = &cascade.data.subsets[0];
    const CascadeClassifierImpl::Data::Stump* cascadeStumps = &cascade.data.stumps[0];
    const CascadeClassifierImpl::Data::Stage* cascadeStages = &cascade.data.stages[0];

    double tmp = 0;
    for( int si = 0; si < nstages; si++ )
    {
        const CascadeClassifierImpl::Data::Stage& stage = cascadeStages[si];
        int wi, ntrees = stage.ntrees;
        tmp = 0;

        for( wi = 0; wi < ntrees; wi++ )
        {
            const CascadeClassifierImpl::Data::Stump& stump = cascadeStumps[wi];
            int c = featureEvaluator(stump.featureIdx);
            const int* subset = &cascadeSubsets[wi*subsetSize];
            tmp += (subset[c>>5] & (1 << (c & 31))) ? stump.left : stump.right;
        }

        if( tmp < stage.threshold )
        {
            sum = tmp;
            return -si;
        }

        cascadeStumps += ntrees;
        cascadeSubsets += ntrees*subsetSize;
    }

    sum = (double)tmp;
    return 1;
}
}
