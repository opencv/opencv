#pragma once

namespace cv
{

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


//----------------------------------------------  HaarEvaluator ---------------------------------------
class HaarEvaluator : public FeatureEvaluator
{
public:
    struct Feature
    {
        Feature();

        float calc( int offset ) const;
        void updatePtrs( const Mat& sum );
        bool read( const FileNode& node );

        bool tilted;

        enum { RECT_NUM = 3 };

        struct
        {
            Rect r;
            float weight;
        } rect[RECT_NUM];

        const int* p[RECT_NUM][4];
    };

    HaarEvaluator();
    virtual ~HaarEvaluator();

    virtual bool read( const FileNode& node );
    virtual Ptr<FeatureEvaluator> clone() const;
    virtual int getFeatureType() const { return FeatureEvaluator::HAAR; }

    virtual bool setImage(const Mat&, Size origWinSize);
    virtual bool setWindow(Point pt);

    double operator()(int featureIdx) const
    { return featuresPtr[featureIdx].calc(offset) * varianceNormFactor; }
    virtual double calcOrd(int featureIdx) const
    { return (*this)(featureIdx); }

protected:
    Size origWinSize;
    Ptr<vector<Feature> > features;
    Feature* featuresPtr; // optimization
    bool hasTiltedFeatures;

    Mat sum0, sqsum0, tilted0;
    Mat sum, sqsum, tilted;

    Rect normrect;
    const int *p[4];
    const double *pq[4];

    int offset;
    double varianceNormFactor;
};

inline HaarEvaluator::Feature :: Feature()
{
    tilted = false;
    rect[0].r = rect[1].r = rect[2].r = Rect();
    rect[0].weight = rect[1].weight = rect[2].weight = 0;
    p[0][0] = p[0][1] = p[0][2] = p[0][3] =
        p[1][0] = p[1][1] = p[1][2] = p[1][3] =
        p[2][0] = p[2][1] = p[2][2] = p[2][3] = 0;
}

inline float HaarEvaluator::Feature :: calc( int _offset ) const
{
    float ret = rect[0].weight * CALC_SUM(p[0], _offset) + rect[1].weight * CALC_SUM(p[1], _offset);

    if( rect[2].weight != 0.0f )
        ret += rect[2].weight * CALC_SUM(p[2], _offset);

    return ret;
}

inline void HaarEvaluator::Feature :: updatePtrs( const Mat& _sum )
{
    const int* ptr = (const int*)_sum.data;
    size_t step = _sum.step/sizeof(ptr[0]);
    if (tilted)
    {
        CV_TILTED_PTRS( p[0][0], p[0][1], p[0][2], p[0][3], ptr, rect[0].r, step );
        CV_TILTED_PTRS( p[1][0], p[1][1], p[1][2], p[1][3], ptr, rect[1].r, step );
        if (rect[2].weight)
            CV_TILTED_PTRS( p[2][0], p[2][1], p[2][2], p[2][3], ptr, rect[2].r, step );
    }
    else
    {
        CV_SUM_PTRS( p[0][0], p[0][1], p[0][2], p[0][3], ptr, rect[0].r, step );
        CV_SUM_PTRS( p[1][0], p[1][1], p[1][2], p[1][3], ptr, rect[1].r, step );
        if (rect[2].weight)
            CV_SUM_PTRS( p[2][0], p[2][1], p[2][2], p[2][3], ptr, rect[2].r, step );
    }
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

        int calc( int offset ) const;
        void updatePtrs( const Mat& sum );
        bool read(const FileNode& node );

        Rect rect; // weight and height for block
        const int* p[16]; // fast
    };

    LBPEvaluator();
    virtual ~LBPEvaluator();

    virtual bool read( const FileNode& node );
    virtual Ptr<FeatureEvaluator> clone() const;
    virtual int getFeatureType() const { return FeatureEvaluator::LBP; }

    virtual bool setImage(const Mat& image, Size _origWinSize);
    virtual bool setWindow(Point pt);

    int operator()(int featureIdx) const
    { return featuresPtr[featureIdx].calc(offset); }
    virtual int calcCat(int featureIdx) const
    { return (*this)(featureIdx); }
protected:
    Size origWinSize;
    Ptr<vector<Feature> > features;
    Feature* featuresPtr; // optimization
    Mat sum0, sum;
    Rect normrect;

    int offset;
};


inline LBPEvaluator::Feature :: Feature()
{
    rect = Rect();
    for( int i = 0; i < 16; i++ )
        p[i] = 0;
}

inline int LBPEvaluator::Feature :: calc( int _offset ) const
{
    int cval = CALC_SUM_( p[5], p[6], p[9], p[10], _offset );

    return (CALC_SUM_( p[0], p[1], p[4], p[5], _offset ) >= cval ? 128 : 0) |   // 0
           (CALC_SUM_( p[1], p[2], p[5], p[6], _offset ) >= cval ? 64 : 0) |    // 1
           (CALC_SUM_( p[2], p[3], p[6], p[7], _offset ) >= cval ? 32 : 0) |    // 2
           (CALC_SUM_( p[6], p[7], p[10], p[11], _offset ) >= cval ? 16 : 0) |  // 5
           (CALC_SUM_( p[10], p[11], p[14], p[15], _offset ) >= cval ? 8 : 0)|  // 8
           (CALC_SUM_( p[9], p[10], p[13], p[14], _offset ) >= cval ? 4 : 0)|   // 7
           (CALC_SUM_( p[8], p[9], p[12], p[13], _offset ) >= cval ? 2 : 0)|    // 6
           (CALC_SUM_( p[4], p[5], p[8], p[9], _offset ) >= cval ? 1 : 0);
}

inline void LBPEvaluator::Feature :: updatePtrs( const Mat& _sum )
{
    const int* ptr = (const int*)_sum.data;
    size_t step = _sum.step/sizeof(ptr[0]);
    Rect tr = rect;
    CV_SUM_PTRS( p[0], p[1], p[4], p[5], ptr, tr, step );
    tr.x += 2*rect.width;
    CV_SUM_PTRS( p[2], p[3], p[6], p[7], ptr, tr, step );
    tr.y += 2*rect.height;
    CV_SUM_PTRS( p[10], p[11], p[14], p[15], ptr, tr, step );
    tr.x -= 2*rect.width;
    CV_SUM_PTRS( p[8], p[9], p[12], p[13], ptr, tr, step );
}

//---------------------------------------------- HOGEvaluator -------------------------------------------

class HOGEvaluator : public FeatureEvaluator
{
public:
    struct Feature
    {
        Feature();
        float calc( int offset ) const;
        void updatePtrs( const vector<Mat>& _hist, const Mat &_normSum );
        bool read( const FileNode& node );

        enum { CELL_NUM = 4, BIN_NUM = 9 };

        Rect rect[CELL_NUM];
        int featComponent; //component index from 0 to 35
        const float* pF[4]; //for feature calculation
        const float* pN[4]; //for normalization calculation
    };
    HOGEvaluator();
    virtual ~HOGEvaluator();
    virtual bool read( const FileNode& node );
    virtual Ptr<FeatureEvaluator> clone() const;
    virtual int getFeatureType() const { return FeatureEvaluator::HOG; }
    virtual bool setImage( const Mat& image, Size winSize );
    virtual bool setWindow( Point pt );
    double operator()(int featureIdx) const
    {
        return featuresPtr[featureIdx].calc(offset);
    }
    virtual double calcOrd( int featureIdx ) const
    {
        return (*this)(featureIdx);
    }

private:
    virtual void integralHistogram( const Mat& srcImage, vector<Mat> &histogram, Mat &norm, int nbins ) const;

    Size origWinSize;
    Ptr<vector<Feature> > features;
    Feature* featuresPtr;
    vector<Mat> hist;
    Mat normSum;
    int offset;
};

inline HOGEvaluator::Feature :: Feature()
{
    rect[0] = rect[1] = rect[2] = rect[3] = Rect();
    pF[0] = pF[1] = pF[2] = pF[3] = 0;
    pN[0] = pN[1] = pN[2] = pN[3] = 0;
    featComponent = 0;
}

inline float HOGEvaluator::Feature :: calc( int _offset ) const
{
    float res = CALC_SUM(pF, _offset);
    float normFactor = CALC_SUM(pN, _offset);
    res = (res > 0.001f) ? (res / ( normFactor + 0.001f) ) : 0.f;
    return res;
}

inline void HOGEvaluator::Feature :: updatePtrs( const vector<Mat> &_hist, const Mat &_normSum )
{
    int binIdx = featComponent % BIN_NUM;
    int cellIdx = featComponent / BIN_NUM;
    Rect normRect = Rect( rect[0].x, rect[0].y, 2*rect[0].width, 2*rect[0].height );

    const float* featBuf = (const float*)_hist[binIdx].data;
    size_t featStep = _hist[0].step / sizeof(featBuf[0]);

    const float* normBuf = (const float*)_normSum.data;
    size_t normStep = _normSum.step / sizeof(normBuf[0]);

    CV_SUM_PTRS( pF[0], pF[1], pF[2], pF[3], featBuf, rect[cellIdx], featStep );
    CV_SUM_PTRS( pN[0], pN[1], pN[2], pN[3], normBuf, normRect, normStep );
}




//----------------------------------------------  predictor functions -------------------------------------

template<class FEval>
inline int predictOrdered( CascadeClassifier& cascade, Ptr<FeatureEvaluator> &_featureEvaluator, double& sum )
{
    int nstages = (int)cascade.data.stages.size();
    int nodeOfs = 0, leafOfs = 0;
    FEval& featureEvaluator = (FEval&)*_featureEvaluator;
    float* cascadeLeaves = &cascade.data.leaves[0];
    CascadeClassifier::Data::DTreeNode* cascadeNodes = &cascade.data.nodes[0];
    CascadeClassifier::Data::DTree* cascadeWeaks = &cascade.data.classifiers[0];
    CascadeClassifier::Data::Stage* cascadeStages = &cascade.data.stages[0];

    for( int si = 0; si < nstages; si++ )
    {
        CascadeClassifier::Data::Stage& stage = cascadeStages[si];
        int wi, ntrees = stage.ntrees;
        sum = 0;

        for( wi = 0; wi < ntrees; wi++ )
        {
            CascadeClassifier::Data::DTree& weak = cascadeWeaks[stage.first + wi];
            int idx = 0, root = nodeOfs;

            do
            {
                CascadeClassifier::Data::DTreeNode& node = cascadeNodes[root + idx];
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
inline int predictCategorical( CascadeClassifier& cascade, Ptr<FeatureEvaluator> &_featureEvaluator, double& sum )
{
    int nstages = (int)cascade.data.stages.size();
    int nodeOfs = 0, leafOfs = 0;
    FEval& featureEvaluator = (FEval&)*_featureEvaluator;
    size_t subsetSize = (cascade.data.ncategories + 31)/32;
    int* cascadeSubsets = &cascade.data.subsets[0];
    float* cascadeLeaves = &cascade.data.leaves[0];
    CascadeClassifier::Data::DTreeNode* cascadeNodes = &cascade.data.nodes[0];
    CascadeClassifier::Data::DTree* cascadeWeaks = &cascade.data.classifiers[0];
    CascadeClassifier::Data::Stage* cascadeStages = &cascade.data.stages[0];

    for(int si = 0; si < nstages; si++ )
    {
        CascadeClassifier::Data::Stage& stage = cascadeStages[si];
        int wi, ntrees = stage.ntrees;
        sum = 0;

        for( wi = 0; wi < ntrees; wi++ )
        {
            CascadeClassifier::Data::DTree& weak = cascadeWeaks[stage.first + wi];
            int idx = 0, root = nodeOfs;
            do
            {
                CascadeClassifier::Data::DTreeNode& node = cascadeNodes[root + idx];
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
inline int predictOrderedStump( CascadeClassifier& cascade, Ptr<FeatureEvaluator> &_featureEvaluator, double& sum )
{
    int nodeOfs = 0, leafOfs = 0;
    FEval& featureEvaluator = (FEval&)*_featureEvaluator;
    float* cascadeLeaves = &cascade.data.leaves[0];
    CascadeClassifier::Data::DTreeNode* cascadeNodes = &cascade.data.nodes[0];
    CascadeClassifier::Data::Stage* cascadeStages = &cascade.data.stages[0];

    int nstages = (int)cascade.data.stages.size();
    for( int stageIdx = 0; stageIdx < nstages; stageIdx++ )
    {
        CascadeClassifier::Data::Stage& stage = cascadeStages[stageIdx];
        sum = 0.0;

        int ntrees = stage.ntrees;
        for( int i = 0; i < ntrees; i++, nodeOfs++, leafOfs+= 2 )
        {
            CascadeClassifier::Data::DTreeNode& node = cascadeNodes[nodeOfs];
            double value = featureEvaluator(node.featureIdx);
            sum += cascadeLeaves[ value < node.threshold ? leafOfs : leafOfs + 1 ];
        }

        if( sum < stage.threshold )
            return -stageIdx;
    }

    return 1;
}

template<class FEval>
inline int predictCategoricalStump( CascadeClassifier& cascade, Ptr<FeatureEvaluator> &_featureEvaluator, double& sum )
{
    int nstages = (int)cascade.data.stages.size();
    int nodeOfs = 0, leafOfs = 0;
    FEval& featureEvaluator = (FEval&)*_featureEvaluator;
    size_t subsetSize = (cascade.data.ncategories + 31)/32;
    int* cascadeSubsets = &cascade.data.subsets[0];
    float* cascadeLeaves = &cascade.data.leaves[0];
    CascadeClassifier::Data::DTreeNode* cascadeNodes = &cascade.data.nodes[0];
    CascadeClassifier::Data::Stage* cascadeStages = &cascade.data.stages[0];

#ifdef HAVE_TEGRA_OPTIMIZATION
    float tmp = 0; // float accumulator -- float operations are quicker
#endif
    for( int si = 0; si < nstages; si++ )
    {
        CascadeClassifier::Data::Stage& stage = cascadeStages[si];
        int wi, ntrees = stage.ntrees;
#ifdef HAVE_TEGRA_OPTIMIZATION
        tmp = 0;
#else
        sum = 0;
#endif

        for( wi = 0; wi < ntrees; wi++ )
        {
            CascadeClassifier::Data::DTreeNode& node = cascadeNodes[nodeOfs];
            int c = featureEvaluator(node.featureIdx);
            const int* subset = &cascadeSubsets[nodeOfs*subsetSize];
#ifdef HAVE_TEGRA_OPTIMIZATION
            tmp += cascadeLeaves[ subset[c>>5] & (1 << (c & 31)) ? leafOfs : leafOfs+1];
#else
            sum += cascadeLeaves[ subset[c>>5] & (1 << (c & 31)) ? leafOfs : leafOfs+1];
#endif
            nodeOfs++;
            leafOfs += 2;
        }
#ifdef HAVE_TEGRA_OPTIMIZATION
        if( tmp < stage.threshold ) {
            sum = (double)tmp;
            return -si;
        }
#else
        if( sum < stage.threshold )
            return -si;
#endif
    }

#ifdef HAVE_TEGRA_OPTIMIZATION
    sum = (double)tmp;
#endif

    return 1;
}
}

