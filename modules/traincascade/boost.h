#ifndef _OPENCV_BOOST_H_
#define _OPENCV_BOOST_H_

#include "traincascade_features.h"
#include "ml.h"

struct CvCascadeBoostParams : CvBoostParams
{
    float minHitRate;
    float maxFalseAlarm;
    
    CvCascadeBoostParams();
    CvCascadeBoostParams( int _boostType, float _minHitRate, float _maxFalseAlarm,
                          double _weightTrimRate, int _maxDepth, int _maxWeakCount );
    virtual ~CvCascadeBoostParams() {}
    void write( FileStorage &fs ) const;
    bool read( const FileNode &node );
    virtual void printDefaults() const;
    virtual void printAttrs() const;
    virtual bool scanAttr( const String prmName, const String val);
};

struct CvCascadeBoostTrainData : CvDTreeTrainData
{
    CvCascadeBoostTrainData( const CvFeatureEvaluator* _featureEvaluator,
                             const CvDTreeParams& _params );
    CvCascadeBoostTrainData( const CvFeatureEvaluator* _featureEvaluator,
                             int _numSamples, int _precalcValBufSize, int _precalcIdxBufSize,
                             const CvDTreeParams& _params = CvDTreeParams() );
    virtual void setData( const CvFeatureEvaluator* _featureEvaluator,
                          int _numSamples, int _precalcValBufSize, int _precalcIdxBufSize,
                          const CvDTreeParams& _params=CvDTreeParams() );
    void precalculate();

    virtual const int* get_class_labels( CvDTreeNode* n, int* labelsBuf );
    virtual const int* get_cv_labels( CvDTreeNode* n, int* labelsBuf);
    virtual const int* get_sample_indices( CvDTreeNode* n, int* indicesBuf );
    
    virtual void get_ord_var_data( CvDTreeNode* n, int vi, float* ordValuesBuf, int* sortedIndicesBuf,
                                  const float** ordValues, const int** sortedIndices, int* sampleIndicesBuf );
    virtual const int* get_cat_var_data( CvDTreeNode* n, int vi, int* catValuesBuf );
    virtual float getVarValue( int vi, int si );
    virtual void free_train_data();

    const CvFeatureEvaluator* featureEvaluator;
    Mat valCache; // precalculated feature values (CV_32FC1)
    CvMat _resp; // for casting
    int numPrecalcVal, numPrecalcIdx;
};

class CvCascadeBoostTree : public CvBoostTree
{
public:
    virtual CvDTreeNode* predict( int sampleIdx ) const;
    void write( FileStorage &fs, const Mat& featureMap );
    void read( const FileNode &node, CvBoost* _ensemble, CvDTreeTrainData* _data );
    void markFeaturesInMap( Mat& featureMap );
protected:
    virtual void split_node_data( CvDTreeNode* n );
};

class CvCascadeBoost : public CvBoost
{
public:
    virtual bool train( const CvFeatureEvaluator* _featureEvaluator,
                        int _numSamples, int _precalcValBufSize, int _precalcIdxBufSize,
                        const CvCascadeBoostParams& _params=CvCascadeBoostParams() );
    virtual float predict( int sampleIdx, bool returnSum = false ) const;

    float getThreshold() const { return threshold; }; 
    void write( FileStorage &fs, const Mat& featureMap ) const;
    bool read( const FileNode &node, const CvFeatureEvaluator* _featureEvaluator,
               const CvCascadeBoostParams& _params );
    void markUsedFeaturesInMap( Mat& featureMap );
protected:
    virtual bool set_params( const CvBoostParams& _params );
    virtual void update_weights( CvBoostTree* tree );
    virtual bool isErrDesired();

    float threshold;
    float minHitRate, maxFalseAlarm;
};

#endif
