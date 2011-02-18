#include "boost.h"
#include "cascadeclassifier.h"
#include <queue>
#include "cxmisc.h"

using namespace std;

static inline double
logRatio( double val )
{
    const double eps = 1e-5;

    val = max( val, eps );
    val = min( val, 1. - eps );
    return log( val/(1. - val) );
}

#define CV_CMP_FLT(i,j) (i < j)
static CV_IMPLEMENT_QSORT_EX( icvSortFlt, float, CV_CMP_FLT, const float* )

#define CV_CMP_NUM_IDX(i,j) (aux[i] < aux[j])
static CV_IMPLEMENT_QSORT_EX( icvSortIntAux, int, CV_CMP_NUM_IDX, const float* )
static CV_IMPLEMENT_QSORT_EX( icvSortUShAux, unsigned short, CV_CMP_NUM_IDX, const float* )

#define CV_THRESHOLD_EPS (0.00001F)

static const int MinBlockSize = 1 << 16;
static const int BlockSizeDelta = 1 << 10;

//----------------------------- CascadeBoostParams -------------------------------------------------

CvCascadeBoostParams::CvCascadeBoostParams() : minHitRate( 0.995F), maxFalseAlarm( 0.5F )
{  
    boost_type = CvBoost::GENTLE;
    use_surrogates = use_1se_rule = truncate_pruned_tree = false;
}

CvCascadeBoostParams::CvCascadeBoostParams( int _boostType,
        float _minHitRate, float _maxFalseAlarm,
        double _weightTrimRate, int _maxDepth, int _maxWeakCount ) :
    CvBoostParams( _boostType, _maxWeakCount, _weightTrimRate, _maxDepth, false, 0 )
{
    boost_type = CvBoost::GENTLE;
    minHitRate = _minHitRate;
    maxFalseAlarm = _maxFalseAlarm;
    use_surrogates = use_1se_rule = truncate_pruned_tree = false;
}

void CvCascadeBoostParams::write( FileStorage &fs ) const
{
    String boostTypeStr = boost_type == CvBoost::DISCRETE ? CC_DISCRETE_BOOST : 
                          boost_type == CvBoost::REAL ? CC_REAL_BOOST :
                          boost_type == CvBoost::LOGIT ? CC_LOGIT_BOOST :
                          boost_type == CvBoost::GENTLE ? CC_GENTLE_BOOST : String();
    CV_Assert( !boostTypeStr.empty() );
    fs << CC_BOOST_TYPE << boostTypeStr;
    fs << CC_MINHITRATE << minHitRate;
    fs << CC_MAXFALSEALARM << maxFalseAlarm;
    fs << CC_TRIM_RATE << weight_trim_rate;
    fs << CC_MAX_DEPTH << max_depth;
    fs << CC_WEAK_COUNT << weak_count;
}

bool CvCascadeBoostParams::read( const FileNode &node )
{
    String boostTypeStr;
    FileNode rnode = node[CC_BOOST_TYPE];
    rnode >> boostTypeStr;
    boost_type = !boostTypeStr.compare( CC_DISCRETE_BOOST ) ? CvBoost::DISCRETE :
                 !boostTypeStr.compare( CC_REAL_BOOST ) ? CvBoost::REAL :
                 !boostTypeStr.compare( CC_LOGIT_BOOST ) ? CvBoost::LOGIT :
                 !boostTypeStr.compare( CC_GENTLE_BOOST ) ? CvBoost::GENTLE : -1;
    if (boost_type == -1)
        CV_Error( CV_StsBadArg, "unsupported Boost type" );
    node[CC_MINHITRATE] >> minHitRate;
    node[CC_MAXFALSEALARM] >> maxFalseAlarm;
    node[CC_TRIM_RATE] >> weight_trim_rate ;
    node[CC_MAX_DEPTH] >> max_depth ;
    node[CC_WEAK_COUNT] >> weak_count ;
    if ( minHitRate <= 0 || minHitRate > 1 ||
         maxFalseAlarm <= 0 || maxFalseAlarm > 1 ||
         weight_trim_rate <= 0 || weight_trim_rate > 1 ||
         max_depth <= 0 || weak_count <= 0 )
        CV_Error( CV_StsBadArg, "bad parameters range");
    return true;
}

void CvCascadeBoostParams::printDefaults() const
{
    cout << "--boostParams--" << endl;
    cout << "  [-bt <{" << CC_DISCRETE_BOOST << ", " 
                        << CC_REAL_BOOST << ", "
                        << CC_LOGIT_BOOST ", "
                        << CC_GENTLE_BOOST << "(default)}>]" << endl;
    cout << "  [-minHitRate <min_hit_rate> = " << minHitRate << ">]" << endl;
    cout << "  [-maxFalseAlarmRate <max_false_alarm_rate = " << maxFalseAlarm << ">]" << endl;
    cout << "  [-weightTrimRate <weight_trim_rate = " << weight_trim_rate << ">]" << endl;
    cout << "  [-maxDepth <max_depth_of_weak_tree = " << max_depth << ">]" << endl;
    cout << "  [-maxWeakCount <max_weak_tree_count = " << weak_count << ">]" << endl;
}

void CvCascadeBoostParams::printAttrs() const
{
    String boostTypeStr = boost_type == CvBoost::DISCRETE ? CC_DISCRETE_BOOST : 
                          boost_type == CvBoost::REAL ? CC_REAL_BOOST :
                          boost_type == CvBoost::LOGIT  ? CC_LOGIT_BOOST :
                          boost_type == CvBoost::GENTLE ? CC_GENTLE_BOOST : String();
    CV_Assert( !boostTypeStr.empty() );
    cout << "boostType: " << boostTypeStr << endl;
    cout << "minHitRate: " << minHitRate << endl;
    cout << "maxFalseAlarmRate: " <<  maxFalseAlarm << endl;
    cout << "weightTrimRate: " << weight_trim_rate << endl;
    cout << "maxDepth: " << max_depth << endl;
    cout << "maxWeakCount: " << weak_count << endl;
}

bool CvCascadeBoostParams::scanAttr( const String prmName, const String val)
{
    bool res = true;

    if( !prmName.compare( "-bt" ) )
    {
        boost_type = !val.compare( CC_DISCRETE_BOOST ) ? CvBoost::DISCRETE :
                     !val.compare( CC_REAL_BOOST ) ? CvBoost::REAL :
                     !val.compare( CC_LOGIT_BOOST ) ? CvBoost::LOGIT :
                     !val.compare( CC_GENTLE_BOOST ) ? CvBoost::GENTLE : -1;
        if (boost_type == -1)
            res = false;
    }
    else if( !prmName.compare( "-minHitRate" ) )
    {
        minHitRate = (float) atof( val.c_str() );
    }
    else if( !prmName.compare( "-maxFalseAlarmRate" ) )
    {
        maxFalseAlarm = (float) atof( val.c_str() );
    }
    else if( !prmName.compare( "-weightTrimRate" ) )
    {
        weight_trim_rate = (float) atof( val.c_str() );
    }
    else if( !prmName.compare( "-maxDepth" ) )
    {
        max_depth = atoi( val.c_str() );
    }
    else if( !prmName.compare( "-maxWeakCount" ) )
    {
        weak_count = atoi( val.c_str() );
    }
    else
        res = false;

    return res;        
}

//---------------------------- CascadeBoostTrainData -----------------------------

CvCascadeBoostTrainData::CvCascadeBoostTrainData( const CvFeatureEvaluator* _featureEvaluator,
                                                  const CvDTreeParams& _params )
{
    is_classifier = true;
    var_all = var_count = (int)_featureEvaluator->getNumFeatures();

    featureEvaluator = _featureEvaluator;
    shared = true;
    set_params( _params );
    max_c_count = MAX( 2, featureEvaluator->getMaxCatCount() );
    var_type = cvCreateMat( 1, var_count + 2, CV_32SC1 );
    if ( featureEvaluator->getMaxCatCount() > 0 ) 
    {
        numPrecalcIdx = 0;
        cat_var_count = var_count;
        ord_var_count = 0;
        for( int vi = 0; vi < var_count; vi++ )
        {
            var_type->data.i[vi] = vi;
        }    
    }
    else
    {
        cat_var_count = 0;
        ord_var_count = var_count;
        for( int vi = 1; vi <= var_count; vi++ )
        {
            var_type->data.i[vi-1] = -vi;
        }        
    }    
    var_type->data.i[var_count] = cat_var_count;
    var_type->data.i[var_count+1] = cat_var_count+1;

    int maxSplitSize = cvAlign(sizeof(CvDTreeSplit) + (MAX(0,max_c_count - 33)/32)*sizeof(int),sizeof(void*));
    int treeBlockSize = MAX((int)sizeof(CvDTreeNode)*8, maxSplitSize);
    treeBlockSize = MAX(treeBlockSize + BlockSizeDelta, MinBlockSize);
    tree_storage = cvCreateMemStorage( treeBlockSize );
    node_heap = cvCreateSet( 0, sizeof(node_heap[0]), sizeof(CvDTreeNode), tree_storage );
    split_heap = cvCreateSet( 0, sizeof(split_heap[0]), maxSplitSize, tree_storage );  
}

CvCascadeBoostTrainData::CvCascadeBoostTrainData( const CvFeatureEvaluator* _featureEvaluator,
                                                 int _numSamples,
                                                 int _precalcValBufSize, int _precalcIdxBufSize,
                                                 const CvDTreeParams& _params )
{
    setData( _featureEvaluator, _numSamples, _precalcValBufSize, _precalcIdxBufSize, _params );
}
 
void CvCascadeBoostTrainData::setData( const CvFeatureEvaluator* _featureEvaluator,
                                      int _numSamples,
                                      int _precalcValBufSize, int _precalcIdxBufSize,
									  const CvDTreeParams& _params )
{    
    int* idst = 0;
    unsigned short* udst = 0;
        
    clear();
    shared = true;
    have_labels = true;
    have_priors = false;
    is_classifier = true;

    rng = &cv::theRNG();

    set_params( _params );

    CV_Assert( _featureEvaluator );
    featureEvaluator = _featureEvaluator;

    max_c_count = MAX( 2, featureEvaluator->getMaxCatCount() );
    _resp = featureEvaluator->getCls();
    responses = &_resp;
    // TODO: check responses: elements must be 0 or 1
    
	if( _precalcValBufSize < 0 || _precalcIdxBufSize < 0)
        CV_Error( CV_StsOutOfRange, "_numPrecalcVal and _numPrecalcIdx must be positive or 0" );

	var_count = var_all = featureEvaluator->getNumFeatures();
    sample_count = _numSamples;
    
    is_buf_16u = false;     
    if (sample_count < 65536) 
        is_buf_16u = true; 

    numPrecalcVal = min( cvRound((double)_precalcValBufSize*1048576. / (sizeof(float)*sample_count)), var_count );
    numPrecalcIdx = min( cvRound((double)_precalcIdxBufSize*1048576. /
                ((is_buf_16u ? sizeof(unsigned short) : sizeof (int))*sample_count)), var_count );

    assert( numPrecalcIdx >= 0 && numPrecalcVal >= 0 );

    valCache.create( numPrecalcVal, sample_count, CV_32FC1 );
    var_type = cvCreateMat( 1, var_count + 2, CV_32SC1 );
    
    if ( featureEvaluator->getMaxCatCount() > 0 ) 
    {
        numPrecalcIdx = 0;
        cat_var_count = var_count;
        ord_var_count = 0;
        for( int vi = 0; vi < var_count; vi++ )
        {
            var_type->data.i[vi] = vi;
        }    
    }
    else
    {
        cat_var_count = 0;
        ord_var_count = var_count;
        for( int vi = 1; vi <= var_count; vi++ )
        {
            var_type->data.i[vi-1] = -vi;
        }        
    }
    var_type->data.i[var_count] = cat_var_count;
    var_type->data.i[var_count+1] = cat_var_count+1;
    work_var_count = ( cat_var_count ? 0 : numPrecalcIdx ) + 1;
    buf_size = (work_var_count + 1) * sample_count;
    buf_count = 2;
    
    if ( is_buf_16u )
        buf = cvCreateMat( buf_count, buf_size, CV_16UC1 );
    else
        buf = cvCreateMat( buf_count, buf_size, CV_32SC1 );

    cat_count = cvCreateMat( 1, cat_var_count + 1, CV_32SC1 );

    // precalculate valCache and set indices in buf
	precalculate();

    // now calculate the maximum size of split,
    // create memory storage that will keep nodes and splits of the decision tree
    // allocate root node and the buffer for the whole training data
    int maxSplitSize = cvAlign(sizeof(CvDTreeSplit) +
        (MAX(0,sample_count - 33)/32)*sizeof(int),sizeof(void*));
    int treeBlockSize = MAX((int)sizeof(CvDTreeNode)*8, maxSplitSize);
    treeBlockSize = MAX(treeBlockSize + BlockSizeDelta, MinBlockSize);
    tree_storage = cvCreateMemStorage( treeBlockSize );
    node_heap = cvCreateSet( 0, sizeof(*node_heap), sizeof(CvDTreeNode), tree_storage );

    int nvSize = var_count*sizeof(int);
    nvSize = cvAlign(MAX( nvSize, (int)sizeof(CvSetElem) ), sizeof(void*));
    int tempBlockSize = nvSize;
    tempBlockSize = MAX( tempBlockSize + BlockSizeDelta, MinBlockSize );
    temp_storage = cvCreateMemStorage( tempBlockSize );
    nv_heap = cvCreateSet( 0, sizeof(*nv_heap), nvSize, temp_storage );
    
    data_root = new_node( 0, sample_count, 0, 0 );

    // set sample labels
    if (is_buf_16u)
        udst = (unsigned short*)(buf->data.s + work_var_count*sample_count);
    else
        idst = buf->data.i + work_var_count*sample_count;

    for (int si = 0; si < sample_count; si++)
    {
        if (udst)
            udst[si] = (unsigned short)si;
        else
            idst[si] = si;
    }
    for( int vi = 0; vi < var_count; vi++ )
        data_root->set_num_valid(vi, sample_count);
    for( int vi = 0; vi < cat_var_count; vi++ )
        cat_count->data.i[vi] = max_c_count;

    cat_count->data.i[cat_var_count] = 2;

    maxSplitSize = cvAlign(sizeof(CvDTreeSplit) +
        (MAX(0,max_c_count - 33)/32)*sizeof(int),sizeof(void*));
    split_heap = cvCreateSet( 0, sizeof(*split_heap), maxSplitSize, tree_storage );

    priors = cvCreateMat( 1, get_num_classes(), CV_64F );
    cvSet(priors, cvScalar(1));
    priors_mult = cvCloneMat( priors );
    counts = cvCreateMat( 1, get_num_classes(), CV_32SC1 );
    direction = cvCreateMat( 1, sample_count, CV_8UC1 );
    split_buf = cvCreateMat( 1, sample_count, CV_32SC1 );
}

void CvCascadeBoostTrainData::free_train_data()
{
    CvDTreeTrainData::free_train_data();
    valCache.release();
}

const int* CvCascadeBoostTrainData::get_class_labels( CvDTreeNode* n, int* labelsBuf)
{
    int nodeSampleCount = n->sample_count; 
    int rStep = CV_IS_MAT_CONT( responses->type ) ? 1 : responses->step / CV_ELEM_SIZE( responses->type );

    int* sampleIndicesBuf = labelsBuf; //
    const int* sampleIndices = get_sample_indices(n, sampleIndicesBuf);
    for( int si = 0; si < nodeSampleCount; si++ )
    {
        int sidx = sampleIndices[si];
        labelsBuf[si] = (int)responses->data.fl[sidx*rStep];
    }    
    return labelsBuf;
}

const int* CvCascadeBoostTrainData::get_sample_indices( CvDTreeNode* n, int* indicesBuf )
{
    return CvDTreeTrainData::get_cat_var_data( n, get_work_var_count(), indicesBuf );
}

const int* CvCascadeBoostTrainData::get_cv_labels( CvDTreeNode* n, int* labels_buf )
{
    return CvDTreeTrainData::get_cat_var_data( n, get_work_var_count() - 1, labels_buf );
}

void CvCascadeBoostTrainData::get_ord_var_data( CvDTreeNode* n, int vi, float* ordValuesBuf, int* sortedIndicesBuf,
        const float** ordValues, const int** sortedIndices, int* sampleIndicesBuf )
{
    int nodeSampleCount = n->sample_count; 
    const int* sampleIndices = get_sample_indices(n, sampleIndicesBuf);
    
	if ( vi < numPrecalcIdx )
	{
		if( !is_buf_16u )
            *sortedIndices = buf->data.i + n->buf_idx*buf->cols + vi*sample_count + n->offset;
	    else 
        {
	        const unsigned short* shortIndices = (const unsigned short*)(buf->data.s + n->buf_idx*buf->cols + 
	                                              vi*sample_count + n->offset );
	        for( int i = 0; i < nodeSampleCount; i++ )
                sortedIndicesBuf[i] = shortIndices[i];
            *sortedIndices = sortedIndicesBuf;
	    }
		
		if ( vi < numPrecalcVal )
		{
			for( int i = 0; i < nodeSampleCount; i++ )
	        {
                int idx = (*sortedIndices)[i];
	            idx = sampleIndices[idx];
	            ordValuesBuf[i] =  valCache.at<float>( vi, idx);
	        }
		}
		else
		{
            for( int i = 0; i < nodeSampleCount; i++ )
	        {
                int idx = (*sortedIndices)[i];
	            idx = sampleIndices[idx];
				ordValuesBuf[i] = (*featureEvaluator)( vi, idx);
			}
		}
    }
	else // vi >= numPrecalcIdx
	{
        vector<float> sampleValuesBuf;
        float* sampleValues = 0;

        if( sizeof(float) == sizeof(int) )
        {
            // use sampleIndices as temporary buffer for values
            sampleValues = (float*)sampleIndices;
        }
        else
        {
            sampleValuesBuf.resize(nodeSampleCount);
            sampleValues = &sampleValuesBuf[0];
        }

		if ( vi < numPrecalcVal )
		{
			for( int i = 0; i < nodeSampleCount; i++ )
	        {
                sortedIndicesBuf[i] = i;
                sampleValues[i] = valCache.at<float>( vi, sampleIndices[i] );
	        }
		}
		else
		{
            for( int i = 0; i < nodeSampleCount; i++ )
	        {
                sortedIndicesBuf[i] = i;
                sampleValues[i] = (*featureEvaluator)( vi, sampleIndices[i]);
			}
		}
        icvSortIntAux( sortedIndicesBuf, nodeSampleCount, &sampleValues[0] );
		for( int i = 0; i < nodeSampleCount; i++ )
            ordValuesBuf[i] = (&sampleValues[0])[sortedIndicesBuf[i]];
        *sortedIndices = sortedIndicesBuf;
	}
	
    *ordValues = ordValuesBuf;
}
 
const int* CvCascadeBoostTrainData::get_cat_var_data( CvDTreeNode* n, int vi, int* catValuesBuf )
{
    int nodeSampleCount = n->sample_count;
    int* sampleIndicesBuf = catValuesBuf; //
    const int* sampleIndices = get_sample_indices(n, sampleIndicesBuf);

    if ( vi < numPrecalcVal )
    {
        for( int i = 0; i < nodeSampleCount; i++ )
            catValuesBuf[i] = (int) valCache.at<float>( vi, sampleIndices[i]);
    }
    else
    {
        if( vi >= numPrecalcVal && vi < var_count )
        {
            for( int i = 0; i < nodeSampleCount; i++ )
                catValuesBuf[i] = (int)(*featureEvaluator)( vi, sampleIndices[i] );
        }
        else
        {
            get_cv_labels( n, catValuesBuf );
        }
    }

    return catValuesBuf;
}

float CvCascadeBoostTrainData::getVarValue( int vi, int si )
{
    if ( vi < numPrecalcVal && !valCache.empty() )
	    return valCache.at<float>( vi, si );
	return (*featureEvaluator)( vi, si );
}


struct FeatureIdxOnlyPrecalc
{
    FeatureIdxOnlyPrecalc( const CvFeatureEvaluator* _featureEvaluator, CvMat* _buf, int _sample_count, bool _is_buf_16u )
    {
        featureEvaluator = _featureEvaluator;
        sample_count = _sample_count;
        udst = (unsigned short*)_buf->data.s;
        idst = _buf->data.i;
        is_buf_16u = _is_buf_16u;
    }
    void operator()( const BlockedRange& range ) const
    {
        cv::AutoBuffer<float> valCache(sample_count);
        float* valCachePtr = (float*)valCache;
        for ( int fi = range.begin(); fi < range.end(); fi++)
        {
            for( int si = 0; si < sample_count; si++ )
            {
                valCachePtr[si] = (*featureEvaluator)( fi, si );
                if ( is_buf_16u )
                    *(udst + fi*sample_count + si) = (unsigned short)si;
                else
                    *(idst + fi*sample_count + si) = si;
            }
            if ( is_buf_16u )
                icvSortUShAux( udst + fi*sample_count, sample_count, valCachePtr );
            else
                icvSortIntAux( idst + fi*sample_count, sample_count, valCachePtr );
        }
    }
    const CvFeatureEvaluator* featureEvaluator;
    int sample_count;
    int* idst;
    unsigned short* udst;
    bool is_buf_16u;
};

struct FeatureValAndIdxPrecalc
{
    FeatureValAndIdxPrecalc( const CvFeatureEvaluator* _featureEvaluator, CvMat* _buf, Mat* _valCache, int _sample_count, bool _is_buf_16u )
    {
        featureEvaluator = _featureEvaluator;
        valCache = _valCache;
        sample_count = _sample_count;
        udst = (unsigned short*)_buf->data.s;
        idst = _buf->data.i;
        is_buf_16u = _is_buf_16u;
    }
    void operator()( const BlockedRange& range ) const
    {
        for ( int fi = range.begin(); fi < range.end(); fi++)
        {
            for( int si = 0; si < sample_count; si++ )
            {
                valCache->at<float>(fi,si) = (*featureEvaluator)( fi, si );
                if ( is_buf_16u )
                    *(udst + fi*sample_count + si) = (unsigned short)si;
                else
                    *(idst + fi*sample_count + si) = si;
            }
            if ( is_buf_16u )
                icvSortUShAux( udst + fi*sample_count, sample_count, valCache->ptr<float>(fi) );
            else
                icvSortIntAux( idst + fi*sample_count, sample_count, valCache->ptr<float>(fi) );
        }
    }
    const CvFeatureEvaluator* featureEvaluator;
    Mat* valCache;
    int sample_count;
    int* idst;
    unsigned short* udst;
    bool is_buf_16u;
};

struct FeatureValOnlyPrecalc
{
    FeatureValOnlyPrecalc( const CvFeatureEvaluator* _featureEvaluator, Mat* _valCache, int _sample_count )
    {
        featureEvaluator = _featureEvaluator;
        valCache = _valCache;
        sample_count = _sample_count;
    }
    void operator()( const BlockedRange& range ) const
    {
        for ( int fi = range.begin(); fi < range.end(); fi++)
            for( int si = 0; si < sample_count; si++ )
                valCache->at<float>(fi,si) = (*featureEvaluator)( fi, si );
    }
    const CvFeatureEvaluator* featureEvaluator;
    Mat* valCache;
    int sample_count;
};

void CvCascadeBoostTrainData::precalculate()
{
    int minNum = MIN( numPrecalcVal, numPrecalcIdx);

    double proctime = -TIME( 0 );
    parallel_for( BlockedRange(numPrecalcVal, numPrecalcIdx),
                  FeatureIdxOnlyPrecalc(featureEvaluator, buf, sample_count, is_buf_16u!=0) );
    parallel_for( BlockedRange(0, minNum),
                  FeatureValAndIdxPrecalc(featureEvaluator, buf, &valCache, sample_count, is_buf_16u!=0) );
    parallel_for( BlockedRange(minNum, numPrecalcVal),
                  FeatureValOnlyPrecalc(featureEvaluator, &valCache, sample_count) );
    cout << "Precalculation time: " << (proctime + TIME( 0 )) << endl;
}

//-------------------------------- CascadeBoostTree ----------------------------------------

CvDTreeNode* CvCascadeBoostTree::predict( int sampleIdx ) const
{
    CvDTreeNode* node = root;
    if( !node )
        CV_Error( CV_StsError, "The tree has not been trained yet" );
   
    if ( ((CvCascadeBoostTrainData*)data)->featureEvaluator->getMaxCatCount() == 0 ) // ordered
    {
        while( node->left )
        {
            CvDTreeSplit* split = node->split;
            float val = ((CvCascadeBoostTrainData*)data)->getVarValue( split->var_idx, sampleIdx );
            node = val <= split->ord.c ? node->left : node->right;
        }
    }
    else // categorical
    {
        while( node->left )
        {
            CvDTreeSplit* split = node->split;
            int c = (int)((CvCascadeBoostTrainData*)data)->getVarValue( split->var_idx, sampleIdx );
            node = CV_DTREE_CAT_DIR(c, split->subset) < 0 ? node->left : node->right;
        }
    }
    return node;
}

void CvCascadeBoostTree::write( FileStorage &fs, const Mat& featureMap )
{
    int maxCatCount = ((CvCascadeBoostTrainData*)data)->featureEvaluator->getMaxCatCount();
    int subsetN = (maxCatCount + 31)/32;
    queue<CvDTreeNode*> internalNodesQueue;
    int size = (int)pow( 2.f, (float)ensemble->get_params().max_depth);
    Ptr<float> leafVals = new float[size];
    int leafValIdx = 0;
    int internalNodeIdx = 1;
    CvDTreeNode* tempNode;

    CV_DbgAssert( root );
    internalNodesQueue.push( root );

    fs << "{";
    fs << CC_INTERNAL_NODES << "[:";
    while (!internalNodesQueue.empty())
    {
        tempNode = internalNodesQueue.front();
        CV_Assert( tempNode->left );
        if ( !tempNode->left->left && !tempNode->left->right) // left node is leaf
        {
            leafVals[-leafValIdx] = (float)tempNode->left->value;
            fs << leafValIdx-- ;
        }
        else
        {
            internalNodesQueue.push( tempNode->left );
            fs << internalNodeIdx++;
        }
        CV_Assert( tempNode->right );
        if ( !tempNode->right->left && !tempNode->right->right) // right node is leaf
        {
            leafVals[-leafValIdx] = (float)tempNode->right->value;
            fs << leafValIdx--;
        }
        else
        {
            internalNodesQueue.push( tempNode->right );
            fs << internalNodeIdx++;
        }
        int fidx = tempNode->split->var_idx;
        fidx = featureMap.empty() ? fidx : featureMap.at<int>(0, fidx);
        fs << fidx;
        if ( !maxCatCount )
            fs << tempNode->split->ord.c;
        else
            for( int i = 0; i < subsetN; i++ )
                fs << tempNode->split->subset[i];
        internalNodesQueue.pop();
    }
    fs << "]"; // CC_INTERNAL_NODES

    fs << CC_LEAF_VALUES << "[:";
    for (int ni = 0; ni < -leafValIdx; ni++)
        fs << leafVals[ni];
    fs << "]"; // CC_LEAF_VALUES
    fs << "}";
}

void CvCascadeBoostTree::read( const FileNode &node, CvBoost* _ensemble,
                                CvDTreeTrainData* _data )
{
    int maxCatCount = ((CvCascadeBoostTrainData*)_data)->featureEvaluator->getMaxCatCount();
    int subsetN = (maxCatCount + 31)/32;
    int step = 3 + ( maxCatCount>0 ? subsetN : 1 );
    
    queue<CvDTreeNode*> internalNodesQueue;
    FileNodeIterator internalNodesIt, leafValsuesIt;
    CvDTreeNode* prntNode, *cldNode;

    clear();
    data = _data;
    ensemble = _ensemble;
    pruned_tree_idx = 0;

    // read tree nodes
    FileNode rnode = node[CC_INTERNAL_NODES];
    internalNodesIt = rnode.end();
    leafValsuesIt = node[CC_LEAF_VALUES].end();
    internalNodesIt--; leafValsuesIt--;
    for( size_t i = 0; i < rnode.size()/step; i++ )
    {
        prntNode = data->new_node( 0, 0, 0, 0 );
        if ( maxCatCount > 0 )
        {
            prntNode->split = data->new_split_cat( 0, 0 );
            for( int j = subsetN-1; j>=0; j--)
            {
                *internalNodesIt >> prntNode->split->subset[j]; internalNodesIt--;
            }
        }
        else
        {
            float split_value;
            *internalNodesIt >> split_value; internalNodesIt--;
            prntNode->split = data->new_split_ord( 0, split_value, 0, 0, 0);
        }
        *internalNodesIt >> prntNode->split->var_idx; internalNodesIt--;
        int ridx, lidx;
        *internalNodesIt >> ridx; internalNodesIt--;
        *internalNodesIt >> lidx;internalNodesIt--;
        if ( ridx <= 0)
        {
            prntNode->right = cldNode = data->new_node( 0, 0, 0, 0 );
            *leafValsuesIt >> cldNode->value; leafValsuesIt--;
            cldNode->parent = prntNode;            
        }
        else
        {
            prntNode->right = internalNodesQueue.front(); 
            prntNode->right->parent = prntNode;
            internalNodesQueue.pop();
        }

        if ( lidx <= 0)
        {
            prntNode->left = cldNode = data->new_node( 0, 0, 0, 0 );
            *leafValsuesIt >> cldNode->value; leafValsuesIt--;
            cldNode->parent = prntNode;            
        }
        else
        {
            prntNode->left = internalNodesQueue.front();
            prntNode->left->parent = prntNode;
            internalNodesQueue.pop();
        }

        internalNodesQueue.push( prntNode );
    }

    root = internalNodesQueue.front();
    internalNodesQueue.pop();
}

void CvCascadeBoostTree::split_node_data( CvDTreeNode* node )
{
    int n = node->sample_count, nl, nr, scount = data->sample_count;
    char* dir = (char*)data->direction->data.ptr;
    CvDTreeNode *left = 0, *right = 0;
    int* newIdx = data->split_buf->data.i;
    int newBufIdx = data->get_child_buf_idx( node );
    int workVarCount = data->get_work_var_count();
    CvMat* buf = data->buf;
    cv::AutoBuffer<uchar> inn_buf(n*(3*sizeof(int)+sizeof(float)));
    int* tempBuf = (int*)(uchar*)inn_buf;
    bool splitInputData;

    complete_node_dir(node);

    for( int i = nl = nr = 0; i < n; i++ )
    {
        int d = dir[i];
        // initialize new indices for splitting ordered variables
        newIdx[i] = (nl & (d-1)) | (nr & -d); // d ? ri : li
        nr += d;
        nl += d^1;
    }

    node->left = left = data->new_node( node, nl, newBufIdx, node->offset );
    node->right = right = data->new_node( node, nr, newBufIdx, node->offset + nl );

    splitInputData = node->depth + 1 < data->params.max_depth &&
        (node->left->sample_count > data->params.min_sample_count ||
        node->right->sample_count > data->params.min_sample_count);

    // split ordered variables, keep both halves sorted.
    for( int vi = 0; vi < ((CvCascadeBoostTrainData*)data)->numPrecalcIdx; vi++ )
    {
        int ci = data->get_var_type(vi);
        if( ci >= 0 || !splitInputData )
            continue;

        int n1 = node->get_num_valid(vi);
        float *src_val_buf = (float*)(tempBuf + n);
        int *src_sorted_idx_buf = (int*)(src_val_buf + n);
        int *src_sample_idx_buf = src_sorted_idx_buf + n;
        const int* src_sorted_idx = 0;
        const float* src_val = 0;
        data->get_ord_var_data(node, vi, src_val_buf, src_sorted_idx_buf, &src_val, &src_sorted_idx, src_sample_idx_buf);

        for(int i = 0; i < n; i++)
            tempBuf[i] = src_sorted_idx[i];

        if (data->is_buf_16u)
        {
            unsigned short *ldst, *rdst, *ldst0, *rdst0;
            ldst0 = ldst = (unsigned short*)(buf->data.s + left->buf_idx*buf->cols + 
                vi*scount + left->offset);
            rdst0 = rdst = (unsigned short*)(ldst + nl);

            // split sorted
            for( int i = 0; i < n1; i++ )
            {
                int idx = tempBuf[i];
                int d = dir[idx];
                idx = newIdx[idx];
                if (d)
                {
                    *rdst = (unsigned short)idx;
                    rdst++;
                }
                else
                {
                    *ldst = (unsigned short)idx;
                    ldst++;
                }
            }
            assert( n1 == n);

            left->set_num_valid(vi, (int)(ldst - ldst0));
            right->set_num_valid(vi, (int)(rdst - rdst0));
        }   
        else
        {
            int *ldst0, *ldst, *rdst0, *rdst;
            ldst0 = ldst = buf->data.i + left->buf_idx*buf->cols + 
                vi*scount + left->offset;
            rdst0 = rdst = buf->data.i + right->buf_idx*buf->cols + 
                vi*scount + right->offset;

            // split sorted
            for( int i = 0; i < n1; i++ )
            {
                int idx = tempBuf[i];
                int d = dir[idx];
                idx = newIdx[idx];
                if (d)
                {
                    *rdst = idx;
                    rdst++;
                }
                else
                {
                    *ldst = idx;
                    ldst++;
                }
            }

            left->set_num_valid(vi, (int)(ldst - ldst0));
            right->set_num_valid(vi, (int)(rdst - rdst0));
            CV_Assert( n1 == n);
        }  
    }

    // split cv_labels using newIdx relocation table
    int *src_lbls_buf = tempBuf + n;
    const int* src_lbls = data->get_cv_labels(node, src_lbls_buf);

    for(int i = 0; i < n; i++)
        tempBuf[i] = src_lbls[i];

    if (data->is_buf_16u)
    {
        unsigned short *ldst = (unsigned short *)(buf->data.s + left->buf_idx*buf->cols + 
            (workVarCount-1)*scount + left->offset);
        unsigned short *rdst = (unsigned short *)(buf->data.s + right->buf_idx*buf->cols + 
            (workVarCount-1)*scount + right->offset);            
        
        for( int i = 0; i < n; i++ )
        {
            int idx = tempBuf[i];
            if (dir[i])
            {
                *rdst = (unsigned short)idx;
                rdst++;
            }
            else
            {
                *ldst = (unsigned short)idx;
                ldst++;
            }
        }

    }
    else
    {
        int *ldst = buf->data.i + left->buf_idx*buf->cols + 
            (workVarCount-1)*scount + left->offset;
        int *rdst = buf->data.i + right->buf_idx*buf->cols + 
            (workVarCount-1)*scount + right->offset;
        
        for( int i = 0; i < n; i++ )
        {
            int idx = tempBuf[i];
            if (dir[i])
            {
                *rdst = idx;
                rdst++;
            }
            else
            {
                *ldst = idx;
                ldst++;
            }
        }
    }        
    for( int vi = 0; vi < data->var_count; vi++ )
    {
        left->set_num_valid(vi, (int)(nl));
        right->set_num_valid(vi, (int)(nr));
    }

    // split sample indices
    int *sampleIdx_src_buf = tempBuf + n;
    const int* sampleIdx_src = data->get_sample_indices(node, sampleIdx_src_buf);

    for(int i = 0; i < n; i++)
        tempBuf[i] = sampleIdx_src[i];

    if (data->is_buf_16u)
    {
        unsigned short* ldst = (unsigned short*)(buf->data.s + left->buf_idx*buf->cols + 
            workVarCount*scount + left->offset);
        unsigned short* rdst = (unsigned short*)(buf->data.s + right->buf_idx*buf->cols + 
            workVarCount*scount + right->offset);
        for (int i = 0; i < n; i++)
        {
            unsigned short idx = (unsigned short)tempBuf[i];
            if (dir[i])
            {
                *rdst = idx;
                rdst++;
            }
            else
            {
                *ldst = idx;
                ldst++;
            }
        }
    }
    else
    {
        int* ldst = buf->data.i + left->buf_idx*buf->cols + 
            workVarCount*scount + left->offset;
        int* rdst = buf->data.i + right->buf_idx*buf->cols + 
            workVarCount*scount + right->offset;
        for (int i = 0; i < n; i++)
        {
            int idx = tempBuf[i];
            if (dir[i])
            {
                *rdst = idx;
                rdst++;
            }
            else
            {
                *ldst = idx;
                ldst++;
            }
        }
    }

    // deallocate the parent node data that is not needed anymore
    data->free_node_data(node); 
}

void auxMarkFeaturesInMap( const CvDTreeNode* node, Mat& featureMap)
{
    if ( node && node->split )
    {
        featureMap.ptr<int>(0)[node->split->var_idx] = 1;
        auxMarkFeaturesInMap( node->left, featureMap );
        auxMarkFeaturesInMap( node->right, featureMap );
    }
}

void CvCascadeBoostTree::markFeaturesInMap( Mat& featureMap )
{
    auxMarkFeaturesInMap( root, featureMap );
}

//----------------------------------- CascadeBoost --------------------------------------

bool CvCascadeBoost::train( const CvFeatureEvaluator* _featureEvaluator,
                           int _numSamples,
                           int _precalcValBufSize, int _precalcIdxBufSize,
                           const CvCascadeBoostParams& _params )
{
    CV_Assert( !data );
    clear();
    data = new CvCascadeBoostTrainData( _featureEvaluator, _numSamples,
                                        _precalcValBufSize, _precalcIdxBufSize, _params );
    CvMemStorage *storage = cvCreateMemStorage();
    weak = cvCreateSeq( 0, sizeof(CvSeq), sizeof(CvBoostTree*), storage );
    storage = 0;

    set_params( _params );
    if ( (_params.boost_type == LOGIT) || (_params.boost_type == GENTLE) )
        data->do_responses_copy();
    
    update_weights( 0 );

    cout << "+----+---------+---------+" << endl;
    cout << "|  N |    HR   |    FA   |" << endl;
    cout << "+----+---------+---------+" << endl;

    do
    {
        CvCascadeBoostTree* tree = new CvCascadeBoostTree;
        if( !tree->train( data, subsample_mask, this ) )
        {
            // TODO: may be should finish the loop (!!!)
            assert(0);
            delete tree;
            continue;
        }
        cvSeqPush( weak, &tree );
        update_weights( tree );
        trim_weights();
    }
    while( !isErrDesired() && (weak->total < params.weak_count) );

    data->is_classifier = true;
    data->free_train_data();
    return true;
}

float CvCascadeBoost::predict( int sampleIdx, bool returnSum ) const
{
    CV_Assert( weak );
    double sum = 0;
    CvSeqReader reader;
    cvStartReadSeq( weak, &reader );
    cvSetSeqReaderPos( &reader, 0 );
    for( int i = 0; i < weak->total; i++ )
    {
        CvBoostTree* wtree;
        CV_READ_SEQ_ELEM( wtree, reader );
        sum += ((CvCascadeBoostTree*)wtree)->predict(sampleIdx)->value;
    }
    if( !returnSum )
        sum = sum < threshold - CV_THRESHOLD_EPS ? 0.0 : 1.0;
    return (float)sum;
}

bool CvCascadeBoost::set_params( const CvBoostParams& _params )
{
    minHitRate = ((CvCascadeBoostParams&)_params).minHitRate;
    maxFalseAlarm = ((CvCascadeBoostParams&)_params).maxFalseAlarm;
    return ( ( minHitRate > 0 ) && ( minHitRate < 1) &&
        ( maxFalseAlarm > 0 ) && ( maxFalseAlarm < 1) && 
        CvBoost::set_params( _params ));
}

void CvCascadeBoost::update_weights( CvBoostTree* tree )
{
    int n = data->sample_count;
    double sumW = 0.;
    int step = 0;
    float* fdata = 0;
    int *sampleIdxBuf;
    const int* sampleIdx = 0;
    int inn_buf_size = ((params.boost_type == LOGIT) || (params.boost_type == GENTLE) ? n*sizeof(int) : 0) +
                       ( !tree ? n*sizeof(int) : 0 );
    cv::AutoBuffer<uchar> inn_buf(inn_buf_size);
    uchar* cur_inn_buf_pos = (uchar*)inn_buf;
    if ( (params.boost_type == LOGIT) || (params.boost_type == GENTLE) )
    {
        step = CV_IS_MAT_CONT(data->responses_copy->type) ?
            1 : data->responses_copy->step / CV_ELEM_SIZE(data->responses_copy->type);
        fdata = data->responses_copy->data.fl;
        sampleIdxBuf = (int*)cur_inn_buf_pos; cur_inn_buf_pos = (uchar*)(sampleIdxBuf + n);
        sampleIdx = data->get_sample_indices( data->data_root, sampleIdxBuf );
    }
    CvMat* buf = data->buf;
    if( !tree ) // before training the first tree, initialize weights and other parameters
    {
        int* classLabelsBuf = (int*)cur_inn_buf_pos; cur_inn_buf_pos = (uchar*)(classLabelsBuf + n);
        const int* classLabels = data->get_class_labels(data->data_root, classLabelsBuf);
        // in case of logitboost and gentle adaboost each weak tree is a regression tree,
        // so we need to convert class labels to floating-point values
        double w0 = 1./n;
        double p[2] = { 1, 1 };

        cvReleaseMat( &orig_response );
        cvReleaseMat( &sum_response );
        cvReleaseMat( &weak_eval );
        cvReleaseMat( &subsample_mask );
        cvReleaseMat( &weights );

        orig_response = cvCreateMat( 1, n, CV_32S );
        weak_eval = cvCreateMat( 1, n, CV_64F );
        subsample_mask = cvCreateMat( 1, n, CV_8U );
        weights = cvCreateMat( 1, n, CV_64F );
        subtree_weights = cvCreateMat( 1, n + 2, CV_64F );

        if (data->is_buf_16u)
        {
            unsigned short* labels = (unsigned short*)(buf->data.s + data->data_root->buf_idx*buf->cols + 
                data->data_root->offset + (data->work_var_count-1)*data->sample_count);
            for( int i = 0; i < n; i++ )
            {
                // save original categorical responses {0,1}, convert them to {-1,1}
                orig_response->data.i[i] = classLabels[i]*2 - 1;
                // make all the samples active at start.
                // later, in trim_weights() deactivate/reactive again some, if need
                subsample_mask->data.ptr[i] = (uchar)1;
                // make all the initial weights the same.
                weights->data.db[i] = w0*p[classLabels[i]];
                // set the labels to find (from within weak tree learning proc)
                // the particular sample weight, and where to store the response.
                labels[i] = (unsigned short)i;
            }
        }
        else
        {
            int* labels = buf->data.i + data->data_root->buf_idx*buf->cols + 
                data->data_root->offset + (data->work_var_count-1)*data->sample_count;

            for( int i = 0; i < n; i++ )
            {
                // save original categorical responses {0,1}, convert them to {-1,1}
                orig_response->data.i[i] = classLabels[i]*2 - 1;
                subsample_mask->data.ptr[i] = (uchar)1;
                weights->data.db[i] = w0*p[classLabels[i]];
                labels[i] = i;
            }
        }

        if( params.boost_type == LOGIT )
        {
            sum_response = cvCreateMat( 1, n, CV_64F );

            for( int i = 0; i < n; i++ )
            {
                sum_response->data.db[i] = 0;
                fdata[sampleIdx[i]*step] = orig_response->data.i[i] > 0 ? 2.f : -2.f;
            }

            // in case of logitboost each weak tree is a regression tree.
            // the target function values are recalculated for each of the trees
            data->is_classifier = false;
        }
        else if( params.boost_type == GENTLE )
        {
            for( int i = 0; i < n; i++ )
                fdata[sampleIdx[i]*step] = (float)orig_response->data.i[i];

            data->is_classifier = false;
        }
    }
    else
    {
        // at this moment, for all the samples that participated in the training of the most
        // recent weak classifier we know the responses. For other samples we need to compute them
        if( have_subsample )
        {
            // invert the subsample mask
            cvXorS( subsample_mask, cvScalar(1.), subsample_mask );
            
            // run tree through all the non-processed samples
            for( int i = 0; i < n; i++ )
                if( subsample_mask->data.ptr[i] )
                {
                    weak_eval->data.db[i] = ((CvCascadeBoostTree*)tree)->predict( i )->value;
                }
        }

        // now update weights and other parameters for each type of boosting
        if( params.boost_type == DISCRETE )
        {
            // Discrete AdaBoost:
            //   weak_eval[i] (=f(x_i)) is in {-1,1}
            //   err = sum(w_i*(f(x_i) != y_i))/sum(w_i)
            //   C = log((1-err)/err)
            //   w_i *= exp(C*(f(x_i) != y_i))

            double C, err = 0.;
            double scale[] = { 1., 0. };

            for( int i = 0; i < n; i++ )
            {
                double w = weights->data.db[i];
                sumW += w;
                err += w*(weak_eval->data.db[i] != orig_response->data.i[i]);
            }

            if( sumW != 0 )
                err /= sumW;
            C = err = -logRatio( err );
            scale[1] = exp(err);

            sumW = 0;
            for( int i = 0; i < n; i++ )
            {
                double w = weights->data.db[i]*
                    scale[weak_eval->data.db[i] != orig_response->data.i[i]];
                sumW += w;
                weights->data.db[i] = w;
            }

            tree->scale( C );
        }
        else if( params.boost_type == REAL )
        {
            // Real AdaBoost:
            //   weak_eval[i] = f(x_i) = 0.5*log(p(x_i)/(1-p(x_i))), p(x_i)=P(y=1|x_i)
            //   w_i *= exp(-y_i*f(x_i))

            for( int i = 0; i < n; i++ )
                weak_eval->data.db[i] *= -orig_response->data.i[i];

            cvExp( weak_eval, weak_eval );

            for( int i = 0; i < n; i++ )
            {
                double w = weights->data.db[i]*weak_eval->data.db[i];
                sumW += w;
                weights->data.db[i] = w;
            }
        }
        else if( params.boost_type == LOGIT )
        {
            // LogitBoost:
            //   weak_eval[i] = f(x_i) in [-z_max,z_max]
            //   sum_response = F(x_i).
            //   F(x_i) += 0.5*f(x_i)
            //   p(x_i) = exp(F(x_i))/(exp(F(x_i)) + exp(-F(x_i))=1/(1+exp(-2*F(x_i)))
            //   reuse weak_eval: weak_eval[i] <- p(x_i)
            //   w_i = p(x_i)*1(1 - p(x_i))
            //   z_i = ((y_i+1)/2 - p(x_i))/(p(x_i)*(1 - p(x_i)))
            //   store z_i to the data->data_root as the new target responses

            const double lbWeightThresh = FLT_EPSILON;
            const double lbZMax = 10.;

            for( int i = 0; i < n; i++ )
            {
                double s = sum_response->data.db[i] + 0.5*weak_eval->data.db[i];
                sum_response->data.db[i] = s;
                weak_eval->data.db[i] = -2*s;
            }

            cvExp( weak_eval, weak_eval );

            for( int i = 0; i < n; i++ )
            {
                double p = 1./(1. + weak_eval->data.db[i]);
                double w = p*(1 - p), z;
                w = MAX( w, lbWeightThresh );
                weights->data.db[i] = w;
                sumW += w;
                if( orig_response->data.i[i] > 0 )
                {
                    z = 1./p;
                    fdata[sampleIdx[i]*step] = (float)min(z, lbZMax);
                }
                else
                {
                    z = 1./(1-p);
                    fdata[sampleIdx[i]*step] = (float)-min(z, lbZMax);
                }
            }
        }
        else
        {
            // Gentle AdaBoost:
            //   weak_eval[i] = f(x_i) in [-1,1]
            //   w_i *= exp(-y_i*f(x_i))
            assert( params.boost_type == GENTLE );

            for( int i = 0; i < n; i++ )
                weak_eval->data.db[i] *= -orig_response->data.i[i];

            cvExp( weak_eval, weak_eval );

            for( int i = 0; i < n; i++ )
            {
                double w = weights->data.db[i] * weak_eval->data.db[i];
                weights->data.db[i] = w;
                sumW += w;
            }
        }
    }

    // renormalize weights
    if( sumW > FLT_EPSILON )
    {
        sumW = 1./sumW;
        for( int i = 0; i < n; ++i )
            weights->data.db[i] *= sumW;
    }
}

bool CvCascadeBoost::isErrDesired()
{
    int sCount = data->sample_count,
        numPos = 0, numNeg = 0, numFalse = 0, numPosTrue = 0;
    vector<float> eval(sCount);
    
    for( int i = 0; i < sCount; i++ )
        if( ((CvCascadeBoostTrainData*)data)->featureEvaluator->getCls( i ) == 1.0F )
            eval[numPos++] = predict( i, true );
    icvSortFlt( &eval[0], numPos, 0 );
    int thresholdIdx = (int)((1.0F - minHitRate) * numPos);
    threshold = eval[ thresholdIdx ];
    numPosTrue = numPos - thresholdIdx;
    for( int i = thresholdIdx - 1; i >= 0; i--)
        if ( abs( eval[i] - threshold) < FLT_EPSILON )
            numPosTrue++;
    float hitRate = ((float) numPosTrue) / ((float) numPos);

    for( int i = 0; i < sCount; i++ )
    {
        if( ((CvCascadeBoostTrainData*)data)->featureEvaluator->getCls( i ) == 0.0F )
        {
            numNeg++;
            if( predict( i ) )
                numFalse++;
        }
    }
    float falseAlarm = ((float) numFalse) / ((float) numNeg);

    cout << "|"; cout.width(4); cout << right << weak->total;
    cout << "|"; cout.width(9); cout << right << hitRate;
    cout << "|"; cout.width(9); cout << right << falseAlarm;
    cout << "|" << endl;
    cout << "+----+---------+---------+" << endl;

    return falseAlarm <= maxFalseAlarm;
}

void CvCascadeBoost::write( FileStorage &fs, const Mat& featureMap ) const
{
//    char cmnt[30];
    CvCascadeBoostTree* weakTree;
    fs << CC_WEAK_COUNT << weak->total;
    fs << CC_STAGE_THRESHOLD << threshold;
    fs << CC_WEAK_CLASSIFIERS << "[";
    for( int wi = 0; wi < weak->total; wi++)
    {
        /*sprintf( cmnt, "tree %i", wi );
        cvWriteComment( fs, cmnt, 0 );*/
        weakTree = *((CvCascadeBoostTree**) cvGetSeqElem( weak, wi ));
        weakTree->write( fs, featureMap );
    }
    fs << "]";
}

bool CvCascadeBoost::read( const FileNode &node,
                           const CvFeatureEvaluator* _featureEvaluator,
                           const CvCascadeBoostParams& _params )
{
    CvMemStorage* storage;
    clear();
    data = new CvCascadeBoostTrainData( _featureEvaluator, _params );
    set_params( _params );

    node[CC_STAGE_THRESHOLD] >> threshold;
    FileNode rnode = node[CC_WEAK_CLASSIFIERS]; 

    storage = cvCreateMemStorage();
    weak = cvCreateSeq( 0, sizeof(CvSeq), sizeof(CvBoostTree*), storage );
    for( FileNodeIterator it = rnode.begin(); it != rnode.end(); it++ )
    {
        CvCascadeBoostTree* tree = new CvCascadeBoostTree();
        tree->read( *it, this, data );
        cvSeqPush( weak, &tree );
    }
    return true;
}

void CvCascadeBoost::markUsedFeaturesInMap( Mat& featureMap )
{
    for( int wi = 0; wi < weak->total; wi++ )
    {
        CvCascadeBoostTree* weakTree = *((CvCascadeBoostTree**) cvGetSeqElem( weak, wi ));
        weakTree->markFeaturesInMap( featureMap );
    }
}
