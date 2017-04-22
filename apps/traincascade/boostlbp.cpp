#include "boostlbp.h"


#define MY_CMP_DBL(i,j) (i < j)
static MY_IMPLEMENT_QSORT_EX( mySortDbl, double, MY_CMP_DBL, const double* )


// only for calAUC function
class Comp{
	const double *dec_val;
public:
	Comp(const double *ptr): dec_val(ptr){}
	bool operator()(int i, int j) const{
		return dec_val[i] > dec_val[j];
	}
};

double calAUC(const std::vector<double> & dec_values, const Mat& labels){
	double roc = 0;
	size_t size = dec_values.size();
	size_t i;
    vector<size_t> indices(size);
    const unsigned char * pLabels = labels.ptr(0);
    
	for(i = 0; i < size; ++i) indices[i] = i;
    
	std::sort(indices.begin(), indices.end(), Comp( &dec_values[0] ));
    
	int tp = 0,fp = 0;
	for(i = 0; i < size; i++) {
		if(pLabels[indices[i]] == 1) tp++;
		else if(pLabels[indices[i]] == 0) {
			roc += tp;
			fp++;
		}
	}
    
	if(tp == 0 || fp == 0)
	{
		cerr << "warning: Too few postive true labels or negative true labels" << endl;
		roc = 0;
	}
	else
		roc = roc / tp / fp;
    
	return roc;
}


//cachedValues will be updated in the function
int select_the_best_weakclassifier_auc(const Mat & samplesLBP, 
                                       const Mat & labels,
                                       MBLBPWeakf * features, 
                                       bool * featuresMask, 
                                       int numFeatures_,
                                       Mat & cachedValues,
                                       double & current_auc_value)
{
    int numSamples = samplesLBP.cols;
    int numFeatures = samplesLBP.rows;

    CV_Assert(!samplesLBP.empty());
    CV_Assert(features);
    CV_Assert(featuresMask);
    CV_Assert(numFeatures == numFeatures_);
    CV_Assert(numSamples == labels.cols);
    CV_Assert(numSamples == cachedValues.cols);

    double t = (double)cvGetTickCount();
    //calculate auc for each feature
    vector<double> auc_values(numFeatures);

#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for(int weak_idx = 0; weak_idx < numFeatures; weak_idx++)
    {
        if(featuresMask[weak_idx] == true)
        {
            auc_values[weak_idx] = 0.0;
            continue;
        }

        const unsigned char * pLBP = samplesLBP.ptr(weak_idx);
        vector<double> pred_values(numSamples);
        Mat tmpCachedValues = Mat::zeros(1, numSamples, CV_64FC1);
        CV_Assert(!tmpCachedValues.empty());
        memset(tmpCachedValues.ptr(0), 0, tmpCachedValues.step * tmpCachedValues.rows);

        {
            double * cached_values = cachedValues.ptr<double>(0);
            double * tmp_cached_values = tmpCachedValues.ptr<double>(0);
            for(int i = 0; i < numSamples; i++)
            {
                unsigned char code = pLBP[i];
                tmp_cached_values[i] = cached_values[i] + features[weak_idx].look_up_table[code];
            }
        }

        for(int i = 0; i < numSamples; i++)
        {
            double tmp;
            pred_values[i] = tmpCachedValues.at<double>(0,i);
            tmp = tmpCachedValues.at<double>(0,i);
            if(pred_values[i] < tmp)
                pred_values[i] = tmp;
        }

        auc_values[weak_idx] = calAUC(pred_values, labels);
    }
   
    t = (double)cvGetTickCount() - t;
    // cout << "AUCInvoker time: " << (t/((double)cvGetTickFrequency()*1000.)) / 1000  << "s" << endl;

    //find the best one
    int best_weak_index = 0;
	double best_auc_value = auc_values[0];
    for(int weak_index = 1; weak_index < numFeatures; weak_index++) {
        if(auc_values[weak_index] > best_auc_value)
        {
            best_weak_index = weak_index;
            best_auc_value = auc_values[weak_index];
        }
    }

    //update cached values
    current_auc_value = best_auc_value;
    MBLBPWeakf * bestFeature = features + best_weak_index;
    const unsigned char * pLBP = samplesLBP.ptr(best_weak_index);
    double * cached_values = cachedValues.ptr<double>(0);
    for(int i = 0; i < numSamples; i++)
    {
        int code = pLBP[i];
        cached_values[i] += bestFeature->look_up_table[code];
    }

    return best_weak_index;
}

bool isErrDesiredAUC(const Mat & cachedValues,
                     const Mat & labels,
                     double min_hit_rate,
                     double current_auc_value,
                     double prev_auc_value,
                     double & stage_threshold,
                     double & soft_threshold,
                     double & false_alarm)
{
	int sample_count = cachedValues.cols;
    int num_pos = 0;
    int num_neg = 0;
    int num_false = 0;
    int num_pos_true = 0;

	vector<double> eval(sample_count);

    //const double * cached_values = cachedValues.ptr<double>(0);
    const unsigned char * pLabels = labels.ptr(0);
    vector<double> max_cached_values(sample_count);

	for( int i = 0; i < sample_count; i++ )
    {
        max_cached_values[i] = cachedValues.at<double>(0, i);
        double tmp = cachedValues.at<double>(0,i);
        if( max_cached_values[i] < tmp)
            max_cached_values[i] = tmp;

		if( pLabels[i] == 1 )
		{
			eval[num_pos] = max_cached_values[i];
            num_pos++;
		}
    }
                
	mySortDbl( &eval[0], num_pos, 0 );
    
	int thresholdIdx = (int)((1.0 - min_hit_rate) * num_pos);

    stage_threshold = eval[ thresholdIdx ];
    soft_threshold = eval[0];

    // weak threshold
	num_pos_true = num_pos - thresholdIdx;
	for( int i = thresholdIdx - 1; i >= 0; i--)
        if( fabs(eval[i] - stage_threshold) < DBL_EPSILON)
			num_pos_true++;
	double hit_rate = ((double) num_pos_true) / ((double) num_pos);
	for( int i = 0; i < sample_count; i++ )
	{
        if(pLabels[i] == 0)
		{
			num_neg++;

            if( max_cached_values[i] > stage_threshold)
                num_false++;
		}
	}
    
	false_alarm = ((double) num_false) / ((double) num_neg);
    
	cout << "|"; cout.width(9); cout << right << hit_rate;
	cout << "|"; cout.width(9); cout << right << false_alarm;
	cout << "|"; cout.width(9); cout << right << current_auc_value;
	cout << "|"; cout.width(9); cout << right << current_auc_value - prev_auc_value;
    cout << "|"; cout.width(9); cout << right << soft_threshold;
	cout << "|" << endl;
	cout << "+----+---------+---------+---------+---------+---------+" << endl;
    
	//if( current_auc_value - prev_auc_value < 0.0001 && false_alarm < 0.5f || false_alarm < 0.1f )
	//	return true;
    //else
		return false;
    
}

void update_weights(const Mat & samplesLBP, 
                    const Mat & labels,
                    int numPos,
                    int numNeg,
                    const MBLBPWeakf * features,
                    MBLBPStagef * pStrong,  
                    Mat & weights)
{
    int numSamples = samplesLBP.cols;

    CV_Assert(numSamples == labels.cols);
    CV_Assert(numSamples == weights.cols);
    CV_Assert(numSamples == numPos + numNeg);

    int feature_index = pStrong->weak_classifiers_idx[pStrong->count-1];
    const unsigned char * pLBP = samplesLBP.ptr(feature_index);

    double weight_sum = 0.0;

    //reweighting samples
    //for each sample
    for(int i = 0; i < numSamples; i++)
    {
        double s = (i<numPos)*(-2) + 1; //s=-1 for positives; s=+1 for negatives
        int code = pLBP[i];
        weights.at<double>(0,i) *= exp( features[feature_index].look_up_table[code] * s );
        weight_sum += weights.at<double>(0,i);
    }

    //normalization weights
    if( weight_sum > DBL_EPSILON )
    {
        weight_sum = 1.0/weight_sum;
        for(int i = 0; i < numSamples; i++)
            weights.at<double>(0,i) *= weight_sum;
    }
}


bool updateFeatureLUT(const Mat & samplesLBP, 
                      const Mat & weights,
                      int numPos,
                      int numNeg,
                      MBLBPWeakf * features, 
                      bool * featuresMask, 
                      int numFeatures_)
{
    int numSamples = samplesLBP.cols;
    int numFeatures = samplesLBP.rows;
    
    CV_Assert(!samplesLBP.empty());
    CV_Assert(features);
    CV_Assert(featuresMask);
    CV_Assert(numFeatures == numFeatures_);
    CV_Assert(numSamples == weights.cols);

#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for(int fidx = 0; fidx < numFeatures; fidx++)
    {
        if(featuresMask[fidx])
            continue;

        MBLBPWeakf * feature = features + fidx;
        memset(feature->look_up_table, 0, sizeof(feature->look_up_table));

        const unsigned char * pLBP = samplesLBP.ptr(fidx);
        const double * pWeights = weights.ptr<double>(0);

        double w_plus[MBLBP_LUTLENGTH];
        double w_minus[MBLBP_LUTLENGTH];
        memset(w_plus, 0, sizeof(w_plus));
        memset(w_minus, 0, sizeof(w_minus));

        //for each sample
        for(int i = 0; i < numSamples; i++)
        {
            int code = pLBP[i];
            double w = pWeights[i];
                    
            if(i < numPos) //positive samples
                w_plus[code] += w;
            else //negative samples
                w_minus[code] += w;
        }

        for(int i = 0; i < MBLBP_LUTLENGTH; i++)
        {
            double s = w_plus[i] + w_minus[i];
            double d = w_plus[i] - w_minus[i];

            if( fabs(s) < DBL_EPSILON )
                feature->look_up_table[i] = 0;
            else
                feature->look_up_table[i] = d / s;

        }//end for i

   }//end for fidx
    
    return true;
}


bool boostTrain(MBLBPStagef * pStrong,
                const Mat & samplesLBP,
                const Mat & labels,
                MBLBPWeakf * features,
                bool * featuresMask,
                int numFeatures,
                int numPos,
                int numNeg,
                int maxWeakCount,
                double min_hit_rate)
{
    int numSamples = 0;
    numSamples += numPos;
    numSamples += numNeg;

    CV_Assert(pStrong);
    CV_Assert(numSamples > 0);
    CV_Assert(numSamples == samplesLBP.cols);
    CV_Assert(numSamples == labels.cols);
    CV_Assert(numFeatures == samplesLBP.rows);

	// alloc mem for weights
	Mat weights = Mat::zeros( 1, numSamples, CV_64FC1);
    CV_Assert(!weights.empty());

    //assign initial value
    double * p = weights.ptr<double>(0);
    for(int i = 0; i < numPos; i++)
    {
        p[i] = 0.5/numPos;
    }
    for(int i = numPos; i < numSamples; i++)
    {
        p[i] = 0.5/numNeg;
    }


	// alloc and zero cachepred_
	Mat cachedValues = Mat::zeros(1, numSamples, CV_64FC1);
    CV_Assert(!cachedValues.empty());
    
	double prev_auc_value = 0.;
    double current_auc_value = 0.;
    

    //init LUT
    if (! updateFeatureLUT(samplesLBP, weights, numPos, numNeg, features, featuresMask, numFeatures)){
        cout<<"update feature LUT Failed"<<endl;    
        return false;
    }

	cout << "+----+---------+---------+---------+---------+---------+" << endl;
	cout << "|  N |    HR   |    FA   |    AUC  | AUC inc | thesh   |" << endl;
	cout << "+----+---------+---------+---------+---------+---------+" << endl;

    
	do
	{
        double t = (double)cvGetTickCount();
        prev_auc_value = current_auc_value;
		int weak_index = select_the_best_weakclassifier_auc(samplesLBP, 
                                                            labels,
                                                            features, 
                                                            featuresMask, 
                                                            numFeatures,
                                                            cachedValues,
                                                            current_auc_value);
        t = (double)cvGetTickCount() - t;
        //cout << "select_the_best_weakclassifier_auc time: " << (t/((double)cvGetTickFrequency()*1000.)) / 1000  << "s" << endl;

        if(weak_index < 0)
            return false;

        pStrong->weak_classifiers_idx[pStrong->count] = weak_index;
        pStrong->weak_classifiers[pStrong->count] = features[weak_index];
        pStrong->count++;
        featuresMask[weak_index] = true;
        

        cout << "|"; cout.width(4); cout << right << pStrong->count;

        //threshold will be set
        double false_alarm = 1.0;
		isErrDesiredAUC( cachedValues,
                            labels,
                            min_hit_rate,                            
                            current_auc_value,
                            prev_auc_value,
                            pStrong->threshold,
                            features[weak_index].soft_threshold,
                            false_alarm);

        if( (false_alarm < 0.4) && ( pStrong->count%60 == 0))
        {
            if(pStrong->count > 1)
			    break;
        }
        else
            pStrong->weak_classifiers[pStrong->count-1].soft_threshold = features[weak_index].soft_threshold;


        if(pStrong->count >= maxWeakCount)  //too many weak classifiers
        {
            break;
        }

        update_weights(samplesLBP, labels, numPos, numNeg, features, pStrong, weights);

        t = (double)cvGetTickCount();
        if (! updateFeatureLUT(samplesLBP, weights, numPos, numNeg, features, featuresMask, numFeatures))
            return false;
        t = (double)cvGetTickCount() - t;

        //cout << "updateFeaturePWFunctions time: " << (t/((double)cvGetTickFrequency()*1000.)) / 1000  << "s" << endl;

        
	}
	while( pStrong->count < maxWeakCount );
    
    
	return true;
}

