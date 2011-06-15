
#include "precomp.hpp"
#include <string>
#include <time.h>

using namespace std;

#define pCvSeq CvSeq*
#define pCvDTreeNode CvDTreeNode*

#define CV_CMP_FLOAT(a,b) ((a) < (b))
static CV_IMPLEMENT_QSORT_EX( icvSortFloat, float, CV_CMP_FLOAT, float)

//===========================================================================
string ToString(int i)
{
    stringstream tmp;
    tmp << i;

    return tmp.str();
}


//===========================================================================
//----------------------------- CvGBTreesParams -----------------------------
//===========================================================================

CvGBTreesParams::CvGBTreesParams() 
            : CvDTreeParams( 3, 10, 0, false, 10, 0, false, false, 0 )
{
    weak_count = 200;
    loss_function_type = CvGBTrees::SQUARED_LOSS;
    subsample_portion = 0.8f;
    shrinkage = 0.01f;
}

//===========================================================================

CvGBTreesParams::CvGBTreesParams( int _loss_function_type, int _weak_count, 
                         float _shrinkage, float _subsample_portion, 
                         int _max_depth, bool _use_surrogates )
            : CvDTreeParams( 3, 10, 0, false, 10, 0, false, false, 0 )
{
    loss_function_type = _loss_function_type;
    weak_count = _weak_count;
    shrinkage = _shrinkage;
    subsample_portion = _subsample_portion;
    max_depth = _max_depth;
    use_surrogates = _use_surrogates;
}

//===========================================================================
//------------------------------- CvGBTrees ---------------------------------
//===========================================================================

CvGBTrees::CvGBTrees()
{
    data = 0;
    weak = 0;
    default_model_name = "my_boost_tree";
    orig_response = sum_response = sum_response_tmp = 0;
    subsample_train = subsample_test = 0;
    missing = sample_idx = 0;
    class_labels = 0;
    class_count = 1;
    delta = 0.0f;
    
    clear();
}

//===========================================================================

int CvGBTrees::get_len(const CvMat* mat) const
{
    return (mat->cols > mat->rows) ? mat->cols : mat->rows;
}

//===========================================================================

void CvGBTrees::clear()
{
    if( weak )
    {
        CvSeqReader reader;
        CvSlice slice = CV_WHOLE_SEQ;
        CvDTree* tree;

        //data->shared = false;
        for (int i=0; i<class_count; ++i)
        {
			int weak_count = cvSliceLength( slice, weak[i] );
            if ((weak[i]) && (weak_count))
            {
                cvStartReadSeq( weak[i], &reader ); 
                cvSetSeqReaderPos( &reader, slice.start_index );
                for (int j=0; j<weak_count; ++j)
                {
                    CV_READ_SEQ_ELEM( tree, reader );
                    //tree->clear();
                    delete tree;
                    tree = 0;
                }
            }
        }
        for (int i=0; i<class_count; ++i)
            if (weak[i]) cvReleaseMemStorage( &(weak[i]->storage) );
        delete[] weak;
    }
    if (data) 
    {
        data->shared = false;
        delete data;
    }
    weak = 0;
    data = 0;
    delta = 0.0f;
    cvReleaseMat( &orig_response );
    cvReleaseMat( &sum_response );
    cvReleaseMat( &sum_response_tmp );
    cvReleaseMat( &subsample_train );
    cvReleaseMat( &subsample_test );
    cvReleaseMat( &sample_idx );
    cvReleaseMat( &missing );
    cvReleaseMat( &class_labels );
}

//===========================================================================

CvGBTrees::~CvGBTrees()
{
    clear();
}

//===========================================================================

CvGBTrees::CvGBTrees( const CvMat* _train_data, int _tflag,
                  const CvMat* _responses, const CvMat* _var_idx,
                  const CvMat* _sample_idx, const CvMat* _var_type,
                  const CvMat* _missing_mask, CvGBTreesParams _params )
{
    weak = 0;
    data = 0;
    default_model_name = "my_boost_tree";
    orig_response = sum_response = sum_response_tmp = 0;
    subsample_train = subsample_test = 0;
    missing = sample_idx = 0;
    class_labels = 0;
    class_count = 1;
    delta = 0.0f;

    train( _train_data, _tflag, _responses, _var_idx, _sample_idx,
           _var_type, _missing_mask, _params );
}

//===========================================================================

bool CvGBTrees::problem_type() const
{
    switch (params.loss_function_type)
    {
    case DEVIANCE_LOSS: return false;
    default: return true;
    }
}

//===========================================================================

bool 
CvGBTrees::train( CvMLData* data, CvGBTreesParams params, bool update )
{
    bool result;
    result = train ( data->get_values(), CV_ROW_SAMPLE,
            data->get_responses(), data->get_var_idx(),
            data->get_train_sample_idx(), data->get_var_types(),
            data->get_missing(), params, update);
                                         //update is not supported
    return result;
}

//===========================================================================


bool
CvGBTrees::train( const CvMat* _train_data, int _tflag,
              const CvMat* _responses, const CvMat* _var_idx,
              const CvMat* _sample_idx, const CvMat* _var_type,
              const CvMat* _missing_mask,
              CvGBTreesParams _params, bool /*_update*/ ) //update is not supported
{
    CvMemStorage* storage = 0;

    params = _params;
    bool is_regression = problem_type();

    clear();
    /*
      n - count of samples
      m - count of variables
    */
    int n = _train_data->rows;
    int m = _train_data->cols;
    if (_tflag != CV_ROW_SAMPLE)
    {
        int tmp;
        CV_SWAP(n,m,tmp);
    }

    CvMat* new_responses = cvCreateMat( n, 1, CV_32F);
    cvZero(new_responses);

    data = new CvDTreeTrainData( _train_data, _tflag, new_responses, _var_idx,
        _sample_idx, _var_type, _missing_mask, _params, true, true );
    if (_missing_mask)
    {
        missing = cvCreateMat(_missing_mask->rows, _missing_mask->cols,
                              _missing_mask->type);
        cvCopy( _missing_mask, missing);
    }

    orig_response = cvCreateMat( 1, n, CV_32F );
	int step = (_responses->cols > _responses->rows) ? 1 : _responses->step / CV_ELEM_SIZE(_responses->type);
    switch (CV_MAT_TYPE(_responses->type))
    {
        case CV_32FC1:
		{
			for (int i=0; i<n; ++i)
                orig_response->data.fl[i] = _responses->data.fl[i*step];
		}; break;
        case CV_32SC1:
        {
            for (int i=0; i<n; ++i)
                orig_response->data.fl[i] = (float) _responses->data.i[i*step];
        }; break;
        default:
            CV_Error(CV_StsUnmatchedFormats, "Response should be a 32fC1 or 32sC1 vector.");
    }

    if (!is_regression)
    {
        class_count = 0;
        unsigned char * mask = new unsigned char[n];
        memset(mask, 0, n);
        // compute the count of different output classes
        for (int i=0; i<n; ++i)
            if (!mask[i])
            {
                class_count++;
                for (int j=i; j<n; ++j)
                    if (int(orig_response->data.fl[j]) == int(orig_response->data.fl[i]))
                        mask[j] = 1;
            }
        delete[] mask;
    
        class_labels = cvCreateMat(1, class_count, CV_32S);
        class_labels->data.i[0] = int(orig_response->data.fl[0]);
        int j = 1;
        for (int i=1; i<n; ++i)
        {
            int k = 0;
            while ((int(orig_response->data.fl[i]) - class_labels->data.i[k]) && (k<j))
                k++;
            if (k == j)
            {
                class_labels->data.i[k] = int(orig_response->data.fl[i]);
                j++;
            }
        }
    }

    // inside gbt learning proccess only regression decision trees are built
    data->is_classifier = false;

    // preproccessing sample indices
    if (_sample_idx)
    {
        int sample_idx_len = get_len(_sample_idx);
        
        switch (CV_MAT_TYPE(_sample_idx->type))
        {
            case CV_32SC1:
            {
                sample_idx = cvCreateMat( 1, sample_idx_len, CV_32S );
                for (int i=0; i<sample_idx_len; ++i)
					sample_idx->data.i[i] = _sample_idx->data.i[i];
            } break;
            case CV_8S:
            case CV_8U:
            {
                int active_samples_count = 0;
                for (int i=0; i<sample_idx_len; ++i)
                    active_samples_count += int( _sample_idx->data.ptr[i] );
                sample_idx = cvCreateMat( 1, active_samples_count, CV_32S );
                active_samples_count = 0;
                for (int i=0; i<sample_idx_len; ++i)
                    if (int( _sample_idx->data.ptr[i] ))
                        sample_idx->data.i[active_samples_count++] = i;
                    
            } break;
            default: CV_Error(CV_StsUnmatchedFormats, "_sample_idx should be a 32sC1, 8sC1 or 8uC1 vector.");
        }
        icvSortFloat(sample_idx->data.fl, sample_idx_len, 0);
    }
    else
    {
        sample_idx = cvCreateMat( 1, n, CV_32S );
        for (int i=0; i<n; ++i)
            sample_idx->data.i[i] = i;
    }

    sum_response = cvCreateMat(class_count, n, CV_32F);
    sum_response_tmp = cvCreateMat(class_count, n, CV_32F);
    cvZero(sum_response);

    delta = 0.0f;
    /*
      in the case of a regression problem the initial guess (the zero term
      in the sum) is set to the mean of all the training responses, that is
      the best constant model
    */
    if (is_regression) base_value = find_optimal_value(sample_idx);
    /*
      in the case of a classification problem the initial guess (the zero term
      in the sum) is set to zero for all the trees sequences
    */
    else base_value = 0.0f;
    /*
      current predicition on all training samples is set to be
      equal to the base_value
    */
    cvSet( sum_response, cvScalar(base_value) );

    weak = new pCvSeq[class_count];
    for (int i=0; i<class_count; ++i)
    {
        storage = cvCreateMemStorage();
        weak[i] = cvCreateSeq( 0, sizeof(CvSeq), sizeof(CvDTree*), storage );
        storage = 0;
    }    

    // subsample params and data
    rng = &cv::theRNG();

	int samples_count = get_len(sample_idx);

    params.subsample_portion = params.subsample_portion <= FLT_EPSILON || 
        1 - params.subsample_portion <= FLT_EPSILON
        ? 1 : params.subsample_portion;
    int train_sample_count = cvFloor(params.subsample_portion * samples_count);
    if (train_sample_count == 0)
        train_sample_count = samples_count;
    int test_sample_count = samples_count - train_sample_count;
    int* idx_data = new int[samples_count];
    subsample_train = cvCreateMatHeader( 1, train_sample_count, CV_32SC1 );
    *subsample_train = cvMat( 1, train_sample_count, CV_32SC1, idx_data );
    if (test_sample_count)
    {
        subsample_test  = cvCreateMatHeader( 1, test_sample_count, CV_32SC1 );
        *subsample_test = cvMat( 1, test_sample_count, CV_32SC1,
                                 idx_data + train_sample_count );
    }
    
    // training procedure

    for ( int i=0; i < params.weak_count; ++i )
    {
		do_subsample();
        for ( int k=0; k < class_count; ++k )
        {
            find_gradient(k);
            CvDTree* tree = new CvDTree;
            tree->train( data, subsample_train );
            change_values(tree, k);

            if (subsample_test)
            {
                CvMat x;
                CvMat x_miss;
                int* sample_data = sample_idx->data.i;
                int* subsample_data = subsample_test->data.i;
                int s_step = (sample_idx->cols > sample_idx->rows) ? 1
                             : sample_idx->step/CV_ELEM_SIZE(sample_idx->type);
                for (int j=0; j<get_len(subsample_test); ++j)
                {
                    int idx = *(sample_data + subsample_data[j]*s_step);
                    float res = 0.0f;
                    if (_tflag == CV_ROW_SAMPLE)
                        cvGetRow( data->train_data, &x, idx);
                    else
                        cvGetCol( data->train_data, &x, idx);
                        
                    if (missing)
                    {
                        if (_tflag == CV_ROW_SAMPLE)
                            cvGetRow( missing, &x_miss, idx);
                        else
                            cvGetCol( missing, &x_miss, idx);
                        
                        res = (float)tree->predict(&x, &x_miss)->value;
                    }
                    else
                    {
                        res = (float)tree->predict(&x)->value;
                    }
                    sum_response_tmp->data.fl[idx + k*n] = 
                                    sum_response->data.fl[idx + k*n] +
                                    params.shrinkage * res;
                }
            }

            cvSeqPush( weak[k], &tree );
            tree = 0;
        } // k=0..class_count
    CvMat* tmp;
    tmp = sum_response_tmp;
    sum_response_tmp = sum_response;
    sum_response = tmp;
    tmp = 0;
    } // i=0..params.weak_count

    delete[] idx_data;
    cvReleaseMat(&new_responses);
    data->free_train_data();

	return true;

} // CvGBTrees::train(...)

//===========================================================================

float Sign(float x)
  {
  if (x<0.0f) return -1.0f;
  else if (x>0.0f) return 1.0f;
  return 0.0f;
  }

//===========================================================================

void CvGBTrees::find_gradient(const int k)
{
    int* sample_data = sample_idx->data.i;
    int* subsample_data = subsample_train->data.i;
    float* grad_data = data->responses->data.fl;
    float* resp_data = orig_response->data.fl;
    float* current_data = sum_response->data.fl;

    switch (params.loss_function_type)
    // loss_function_type in
    // {SQUARED_LOSS, ABSOLUTE_LOSS, HUBER_LOSS, DEVIANCE_LOSS}
    {
        case SQUARED_LOSS:
        {
            for (int i=0; i<get_len(subsample_train); ++i)
            {
                int s_step = (sample_idx->cols > sample_idx->rows) ? 1
                             : sample_idx->step/CV_ELEM_SIZE(sample_idx->type);
                int idx = *(sample_data + subsample_data[i]*s_step);
                grad_data[idx] = resp_data[idx] - current_data[idx];
            }
        }; break;

        case ABSOLUTE_LOSS:
        {
            for (int i=0; i<get_len(subsample_train); ++i)
            {
                int s_step = (sample_idx->cols > sample_idx->rows) ? 1
                             : sample_idx->step/CV_ELEM_SIZE(sample_idx->type);
                int idx = *(sample_data + subsample_data[i]*s_step);
                grad_data[idx] = Sign(resp_data[idx] - current_data[idx]);
            }
        }; break;

        case HUBER_LOSS:
        {
            float alpha = 0.2f;
            int n = get_len(subsample_train);
            int s_step = (sample_idx->cols > sample_idx->rows) ? 1
                         : sample_idx->step/CV_ELEM_SIZE(sample_idx->type);

            float* residuals = new float[n];
            for (int i=0; i<n; ++i)
            {
                int idx = *(sample_data + subsample_data[i]*s_step);
                residuals[i] = fabs(resp_data[idx] - current_data[idx]);
            }
            icvSortFloat(residuals, n, 0.0f);
            
            delta = residuals[int(ceil(n*alpha))];

            for (int i=0; i<n; ++i)
            {
                int idx = *(sample_data + subsample_data[i]*s_step);
                float r = resp_data[idx] - current_data[idx];
                grad_data[idx] = (fabs(r) > delta) ? delta*Sign(r) : r;
            }
            delete[] residuals;

        }; break;

        case DEVIANCE_LOSS:
        {
            for (int i=0; i<get_len(subsample_train); ++i)
            {
                double exp_fk = 0;
                double exp_sfi = 0;
                int s_step = (sample_idx->cols > sample_idx->rows) ? 1
                             : sample_idx->step/CV_ELEM_SIZE(sample_idx->type);
                int idx = *(sample_data + subsample_data[i]*s_step);
            
                for (int j=0; j<class_count; ++j)
                {
                    double res;
                    res = current_data[idx + j*sum_response->cols];
                    res = exp(res);
                    if (j == k) exp_fk = res;
                    exp_sfi += res;
                }
                int orig_label = int(resp_data[idx]);
				/*
                grad_data[idx] = (float)(!(k-class_labels->data.i[orig_label]+1)) -
                                 (float)(exp_fk / exp_sfi);
				*/
				int ensemble_label = 0;
				while (class_labels->data.i[ensemble_label] - orig_label)
					ensemble_label++;				
				
                grad_data[idx] = (float)(!(k-ensemble_label)) -
                                 (float)(exp_fk / exp_sfi);
            }
        }; break;

        default: break;
    }

} // CvGBTrees::find_gradient(...)

//===========================================================================

void CvGBTrees::change_values(CvDTree* tree, const int _k)
{
    CvDTreeNode** predictions = new pCvDTreeNode[get_len(subsample_train)];

    int* sample_data = sample_idx->data.i;
    int* subsample_data = subsample_train->data.i;
    int s_step = (sample_idx->cols > sample_idx->rows) ? 1
                 : sample_idx->step/CV_ELEM_SIZE(sample_idx->type);

    CvMat x;
    CvMat miss_x;

    for (int i=0; i<get_len(subsample_train); ++i)
    {
		int idx = *(sample_data + subsample_data[i]*s_step);
		if (data->tflag == CV_ROW_SAMPLE)
            cvGetRow( data->train_data, &x, idx);
        else
            cvGetCol( data->train_data, &x, idx);
            
        if (missing)
        {
            if (data->tflag == CV_ROW_SAMPLE)
                cvGetRow( missing, &miss_x, idx);
            else
                cvGetCol( missing, &miss_x, idx);
            
            predictions[i] = tree->predict(&x, &miss_x);
        }
        else
            predictions[i] = tree->predict(&x);
    }


    CvDTreeNode** leaves;
    int leaves_count = 0;
    leaves = GetLeaves( tree, leaves_count);

    for (int i=0; i<leaves_count; ++i)
    {
        int samples_in_leaf = 0;
        for (int j=0; j<get_len(subsample_train); ++j)
        {
            if (leaves[i] == predictions[j]) samples_in_leaf++;
        }

        if (!samples_in_leaf) // It should not be done anyways! but...
        {
            leaves[i]->value = 0.0;
            continue; 
        }

        CvMat* leaf_idx = cvCreateMat(1, samples_in_leaf, CV_32S);
        int* leaf_idx_data = leaf_idx->data.i;

        for (int j=0; j<get_len(subsample_train); ++j)
        {
            int idx = *(sample_data + subsample_data[j]*s_step);
            if (leaves[i] == predictions[j])
                *leaf_idx_data++ = idx;
        }

        float value = find_optimal_value(leaf_idx);
        leaves[i]->value = value;

        leaf_idx_data = leaf_idx->data.i;

        int len = sum_response_tmp->cols;
        for (int j=0; j<get_len(leaf_idx); ++j)
        {
            int idx = leaf_idx_data[j];        
            sum_response_tmp->data.fl[idx + _k*len] =
                                    sum_response->data.fl[idx + _k*len] +
                                    params.shrinkage * value;
        }
        leaf_idx_data = 0;     
        cvReleaseMat(&leaf_idx);
    }

    // releasing the memory
    for (int i=0; i<get_len(subsample_train); ++i)
    {
        predictions[i] = 0;
    }
    delete[] predictions;

    for (int i=0; i<leaves_count; ++i)
    {
        leaves[i] = 0;
    }
    delete[] leaves;

}

//===========================================================================
/*
void CvGBTrees::change_values(CvDTree* tree, const int _k)
{
    
    CvDTreeNode** leaves;
    int leaves_count = 0;
	int offset = _k*sum_response_tmp->cols;
	CvMat leaf_idx;
	leaf_idx.rows = 1;
    
    leaves = GetLeaves( tree, leaves_count);

    for (int i=0; i<leaves_count; ++i)
    {
        int n = leaves[i]->sample_count;
        int* leaf_idx_data = new int[n];
        data->get_sample_indices(leaves[i], leaf_idx_data);
        //CvMat* leaf_idx = new CvMat();
        //cvInitMatHeader(leaf_idx, n, 1, CV_32S, leaf_idx_data);
		leaf_idx.cols = n;
		leaf_idx.data.i = leaf_idx_data;

        float value = find_optimal_value(&leaf_idx);
        leaves[i]->value = value;
		float val = params.shrinkage * value;

        
        for (int j=0; j<n; ++j)
        {
            int idx = leaf_idx_data[j] + offset;
            sum_response_tmp->data.fl[idx] = sum_response->data.fl[idx] + val;
        }
        //leaf_idx_data = 0;
        //cvReleaseMat(&leaf_idx);
		leaf_idx.data.i = 0;
		//delete leaf_idx;
		delete[] leaf_idx_data;
    }

    // releasing the memory
    for (int i=0; i<leaves_count; ++i)
    {
        leaves[i] = 0;
    }
    delete[] leaves;

}    //change_values(...);
*/
//===========================================================================

float CvGBTrees::find_optimal_value( const CvMat* _Idx )
{

    double gamma = (double)0.0;

    int* idx = _Idx->data.i;
    float* resp_data = orig_response->data.fl;
    float* cur_data = sum_response->data.fl;
    int n = get_len(_Idx);

    switch (params.loss_function_type)
    // SQUARED_LOSS=0, ABSOLUTE_LOSS=1, HUBER_LOSS=3, DEVIANCE_LOSS=4
    {
    case SQUARED_LOSS:
        {
            for (int i=0; i<n; ++i)
                gamma += resp_data[idx[i]] - cur_data[idx[i]];
            gamma /= (double)n;
        }; break;

    case ABSOLUTE_LOSS:
        {
            float* residuals = new float[n];
            for (int i=0; i<n; ++i, ++idx)
                residuals[i] = (resp_data[*idx] - cur_data[*idx]);
            icvSortFloat(residuals, n, 0.0f);
            if (n % 2) 
                gamma = residuals[n/2];
            else gamma = (residuals[n/2-1] + residuals[n/2]) / 2.0f;
            delete[] residuals;
        }; break;

    case HUBER_LOSS:
        {
            float* residuals = new float[n];
            for (int i=0; i<n; ++i, ++idx)
                residuals[i] = (resp_data[*idx] - cur_data[*idx]);
            icvSortFloat(residuals, n, 0.0f);

            int n_half = n >> 1;
            float r_median = (n == n_half<<1) ?
                        (residuals[n_half-1] + residuals[n_half]) / 2.0f :
                        residuals[n_half];

            for (int i=0; i<n; ++i)
            {
                float dif = residuals[i] - r_median;
                gamma += (delta < fabs(dif)) ? Sign(dif)*delta : dif;
            }
            gamma /= (double)n;
            gamma += r_median;
            delete[] residuals;

        }; break;

    case DEVIANCE_LOSS:
        {
            float* grad_data = data->responses->data.fl;
            double tmp1 = 0;
            double tmp2 = 0;
            double tmp  = 0;
            for (int i=0; i<n; ++i)
            {
                tmp = grad_data[idx[i]];
                tmp1 += tmp;
                tmp2 += fabs(tmp)*(1-fabs(tmp));
            };
            if (tmp2 == 0) 
            {
                tmp2 = 1;
            }

            gamma = ((double)(class_count-1)) / (double)class_count * (tmp1/tmp2);
        }; break;

    default: break;
    }

    return float(gamma);

} // CvGBTrees::find_optimal_value

//===========================================================================


void CvGBTrees::leaves_get( CvDTreeNode** leaves, int& count, CvDTreeNode* node )
{
    if (node->left != NULL)  leaves_get(leaves, count, node->left);
    if (node->right != NULL) leaves_get(leaves, count, node->right);
    if ((node->left == NULL) && (node->right == NULL))
        leaves[count++] = node;
}

//---------------------------------------------------------------------------

CvDTreeNode** CvGBTrees::GetLeaves( const CvDTree* dtree, int& len )
{
    len = 0;
    CvDTreeNode** leaves = new pCvDTreeNode[(size_t)1 << params.max_depth];
    leaves_get(leaves, len, const_cast<pCvDTreeNode>(dtree->get_root()));
    return leaves;
}

//===========================================================================

void CvGBTrees::do_subsample()
{

    int n = get_len(sample_idx);
    int* idx = subsample_train->data.i;

    for (int i = 0; i < n; i++ )
        idx[i] = i;

    if (subsample_test)
        for (int i = 0; i < n; i++)
        {
            int a = (*rng)(n);
            int b = (*rng)(n);
            int t;
            CV_SWAP( idx[a], idx[b], t );
        }

/*
    int n = get_len(sample_idx);
    if (subsample_train == 0)
        subsample_train = cvCreateMat(1, n, CV_32S);
    int* subsample_data = subsample_train->data.i;
    for (int i=0; i<n; ++i)
        subsample_data[i] = i;
    subsample_test = 0;
*/
}

//===========================================================================

float CvGBTrees::predict_serial( const CvMat* _sample, const CvMat* _missing,
        CvMat* weak_responses, CvSlice slice, int k) const 
{
    float result = 0.0f;

    if (!weak) return 0.0f;

    CvSeqReader reader;
    int weak_count = cvSliceLength( slice, weak[class_count-1] );
    CvDTree* tree;
    
    if (weak_responses)
    {
		if (CV_MAT_TYPE(weak_responses->type) != CV_32F)
            return 0.0f;
        if ((k >= 0) && (k<class_count) && (weak_responses->rows != 1))
            return 0.0f;
        if ((k == -1) && (weak_responses->rows != class_count))
            return 0.0f;
        if (weak_responses->cols != weak_count)
            return 0.0f;
    }
    
    float* sum = new float[class_count];
    memset(sum, 0, class_count*sizeof(float));

    for (int i=0; i<class_count; ++i)
    {
        if ((weak[i]) && (weak_count))
        {
            cvStartReadSeq( weak[i], &reader ); 
            cvSetSeqReaderPos( &reader, slice.start_index );
            for (int j=0; j<weak_count; ++j)
            {
                CV_READ_SEQ_ELEM( tree, reader );
                float p = (float)(tree->predict(_sample, _missing)->value);
                sum[i] += params.shrinkage * p;
                if (weak_responses)
                    weak_responses->data.fl[i*weak_count+j] = p;
            }
        }
    }
    
    for (int i=0; i<class_count; ++i)
        sum[i] += base_value;

    if (class_count == 1)
    {
        result = sum[0];
        delete[] sum;
        return result;
    }

    if ((k>=0) && (k<class_count))
    {
        result = sum[k];
        delete[] sum;
        return result;
    }

    float max = sum[0];
    int class_label = 0;
    for (int i=1; i<class_count; ++i)
        if (sum[i] > max)
        {
            max = sum[i];
            class_label = i;
        }

    delete[] sum;

	/*
    int orig_class_label = -1;
    for (int i=0; i<get_len(class_labels); ++i)
        if (class_labels->data.i[i] == class_label+1)
            orig_class_label = i;
	*/
	int orig_class_label = class_labels->data.i[class_label];

    return float(orig_class_label);
}


class Tree_predictor
{
private:
	pCvSeq* weak;
	float* sum;
	const int k;
	const CvMat* sample;
	const CvMat* missing;
    const float shrinkage;
    
#ifdef HAVE_TBB
    static tbb::spin_mutex SumMutex;
#endif


public:
	Tree_predictor() : weak(0), sum(0), k(0), sample(0), missing(0), shrinkage(1.0f) {}
	Tree_predictor(pCvSeq* _weak, const int _k, const float _shrinkage,
				   const CvMat* _sample, const CvMat* _missing, float* _sum ) :
				   weak(_weak), k(_k), sample(_sample),
                   missing(_missing), sum(_sum), shrinkage(_shrinkage)
	{}
	
    Tree_predictor( const Tree_predictor& p, cv::Split ) :
			weak(p.weak), k(p.k), sample(p.sample),
            missing(p.missing), sum(p.sum), shrinkage(p.shrinkage)
	{}

	Tree_predictor& operator=( const Tree_predictor& )
	{}
	
    virtual void operator()(const cv::BlockedRange& range) const
	{
#ifdef HAVE_TBB
        tbb::spin_mutex::scoped_lock lock;
#endif
        CvSeqReader reader;
		int begin = range.begin();
		int end = range.end();
		
		int weak_count = end - begin;
		CvDTree* tree;

		for (int i=0; i<k; ++i)
		{
			float tmp_sum = 0.0f;
			if ((weak[i]) && (weak_count))
			{
				cvStartReadSeq( weak[i], &reader ); 
				cvSetSeqReaderPos( &reader, begin );
				for (int j=0; j<weak_count; ++j)
				{
					CV_READ_SEQ_ELEM( tree, reader );
					tmp_sum += shrinkage*(float)(tree->predict(sample, missing)->value);
				}
			}
#ifdef HAVE_TBB
            lock.acquire(SumMutex);
			sum[i] += tmp_sum;
            lock.release();
#else
            sum[i] += tmp_sum;
#endif
		}
	} // Tree_predictor::operator()
    
}; // class Tree_predictor


#ifdef HAVE_TBB
tbb::spin_mutex Tree_predictor::SumMutex;
#endif



float CvGBTrees::predict( const CvMat* _sample, const CvMat* _missing,
            CvMat* /*weak_responses*/, CvSlice slice, int k) const 
    {
        float result = 0.0f;
	    if (!weak) return 0.0f;
        float* sum = new float[class_count];
        for (int i=0; i<class_count; ++i)
            sum[i] = 0.0f;
	    int begin = slice.start_index;
	    int end = begin + cvSliceLength( slice, weak[0] );
    	
        pCvSeq* weak_seq = weak;
	    Tree_predictor predictor = Tree_predictor(weak_seq, class_count,
                                    params.shrinkage, _sample, _missing, sum);
        
//#ifdef HAVE_TBB
//		tbb::parallel_for(cv::BlockedRange(begin, end), predictor,
//                          tbb::auto_partitioner());
//#else
        cv::parallel_for(cv::BlockedRange(begin, end), predictor);
//#endif

	    for (int i=0; i<class_count; ++i)
            sum[i] = sum[i] /** params.shrinkage*/ + base_value;

        if (class_count == 1)
        {
            result = sum[0];
            delete[] sum;
            return result;
        }

        if ((k>=0) && (k<class_count))
        {
            result = sum[k];
            delete[] sum;
            return result;
        }

        float max = sum[0];
        int class_label = 0;
        for (int i=1; i<class_count; ++i)
            if (sum[i] > max)
            {
                max = sum[i];
                class_label = i;
            }

        delete[] sum;
        int orig_class_label = class_labels->data.i[class_label];

        return float(orig_class_label);
    }


//===========================================================================

void CvGBTrees::write_params( CvFileStorage* fs ) const
{
    const char* loss_function_type_str =
        params.loss_function_type == SQUARED_LOSS ? "SquaredLoss" :
        params.loss_function_type == ABSOLUTE_LOSS ? "AbsoluteLoss" :
        params.loss_function_type == HUBER_LOSS ? "HuberLoss" :
        params.loss_function_type == DEVIANCE_LOSS ? "DevianceLoss" : 0;


    if( loss_function_type_str )
        cvWriteString( fs, "loss_function", loss_function_type_str );
    else
        cvWriteInt( fs, "loss_function", params.loss_function_type );

    cvWriteInt( fs, "ensemble_length", params.weak_count );
    cvWriteReal( fs, "shrinkage", params.shrinkage );
    cvWriteReal( fs, "subsample_portion", params.subsample_portion );
    //cvWriteInt( fs, "max_tree_depth", params.max_depth );
    //cvWriteString( fs, "use_surrogate_splits", params.use_surrogates ? "true" : "false");
    if (class_labels) cvWrite( fs, "class_labels", class_labels);

    data->is_classifier = !problem_type();
    data->write_params( fs );
    data->is_classifier = 0;
}


//===========================================================================

void CvGBTrees::read_params( CvFileStorage* fs, CvFileNode* fnode )
{
    CV_FUNCNAME( "CvGBTrees::read_params" );
    __BEGIN__;


    CvFileNode* temp;

    if( !fnode || !CV_NODE_IS_MAP(fnode->tag) )
        return;

    data = new CvDTreeTrainData();
    CV_CALL( data->read_params(fs, fnode));
    data->shared = true;

    params.max_depth = data->params.max_depth;
    params.min_sample_count = data->params.min_sample_count;
    params.max_categories = data->params.max_categories;
    params.priors = data->params.priors;
    params.regression_accuracy = data->params.regression_accuracy;
    params.use_surrogates = data->params.use_surrogates;

    temp = cvGetFileNodeByName( fs, fnode, "loss_function" );
    if( !temp )
        EXIT;

    if( temp && CV_NODE_IS_STRING(temp->tag) )
    {
        const char* loss_function_type_str = cvReadString( temp, "" );
        params.loss_function_type = strcmp( loss_function_type_str, "SquaredLoss" ) == 0 ? SQUARED_LOSS :
                            strcmp( loss_function_type_str, "AbsoluteLoss" ) == 0 ? ABSOLUTE_LOSS :
                            strcmp( loss_function_type_str, "HuberLoss" ) == 0 ? HUBER_LOSS :
                            strcmp( loss_function_type_str, "DevianceLoss" ) == 0 ? DEVIANCE_LOSS : -1;
    }
    else
        params.loss_function_type = cvReadInt( temp, -1 );


    if( params.loss_function_type < SQUARED_LOSS || params.loss_function_type > DEVIANCE_LOSS ||  params.loss_function_type == 2)
        CV_ERROR( CV_StsBadArg, "Unknown loss function" );

    params.weak_count = cvReadIntByName( fs, fnode, "ensemble_length" );
    params.shrinkage = (float)cvReadRealByName( fs, fnode, "shrinkage", 0.1 );
    params.subsample_portion = (float)cvReadRealByName( fs, fnode, "subsample_portion", 1.0 );

    if (data->is_classifier)
    {
        class_labels = (CvMat*)cvReadByName( fs, fnode, "class_labels" );
        if( class_labels && !CV_IS_MAT(class_labels))
            CV_ERROR( CV_StsParseError, "class_labels must stored as a matrix");
    }
    data->is_classifier = 0;

    __END__;
}




void CvGBTrees::write( CvFileStorage* fs, const char* name ) const
{
    CV_FUNCNAME( "CvGBTrees::write" );

    __BEGIN__;

    CvSeqReader reader;
    int i;
    std::string s;

    cvStartWriteStruct( fs, name, CV_NODE_MAP, CV_TYPE_NAME_ML_GBT );

    if( !weak )
        CV_ERROR( CV_StsBadArg, "The model has not been trained yet" );

    write_params( fs );
    cvWriteReal( fs, "base_value", base_value);
    cvWriteInt( fs, "class_count", class_count);

    for ( int j=0; j < class_count; ++j )
    {
        s = "trees_";
        s += ToString(j);
        cvStartWriteStruct( fs, s.c_str(), CV_NODE_SEQ );

        cvStartReadSeq( weak[j], &reader );

        for( i = 0; i < weak[j]->total; i++ )
        {
            CvDTree* tree;
            CV_READ_SEQ_ELEM( tree, reader );
            cvStartWriteStruct( fs, 0, CV_NODE_MAP );
            tree->write( fs );
            cvEndWriteStruct( fs );
        }

        cvEndWriteStruct( fs );
    }

    cvEndWriteStruct( fs );

    __END__;
}


//===========================================================================


void CvGBTrees::read( CvFileStorage* fs, CvFileNode* node )
{
  
    CV_FUNCNAME( "CvGBTrees::read" );

    __BEGIN__;

    CvSeqReader reader;
    CvFileNode* trees_fnode;
    CvMemStorage* storage;
    int i, ntrees;
    std::string s;

    clear();
    read_params( fs, node );

    if( !data )
        EXIT;

    base_value = (float)cvReadRealByName( fs, node, "base_value", 0.0 );
    class_count = cvReadIntByName( fs, node, "class_count", 1 );

    weak = new pCvSeq[class_count];


    for (int j=0; j<class_count; ++j)
    { 
        s = "trees_";
        s += ToString(j);

        trees_fnode = cvGetFileNodeByName( fs, node, s.c_str() );
        if( !trees_fnode || !CV_NODE_IS_SEQ(trees_fnode->tag) )
            CV_ERROR( CV_StsParseError, "<trees_x> tag is missing" );

        cvStartReadSeq( trees_fnode->data.seq, &reader );
        ntrees = trees_fnode->data.seq->total;

        if( ntrees != params.weak_count )
            CV_ERROR( CV_StsUnmatchedSizes,
            "The number of trees stored does not match <ntrees> tag value" );

        CV_CALL( storage = cvCreateMemStorage() );
        weak[j] = cvCreateSeq( 0, sizeof(CvSeq), sizeof(CvDTree*), storage );

        for( i = 0; i < ntrees; i++ )
        {
            CvDTree* tree = new CvDTree();
            CV_CALL(tree->read( fs, (CvFileNode*)reader.ptr, data ));
            CV_NEXT_SEQ_ELEM( reader.seq->elem_size, reader );
            cvSeqPush( weak[j], &tree );
        }
    }

    __END__;
}

//===========================================================================

class Sample_predictor
{
private:
	const CvGBTrees* gbt;
	float* predictions;
	const CvMat* samples;
	const CvMat* missing;
    const CvMat* idx;
    CvSlice slice;

public:
	Sample_predictor() : gbt(0), predictions(0), samples(0), missing(0),
                         idx(0), slice(CV_WHOLE_SEQ)
    {}

	Sample_predictor(const CvGBTrees* _gbt, float* _predictions,
				   const CvMat* _samples, const CvMat* _missing,
                   const CvMat* _idx, CvSlice _slice=CV_WHOLE_SEQ) :
				   gbt(_gbt), predictions(_predictions), samples(_samples),
                   missing(_missing), idx(_idx), slice(_slice)
	{}
	

    Sample_predictor( const Sample_predictor& p, cv::Split ) :
			gbt(p.gbt), predictions(p.predictions),
            samples(p.samples), missing(p.missing), idx(p.idx),
            slice(p.slice)
	{}


    virtual void operator()(const cv::BlockedRange& range) const
	{
		int begin = range.begin();
		int end = range.end();

		CvMat x;
        CvMat miss;

        for (int i=begin; i<end; ++i)
        {
            int j = idx ? idx->data.i[i] : i;
            cvGetRow(samples, &x, j);
            if (!missing)
            {
                predictions[i] = gbt->predict_serial(&x,0,0,slice);
            }
            else
            {
                cvGetRow(missing, &miss, j);
                predictions[i] = gbt->predict_serial(&x,&miss,0,slice);
            }
        }
	} // Sample_predictor::operator()

}; // class Sample_predictor



// type in {CV_TRAIN_ERROR, CV_TEST_ERROR}
float 
CvGBTrees::calc_error( CvMLData* _data, int type, std::vector<float> *resp )
{

    float err = 0.0f;
    const CvMat* sample_idx = (type == CV_TRAIN_ERROR) ?
                              _data->get_train_sample_idx() :
                              _data->get_test_sample_idx();
    const CvMat* response = _data->get_responses();
                              
    int n = sample_idx ? get_len(sample_idx) : 0;
    n = (type == CV_TRAIN_ERROR && n == 0) ? _data->get_values()->rows : n;
    
    if (!n)
        return -FLT_MAX;
    
    float* pred_resp = 0;  
    if (resp)
    {
        resp->resize(n);
        pred_resp = &((*resp)[0]);
    }
    else
        pred_resp = new float[n];

    Sample_predictor predictor = Sample_predictor(this, pred_resp, _data->get_values(),
            _data->get_missing(), sample_idx);
        
//#ifdef HAVE_TBB
//    tbb::parallel_for(cv::BlockedRange(0,n), predictor, tbb::auto_partitioner());
//#else
    cv::parallel_for(cv::BlockedRange(0,n), predictor);
//#endif
        
    int* sidx = sample_idx ? sample_idx->data.i : 0;
    int r_step = CV_IS_MAT_CONT(response->type) ?
                1 : response->step / CV_ELEM_SIZE(response->type);
    

    if ( !problem_type() )
    {
        for( int i = 0; i < n; i++ )
        {
            int si = sidx ? sidx[i] : i;
            int d = fabs((double)pred_resp[i] - response->data.fl[si*r_step]) <= FLT_EPSILON ? 0 : 1;
            err += d;
        }
        err = err / (float)n * 100.0f;
    }
    else
    {
        for( int i = 0; i < n; i++ )
        {
            int si = sidx ? sidx[i] : i;
            float d = pred_resp[i] - response->data.fl[si*r_step];
            err += d*d;
        }
        err = err / (float)n;    
    }
    
    return err;
}


CvGBTrees::CvGBTrees( const cv::Mat& trainData, int tflag,
          const cv::Mat& responses, const cv::Mat& varIdx,
          const cv::Mat& sampleIdx, const cv::Mat& varType,
          const cv::Mat& missingDataMask,
          CvGBTreesParams params )
{
    data = 0;
    weak = 0;
    default_model_name = "my_boost_tree";
    orig_response = sum_response = sum_response_tmp = 0;
    subsample_train = subsample_test = 0;
    missing = sample_idx = 0;
    class_labels = 0;
    class_count = 1;
    delta = 0.0f;
    
    clear();
    
    train(trainData, tflag, responses, varIdx, sampleIdx, varType, missingDataMask, params, false);
}

bool CvGBTrees::train( const cv::Mat& trainData, int tflag,
                   const cv::Mat& responses, const cv::Mat& varIdx,
                   const cv::Mat& sampleIdx, const cv::Mat& varType,
                   const cv::Mat& missingDataMask,
                   CvGBTreesParams params,
                   bool update )
{
    CvMat _trainData = trainData, _responses = responses;
    CvMat _varIdx = varIdx, _sampleIdx = sampleIdx, _varType = varType;
    CvMat _missingDataMask = missingDataMask;
    
    return train( &_trainData, tflag, &_responses, varIdx.empty() ? 0 : &_varIdx,
                  sampleIdx.empty() ? 0 : &_sampleIdx, varType.empty() ? 0 : &_varType,
                  missingDataMask.empty() ? 0 : &_missingDataMask, params, update);
}

float CvGBTrees::predict( const cv::Mat& sample, const cv::Mat& missing,
                          const cv::Range& slice, int k ) const
{
    CvMat _sample = sample, _missing = missing;
    return predict(&_sample, missing.empty() ? 0 : &_missing, 0,
                   slice==cv::Range::all() ? CV_WHOLE_SEQ : cvSlice(slice.start, slice.end), k);
}
