// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// AUTHOR: Rahul Kavi rahulkavi[at]live[at]com

//
// This is a implementation of the Logistic Regression algorithm
//

#include "precomp.hpp"

using namespace std;

namespace cv {
namespace ml {

class LrParams
{
public:
    LrParams()
    {
        alpha = 0.001;
        num_iters = 1000;
        norm = LogisticRegression::REG_L2;
        train_method = LogisticRegression::BATCH;
        mini_batch_size = 1;
        term_crit = TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, num_iters, alpha);
    }

    double alpha; //!< learning rate.
    int num_iters; //!< number of iterations.
    int norm;
    int train_method;
    int mini_batch_size;
    TermCriteria term_crit;
};

class LogisticRegressionImpl CV_FINAL : public LogisticRegression
{
public:

    LogisticRegressionImpl() { }
    virtual ~LogisticRegressionImpl() {}

    inline double getLearningRate() const CV_OVERRIDE { return params.alpha; }
    inline void setLearningRate(double val) CV_OVERRIDE { params.alpha = val; }
    inline int getIterations() const CV_OVERRIDE { return params.num_iters; }
    inline void setIterations(int val) CV_OVERRIDE { params.num_iters = val; }
    inline int getRegularization() const CV_OVERRIDE { return params.norm; }
    inline void setRegularization(int val) CV_OVERRIDE { params.norm = val; }
    inline int getTrainMethod() const CV_OVERRIDE { return params.train_method; }
    inline void setTrainMethod(int val) CV_OVERRIDE { params.train_method = val; }
    inline int getMiniBatchSize() const CV_OVERRIDE { return params.mini_batch_size; }
    inline void setMiniBatchSize(int val) CV_OVERRIDE { params.mini_batch_size = val; }
    inline TermCriteria getTermCriteria() const CV_OVERRIDE { return params.term_crit; }
    inline void setTermCriteria(TermCriteria val) CV_OVERRIDE { params.term_crit = val; }

    virtual bool train( const Ptr<TrainData>& trainData, int=0 ) CV_OVERRIDE;
    virtual float predict(InputArray samples, OutputArray results, int flags=0) const CV_OVERRIDE;
    virtual void clear() CV_OVERRIDE;
    virtual void write(FileStorage& fs) const CV_OVERRIDE;
    virtual void read(const FileNode& fn) CV_OVERRIDE;
    virtual Mat get_learnt_thetas() const CV_OVERRIDE { return learnt_thetas; }
    virtual int getVarCount() const CV_OVERRIDE { return learnt_thetas.cols; }
    virtual bool isTrained() const CV_OVERRIDE { return !learnt_thetas.empty(); }
    virtual bool isClassifier() const CV_OVERRIDE { return true; }
    virtual String getDefaultName() const CV_OVERRIDE { return "opencv_ml_lr"; }
protected:
    Mat calc_sigmoid(const Mat& data) const;
    double compute_cost(const Mat& _data, const Mat& _labels, const Mat& _init_theta);
    void compute_gradient(const Mat& _data, const Mat& _labels, const Mat &_theta, const double _lambda, Mat & _gradient );
    Mat batch_gradient_descent(const Mat& _data, const Mat& _labels, const Mat& _init_theta);
    Mat mini_batch_gradient_descent(const Mat& _data, const Mat& _labels, const Mat& _init_theta);
    bool set_label_map(const Mat& _labels_i);
    Mat remap_labels(const Mat& _labels_i, const map<int, int>& lmap) const;
protected:
    LrParams params;
    Mat learnt_thetas;
    map<int, int> forward_mapper;
    map<int, int> reverse_mapper;
    Mat labels_o;
    Mat labels_n;
};

Ptr<LogisticRegression> LogisticRegression::create()
{
    return makePtr<LogisticRegressionImpl>();
}

Ptr<LogisticRegression> LogisticRegression::load(const String& filepath, const String& nodeName)
{
    return Algorithm::load<LogisticRegression>(filepath, nodeName);
}


bool LogisticRegressionImpl::train(const Ptr<TrainData>& trainData, int)
{
    CV_TRACE_FUNCTION_SKIP_NESTED();
    CV_Assert(!trainData.empty());

    // return value
    bool ok = false;
    clear();
    Mat _data_i = trainData->getSamples();
    Mat _labels_i = trainData->getResponses();

    // check size and type of training data
    CV_Assert( !_labels_i.empty() && !_data_i.empty());
    if(_labels_i.cols != 1)
    {
        CV_Error( cv::Error::StsBadArg, "labels should be a column matrix" );
    }
    if(_data_i.type() != CV_32FC1 || _labels_i.type() != CV_32FC1)
    {
        CV_Error( cv::Error::StsBadArg, "data and labels must be a floating point matrix" );
    }
    if(_labels_i.rows != _data_i.rows)
    {
        CV_Error( cv::Error::StsBadArg, "number of rows in data and labels should be equal" );
    }

    // class labels
    set_label_map(_labels_i);
    Mat labels_l = remap_labels(_labels_i, this->forward_mapper);
    int num_classes = (int) this->forward_mapper.size();
    if(num_classes < 2)
    {
        CV_Error( cv::Error::StsBadArg, "data should have at least 2 classes" );
    }

    // add a column of ones to the data (bias/intercept term)
    Mat data_t;
    hconcat( cv::Mat::ones( _data_i.rows, 1, CV_32F ), _data_i, data_t );

    // coefficient matrix (zero-initialized)
    Mat thetas;
    Mat init_theta = Mat::zeros(data_t.cols, 1, CV_32F);

    // fit the model (handles binary and multiclass cases)
    Mat new_theta;
    Mat labels;
    if(num_classes == 2)
    {
        labels_l.convertTo(labels, CV_32F);
        if(this->params.train_method == LogisticRegression::BATCH)
            new_theta = batch_gradient_descent(data_t, labels, init_theta);
        else
            new_theta = mini_batch_gradient_descent(data_t, labels, init_theta);
        thetas = new_theta.t();
    }
    else
    {
        /* take each class and rename classes you will get a theta per class
        as in multi class class scenario, we will have n thetas for n classes */
        thetas.create(num_classes, data_t.cols, CV_32F);
        Mat labels_binary;
        int ii = 0;
        for(map<int,int>::iterator it = this->forward_mapper.begin(); it != this->forward_mapper.end(); ++it)
        {
            // one-vs-rest (OvR) scheme
            labels_binary = (labels_l == it->second)/255;
            labels_binary.convertTo(labels, CV_32F);
            if(this->params.train_method == LogisticRegression::BATCH)
                new_theta = batch_gradient_descent(data_t, labels, init_theta);
            else
                new_theta = mini_batch_gradient_descent(data_t, labels, init_theta);
            hconcat(new_theta.t(), thetas.row(ii));
            ii += 1;
        }
    }

    // check that the estimates are stable and finite
    this->learnt_thetas = thetas.clone();
    if( cvIsNaN( (double)sum(this->learnt_thetas)[0] ) )
    {
        CV_Error( cv::Error::StsBadArg, "check training parameters. Invalid training classifier" );
    }

    // success
    ok = true;
    return ok;
}

float LogisticRegressionImpl::predict(InputArray samples, OutputArray results, int flags) const
{
    // check if learnt_mats array is populated
    if(!this->isTrained())
    {
        CV_Error( cv::Error::StsBadArg, "classifier should be trained first" );
    }

    // coefficient matrix
    Mat thetas;
    if ( learnt_thetas.type() == CV_32F )
    {
        thetas = learnt_thetas;
    }
    else
    {
        this->learnt_thetas.convertTo( thetas, CV_32F );
    }
    CV_Assert(thetas.rows > 0);

    // data samples
    Mat data = samples.getMat();
    if(data.type() != CV_32F)
    {
        CV_Error( cv::Error::StsBadArg, "data must be of floating type" );
    }

    // add a column of ones to the data (bias/intercept term)
    Mat data_t;
    hconcat( cv::Mat::ones( data.rows, 1, CV_32F ), data, data_t );
    CV_Assert(data_t.cols == thetas.cols);

    // predict class labels for samples (handles binary and multiclass cases)
    Mat labels_c;
    Mat pred_m;
    Mat temp_pred;
    if(thetas.rows == 1)
    {
        // apply sigmoid function
        temp_pred = calc_sigmoid(data_t * thetas.t());
        CV_Assert(temp_pred.cols==1);
        pred_m = temp_pred.clone();

        // if greater than 0.5, predict class 0 or predict class 1
        temp_pred = (temp_pred > 0.5f) / 255;
        temp_pred.convertTo(labels_c, CV_32S);
    }
    else
    {
        // apply sigmoid function
        pred_m.create(data_t.rows, thetas.rows, data.type());
        for(int i = 0; i < thetas.rows; i++)
        {
            temp_pred = calc_sigmoid(data_t * thetas.row(i).t());
            vconcat(temp_pred, pred_m.col(i));
        }

        // predict class with the maximum output
        Point max_loc;
        Mat labels;
        for(int i = 0; i < pred_m.rows; i++)
        {
            temp_pred = pred_m.row(i);
            minMaxLoc( temp_pred, NULL, NULL, NULL, &max_loc );
            labels.push_back(max_loc.x);
        }
        labels.convertTo(labels_c, CV_32S);
    }

    // return label of the predicted class. class names can be 1,2,3,...
    Mat pred_labs = remap_labels(labels_c, this->reverse_mapper);
    pred_labs.convertTo(pred_labs, CV_32S);

    // return either the labels or the raw output
    if ( results.needed() )
    {
        if ( flags & StatModel::RAW_OUTPUT )
        {
            pred_m.copyTo( results );
        }
        else
        {
            pred_labs.copyTo(results);
        }
    }

    return ( pred_labs.empty() ? 0.f : static_cast<float>(pred_labs.at<int>(0)) );
}

Mat LogisticRegressionImpl::calc_sigmoid(const Mat& data) const
{
    CV_TRACE_FUNCTION();
    Mat dest;
    exp(-data, dest);
    return 1.0/(1.0+dest);
}

double LogisticRegressionImpl::compute_cost(const Mat& _data, const Mat& _labels, const Mat& _init_theta)
{
    CV_TRACE_FUNCTION();
    float llambda = 0;                   /*changed llambda from int to float to solve issue #7924*/
    int m;
    int n;
    double cost = 0;
    double rparameter = 0;
    Mat theta_b;
    Mat theta_c;
    Mat d_a;
    Mat d_b;

    m = _data.rows;
    n = _data.cols;

    theta_b = _init_theta(Range(1, n), Range::all());

    if (params.norm != REG_DISABLE)
    {
        llambda = 1;
    }

    if(this->params.norm == LogisticRegression::REG_L1)
    {
        rparameter = (llambda/(2*m)) * sum(theta_b)[0];
    }
    else
    {
        // assuming it to be L2 by default
        multiply(theta_b, theta_b, theta_c, 1);
        rparameter = (llambda/(2*m)) * sum(theta_c)[0];
    }

    d_a = calc_sigmoid(_data * _init_theta);
    log(d_a, d_a);
    multiply(d_a, _labels, d_a);

    // use the fact that: log(1 - sigmoid(x)) = log(sigmoid(-x))
    d_b = calc_sigmoid(- _data * _init_theta);
    log(d_b, d_b);
    multiply(d_b, 1-_labels, d_b);

    cost = (-1.0/m) * (sum(d_a)[0] + sum(d_b)[0]);
    cost = cost + rparameter;

    if(cvIsNaN( cost ) == 1)
    {
        CV_Error( cv::Error::StsBadArg, "check training parameters. Invalid training classifier" );
    }

    return cost;
}

struct LogisticRegressionImpl_ComputeDradient_Impl : ParallelLoopBody
{
    const Mat* data;
    const Mat* theta;
    const Mat* pcal_a;
    Mat* gradient;
    double lambda;

    LogisticRegressionImpl_ComputeDradient_Impl(const Mat& _data, const Mat &_theta, const Mat& _pcal_a, const double _lambda, Mat & _gradient)
        : data(&_data)
        , theta(&_theta)
        , pcal_a(&_pcal_a)
        , gradient(&_gradient)
        , lambda(_lambda)
    {

    }

    void operator()(const cv::Range& r) const CV_OVERRIDE
    {
        const Mat& _data  = *data;
        const Mat &_theta = *theta;
        Mat & _gradient   = *gradient;
        const Mat & _pcal_a = *pcal_a;
        const int m = _data.rows;
        Mat pcal_ab;

        for (int ii = r.start; ii<r.end; ii++)
        {
            Mat pcal_b = _data(Range::all(), Range(ii,ii+1));
            multiply(_pcal_a, pcal_b, pcal_ab, 1);

            _gradient.row(ii) = (1.0/m)*sum(pcal_ab)[0] + (lambda/m) * _theta.row(ii);
        }
    }
};

void LogisticRegressionImpl::compute_gradient(const Mat& _data, const Mat& _labels, const Mat &_theta, const double _lambda, Mat & _gradient )
{
    CV_TRACE_FUNCTION();
    const int m = _data.rows;
    Mat pcal_a, pcal_b, pcal_ab;

    const Mat z = _data * _theta;

    CV_Assert( _gradient.rows == _theta.rows && _gradient.cols == _theta.cols );

    pcal_a = calc_sigmoid(z) - _labels;
    pcal_b = _data(Range::all(), Range(0,1));
    multiply(pcal_a, pcal_b, pcal_ab, 1);

    _gradient.row(0) = ((float)1/m) * sum(pcal_ab)[0];

    //cout<<"for each training data entry"<<endl;
    LogisticRegressionImpl_ComputeDradient_Impl invoker(_data, _theta, pcal_a, _lambda, _gradient);
    cv::parallel_for_(cv::Range(1, _gradient.rows), invoker);
}


Mat LogisticRegressionImpl::batch_gradient_descent(const Mat& _data, const Mat& _labels, const Mat& _init_theta)
{
    CV_TRACE_FUNCTION();
    // implements batch gradient descent
    if(this->params.alpha<=0)
    {
        CV_Error( cv::Error::StsBadArg, "check training parameters (learning rate) for the classifier" );
    }

    if(this->params.num_iters <= 0)
    {
        CV_Error( cv::Error::StsBadArg, "number of iterations cannot be zero or a negative number" );
    }

    int llambda = 0;
    int m;
    Mat theta_p = _init_theta.clone();
    Mat gradient( theta_p.rows, theta_p.cols, theta_p.type() );
    m = _data.rows;

    if (params.norm != REG_DISABLE)
    {
        llambda = 1;
    }

    for(int i = 0;i<this->params.num_iters;i++)
    {
        // this seems to only be called to ensure that cost is not NaN
        compute_cost(_data, _labels, theta_p);

        compute_gradient( _data, _labels, theta_p, llambda, gradient );

        theta_p = theta_p - ( static_cast<double>(this->params.alpha)/m)*gradient;
    }
    return theta_p;
}

Mat LogisticRegressionImpl::mini_batch_gradient_descent(const Mat& _data, const Mat& _labels, const Mat& _init_theta)
{
    // implements batch gradient descent
    int lambda_l = 0;
    int m;
    int j = 0;
    int size_b = this->params.mini_batch_size;

    if(this->params.mini_batch_size <= 0 || this->params.alpha == 0)
    {
        CV_Error( cv::Error::StsBadArg, "check training parameters for the classifier" );
    }

    if(this->params.num_iters <= 0)
    {
        CV_Error( cv::Error::StsBadArg, "number of iterations cannot be zero or a negative number" );
    }

    Mat theta_p = _init_theta.clone();
    Mat gradient( theta_p.rows, theta_p.cols, theta_p.type() );
    Mat data_d;
    Mat labels_l;

    if (params.norm != REG_DISABLE)
    {
        lambda_l = 1;
    }

    for(int i = 0;i<this->params.term_crit.maxCount;i++)
    {
        if(j+size_b<=_data.rows)
        {
            data_d = _data(Range(j,j+size_b), Range::all());
            labels_l = _labels(Range(j,j+size_b),Range::all());
        }
        else
        {
            data_d = _data(Range(j, _data.rows), Range::all());
            labels_l = _labels(Range(j, _labels.rows),Range::all());
        }

        m = data_d.rows;

        // this seems to only be called to ensure that cost is not NaN
        compute_cost(data_d, labels_l, theta_p);

        compute_gradient(data_d, labels_l, theta_p, lambda_l, gradient);

        theta_p = theta_p - ( static_cast<double>(this->params.alpha)/m)*gradient;

        j += this->params.mini_batch_size;

        // if parsed through all data variables
        if (j >= _data.rows) {
            j = 0;
        }
    }
    return theta_p;
}

bool LogisticRegressionImpl::set_label_map(const Mat &_labels_i)
{
    // this function creates two maps to map user defined labels to program friendly labels two ways.
    int ii = 0;
    Mat labels;

    this->labels_o = Mat(0,1, CV_8U);
    this->labels_n = Mat(0,1, CV_8U);

    _labels_i.convertTo(labels, CV_32S);

    for(int i = 0;i<labels.rows;i++)
    {
        this->forward_mapper[labels.at<int>(i)] += 1;
    }

    for(map<int,int>::iterator it = this->forward_mapper.begin(); it != this->forward_mapper.end(); ++it)
    {
        this->forward_mapper[it->first] = ii;
        this->labels_o.push_back(it->first);
        this->labels_n.push_back(ii);
        ii += 1;
    }

    for(map<int,int>::iterator it = this->forward_mapper.begin(); it != this->forward_mapper.end(); ++it)
    {
        this->reverse_mapper[it->second] = it->first;
    }

    return true;
}

Mat LogisticRegressionImpl::remap_labels(const Mat& _labels_i, const map<int, int>& lmap) const
{
    Mat labels;
    _labels_i.convertTo(labels, CV_32S);

    Mat new_labels = Mat::zeros(labels.rows, labels.cols, labels.type());

    CV_Assert( !lmap.empty() );

    for(int i =0;i<labels.rows;i++)
    {
        map<int, int>::const_iterator val = lmap.find(labels.at<int>(i,0));
        CV_Assert(val != lmap.end());
        new_labels.at<int>(i,0) = val->second;
    }
    return new_labels;
}

void LogisticRegressionImpl::clear()
{
    this->learnt_thetas.release();
    this->labels_o.release();
    this->labels_n.release();
}

void LogisticRegressionImpl::write(FileStorage& fs) const
{
    // check if open
    if(fs.isOpened() == 0)
    {
        CV_Error(cv::Error::StsBadArg,"file can't open. Check file path");
    }
    writeFormat(fs);
    string desc = "Logistic Regression Classifier";
    fs<<"classifier"<<desc.c_str();
    fs<<"alpha"<<this->params.alpha;
    fs<<"iterations"<<this->params.num_iters;
    fs<<"norm"<<this->params.norm;
    fs<<"train_method"<<this->params.train_method;
    if(this->params.train_method == LogisticRegression::MINI_BATCH)
    {
        fs<<"mini_batch_size"<<this->params.mini_batch_size;
    }
    fs<<"learnt_thetas"<<this->learnt_thetas;
    fs<<"n_labels"<<this->labels_n;
    fs<<"o_labels"<<this->labels_o;
}

void LogisticRegressionImpl::read(const FileNode& fn)
{
    // check if empty
    if(fn.empty())
    {
        CV_Error( cv::Error::StsBadArg, "empty FileNode object" );
    }

    this->params.alpha = (double)fn["alpha"];
    this->params.num_iters = (int)fn["iterations"];
    this->params.norm = (int)fn["norm"];
    this->params.train_method = (int)fn["train_method"];

    if(this->params.train_method == LogisticRegression::MINI_BATCH)
    {
        this->params.mini_batch_size = (int)fn["mini_batch_size"];
    }

    fn["learnt_thetas"] >> this->learnt_thetas;
    fn["o_labels"] >> this->labels_o;
    fn["n_labels"] >> this->labels_n;

    for(int ii =0;ii<labels_o.rows;ii++)
    {
        this->forward_mapper[labels_o.at<int>(ii,0)] = labels_n.at<int>(ii,0);
        this->reverse_mapper[labels_n.at<int>(ii,0)] = labels_o.at<int>(ii,0);
    }
}

}
}

/* End of file. */
