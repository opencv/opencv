///////////////////////////////////////////////////////////////////////////////////////
// IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.

//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.

// This is a implementation of the Logistic Regression algorithm in C++ in OpenCV.

// AUTHOR:
// Rahul Kavi rahulkavi[at]live[at]com

// # You are free to use, change, or redistribute the code in any way you wish for
// # non-commercial purposes, but please maintain the name of the original author.
// # This code comes with no warranty of any kind.

// #
// # You are free to use, change, or redistribute the code in any way you wish for
// # non-commercial purposes, but please maintain the name of the original author.
// # This code comes with no warranty of any kind.

// # Logistic Regression ALGORITHM


//                           License Agreement
//                For Open Source Computer Vision Library

// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2008-2011, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.

// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:

//   * Redistributions of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.

//   * Redistributions in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.

//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.

// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.

#include "precomp.hpp"

using namespace cv;
using namespace std;

CvLR_TrainParams::CvLR_TrainParams()
{
    term_crit = CvTermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 10000, 0.001);
}

CvLR_TrainParams::CvLR_TrainParams(double _alpha, int _num_iters, int _norm, int _regularized, int _train_method, int _minibatchsize):
 alpha(_alpha), num_iters(_num_iters), norm(_norm), regularized(_regularized), train_method(_train_method), minibatchsize(_minibatchsize)
///////////////////////////////////////////////////
// CvLR_TrainParams::CvLR_TrainParams(double _alpha, int _num_iters, int _norm, int _debug, int _regularized, int _train_method, int _minibatchsize):
//  alpha(_alpha), num_iters(_num_iters), norm(_norm), debug(_debug), regularized(_regularized), train_method(_train_method), minibatchsize(_minibatchsize)
///////////////////////////////////////////////////
{
    term_crit = CvTermCriteria(TermCriteria::COUNT + TermCriteria::EPS, num_iters, 0.001);
}

CvLR_TrainParams::~CvLR_TrainParams()
{

}

CvLR::CvLR()
{
    default_model_name = "my_lr";
    // set_default_params();
}


CvLR::CvLR(const cv::Mat& _data, const cv::Mat& _labels, const CvLR_TrainParams& _params)
{
    this->params = _params;
    default_model_name = "my_lr";
    train(_data, _labels);
}

CvLR::~CvLR()
{
    clear();
}


bool CvLR::train(const cv::Mat& _data_i, const cv::Mat& _labels_i)
{
    CV_Assert( !_labels_i.empty() && !_data_i.empty());

    // check the number of colums
    CV_Assert( _labels_i.cols == 1);

    if(_labels_i.cols != 1)
    {
        cv::error(Error::StsBadArg, "_labels_i should be a column matrix", "cv::ml::CvLR::train", __FILE__, __LINE__);
    }
    // check data type.
    // data should be of floating type CV_32FC1

    if((_data_i.type() != CV_32FC1) || (_labels_i.type() != CV_32FC1))
    {
        cv::error(Error::StsBadArg, "train: data and labels must be a floating point matrix", "cv::ml::CvLR::train", __FILE__, __LINE__);
    }

    bool ok = false;

    cv::Mat labels;

    //CvLR::set_label_map(_labels_i);
    set_label_map(_labels_i);
    int num_classes = this->forward_mapper.size();

    // add a column of ones
    cv::Mat data_t = cv::Mat::zeros(_data_i.rows, _data_i.cols+1, CV_32F);
    vconcat(cv::Mat(_data_i.rows, 1, _data_i.type(), Scalar::all(1.0)), data_t.col(0));
    for (int i=1;i<data_t.cols;i++)
    {
        vconcat(_data_i.col(i-1), data_t.col(i));
    }

    if(num_classes < 2)
    {
        cv::error(Error::StsBadArg, "train: data should have atleast 2 classes", "cv::ml::CvLR::train", __FILE__, __LINE__);
    }

    if(_labels_i.rows != _data_i.rows)
    {
        cv::error(Error::StsBadArg, "train: number of rows in data and labels should be the equal", "cv::ml::CvLR::train", __FILE__, __LINE__);
    }


    cv::Mat thetas = cv::Mat::zeros(num_classes, data_t.cols, CV_32F);
    cv::Mat init_theta = cv::Mat::zeros(data_t.cols, 1, CV_32F);

    cv::Mat labels_l = remap_labels(_labels_i, this->forward_mapper);
    cv::Mat new_local_labels;

    int ii=0;

    if(num_classes == 2)
    {
        //data_t.convertTo(data, CV_32F);
        labels_l.convertTo(labels, CV_32F);

        //cv::Mat new_theta = CvLR::compute_batch_gradient(data, labels, init_theta);
        cv::Mat new_theta = compute_batch_gradient(data_t, labels, init_theta);

        thetas = new_theta.t();
    }

    else
    {
        /* take each class and rename classes you will get a theta per class
        as in multi class class scenario, we will have n thetas for n classes */
        ii = 0;

        for(map<int,int>::iterator it = this->forward_mapper.begin(); it != this->forward_mapper.end(); ++it)
        {
            new_local_labels = (labels_l == it->second)/255;
            // cout<<"processing class "<<it->second<<endl;
            // data_t.convertTo(data, CV_32F);
            new_local_labels.convertTo(labels, CV_32F);
            // cout<<"initial theta: "<<init_theta<<endl;

            cv::Mat new_theta = compute_batch_gradient(data_t, labels, init_theta);

            // cout<<"learnt theta: "<<new_theta<<endl;
            hconcat(new_theta.t(), thetas.row(ii));
            ii += 1;
        }

    }

    this->learnt_thetas = thetas.clone();
    if( cvIsNaN( (double)cv::sum(this->learnt_thetas)[0] ) )
    {
        cv::error(Error::StsBadArg, "train: check training parameters. Invalid training classifier","cv::ml::CvLR::train", __FILE__, __LINE__);
    }

    ok = true;

    return ok;
}

float CvLR::predict(const Mat& _data)
{
    cv::Mat pred_labs;
    pred_labs = cv::Mat::zeros(1,1, _data.type());

    if(_data.rows >1)
    {
        cv::error(Error::StsBadArg, "predict: _data should have only 1 row", "cv::ml::CvLR::predict", __FILE__, __LINE__);
    }


    predict(_data, pred_labs);

    return static_cast<float>(pred_labs.at<int>(0,0));
}

float CvLR::predict(const cv::Mat& _data, cv::Mat& _pred_labs)
{
    /* returns a class of the predicted class
    class names can be 1,2,3,4, .... etc */
    cv::Mat thetas;

    // check if learnt_mats array is populated
    if(this->learnt_thetas.total()<=0)
    {
        cv::error(Error::StsBadArg, "predict: classifier should be trained first", "cv::ml::CvLR::predict", __FILE__, __LINE__);
    }
    if(_data.type() != CV_32F)
    {
        cv::error(Error::StsBadArg, "predict: _data must be of floating type","cv::ml::CvLR::predict",__FILE__, __LINE__);
    }

    // add a column of ones
    cv::Mat data_t = cv::Mat::zeros(_data.rows, _data.cols+1, CV_32F);
    for (int i=0;i<data_t.cols;i++)
    {
        if(i==0)
        {
            vconcat(cv::Mat(_data.rows, 1, _data.type(), Scalar::all(1.0)), data_t.col(i));
            continue;
        }
        vconcat(_data.col(i-1), data_t.col(i));
    }

    this->learnt_thetas.convertTo(thetas, CV_32F);

    CV_Assert(thetas.rows > 0);

    double min_val;
    double max_val;

    Point min_loc;
    Point max_loc;

    cv::Mat labels;
    cv::Mat labels_c;
    cv::Mat temp_pred;

    cv::Mat pred_m = cv::Mat::zeros(data_t.rows, thetas.rows, _data.type());

    if(thetas.rows == 1)
    {
        temp_pred = calc_sigmoid(data_t*thetas.t());
        CV_Assert(temp_pred.cols==1);
        // if greater than 0.5, predict class 0 or predict class 1
        temp_pred = (temp_pred>0.5)/255;
        temp_pred.convertTo(labels_c, CV_32S);
    }

    else
    {
        for(int i = 0;i<thetas.rows;i++)
        {
            // temp_pred = CvLR::calc_sigmoid(data_t * thetas.row(i).t());
            temp_pred = calc_sigmoid(data_t * thetas.row(i).t());
            cv::vconcat(temp_pred, pred_m.col(i));
        }

        for(int i = 0;i<pred_m.rows;i++)
        {
            temp_pred = pred_m.row(i);
            minMaxLoc( temp_pred, &min_val, &max_val, &min_loc, &max_loc, Mat() );
            labels.push_back(max_loc.x);
        }

        labels.convertTo(labels_c, CV_32S);
    }

    _pred_labs = remap_labels(labels_c, this->reverse_mapper);

    // convert _pred_labs to integer type

    _pred_labs.convertTo(_pred_labs, CV_32S);

    return 0.0;
}

cv::Mat CvLR::calc_sigmoid(const Mat& data)
{
    cv::Mat dest;
    cv::exp(-data, dest);
    return 1.0/(1.0+dest);
}

double CvLR::compute_cost(const cv::Mat& _data, const cv::Mat& _labels, const cv::Mat& _init_theta)
{

    int llambda = 0;
    int m;
    int n;

    double cost = 0;
    double rparameter = 0;

    cv::Mat gradient;
    cv::Mat theta_b;
    cv::Mat theta_c;

    m = _data.rows;
    n = _data.cols;

    gradient = cv::Mat::zeros( _init_theta.rows, _init_theta.cols, _init_theta.type());

    theta_b = _init_theta(Range(1, n), Range::all());

    cv::multiply(theta_b, theta_b, theta_c, 1);

    if(this->params.regularized > 0)
    {
        llambda = 1;
    }

    if(this->params.norm == CvLR::REG_L1)
    {
        rparameter = (llambda/(2*m)) * cv::sum(theta_b)[0];
    }
    else
    {
        // assuming it to be L2 by default
        rparameter = (llambda/(2*m)) * cv::sum(theta_c)[0];
    }


    // cv::Mat d_a = LogisticRegression::CvLR::calc_sigmoid(_data* _init_theta);
    cv::Mat d_a = calc_sigmoid(_data* _init_theta);


    cv::log(d_a, d_a);
    cv::multiply(d_a, _labels, d_a);

    // cv::Mat d_b = 1 - LogisticRegression::CvLR::calc_sigmoid(_data * _init_theta);
    cv::Mat d_b = 1 - calc_sigmoid(_data * _init_theta);
    cv::log(d_b, d_b);
    cv::multiply(d_b, 1-_labels, d_b);

    cost = (-1.0/m) * (cv::sum(d_a)[0] + cv::sum(d_b)[0]);
    cost = cost + rparameter;

    return cost;
}

cv::Mat CvLR::compute_batch_gradient(const cv::Mat& _data, const cv::Mat& _labels, const cv::Mat& _init_theta)
{
    // implements batch gradient descent

    if(this->params.alpha<=0)
    {
        cv::error(Error::StsBadArg, "compute_batch_gradient: check training parameters for the classifier","cv::ml::CvLR::compute_batch_gradient", __FILE__, __LINE__);
    }

    if(this->params.num_iters <= 0)
    {
        cv::error(Error::StsBadArg,"compute_batch_gradient: number of iterations cannot be zero or a negative number","cv::ml::CvLR::compute_batch_gradient",__FILE__,__LINE__);
    }

    int llambda = 0;
    ///////////////////////////////////////////////////
    double ccost;
    ///////////////////////////////////////////////////
    int m, n;

    cv::Mat pcal_a;
    cv::Mat pcal_b;
    cv::Mat pcal_ab;
    cv::Mat gradient;
    cv::Mat theta_p = _init_theta.clone();

    // cout<<"_data size "<<_data.rows<<", "<<_data.cols<<endl;
    // cout<<"_init_theta size "<<_init_theta.rows<<", "<<_init_theta.cols<<endl;

    m = _data.rows;
    n = _data.cols;

    if(this->params.regularized > 0)
    {
        llambda = 1;
    }

    for(int i = 0;i<this->params.num_iters;i++)
    {
        ccost = compute_cost(_data, _labels, theta_p);

        if( cvIsNaN( ccost ) )
        {
            cv::error(Error::StsBadArg, "compute_batch_gradient: check training parameters. Invalid training classifier","cv::ml::CvLR::compute_batch_gradient", __FILE__, __LINE__);
        }

        ///////////////////////////////////////////////////
        // cout<<"calculated cost: "<<ccost<<endl;
        // if(this->params.debug == 1 && i%(this->params.num_iters/2)==0) //
        // {
        //     cout<<"iter: "<<i<<endl;
        //     cout<<"cost: "<<ccost<<endl;
        //     cout<<"alpha"<<this->params.alpha<<endl;
        //     cout<<"num_iters"<<this->params.num_iters<<endl;
        //     cout<<"norm"<<this->params.norm<<endl;
        //     cout<<"debug"<<this->params.debug<<endl;
        //     cout<<"regularized"<<this->params.regularized<<endl;
        //     cout<<"train_method"<<this->params.train_method<<endl;
        // }
        ///////////////////////////////////////////////////

        pcal_b = calc_sigmoid((_data*theta_p) - _labels);

        pcal_a = (static_cast<double>(1/m)) * _data.t();

        gradient = pcal_a * pcal_b;

        pcal_a = calc_sigmoid(_data*theta_p) - _labels;

        pcal_b = _data(Range::all(), Range(0,1));

        cv::multiply(pcal_a, pcal_b, pcal_ab, 1);

        gradient.row(0) = ((float)1/m) * sum(pcal_ab)[0];


        pcal_b = _data(Range::all(), Range(1,n));

        //cout<<"for each training data entry"<<endl;
        for(int ii = 1;ii<gradient.rows;ii++)
        {
            pcal_b = _data(Range::all(), Range(ii,ii+1));

            cv::multiply(pcal_a, pcal_b, pcal_ab, 1);

            gradient.row(ii) = (1.0/m)*cv::sum(pcal_ab)[0] + (llambda/m) * theta_p.row(ii);
        }

        theta_p = theta_p - ( static_cast<double>(this->params.alpha)/m)*gradient;
        //cout<<"updated theta_p"<<endl;
    }

    return theta_p;
}


cv::Mat CvLR::compute_mini_batch_gradient(const cv::Mat& _data, const cv::Mat& _labels, const cv::Mat& _init_theta)
{
    // implements batch gradient descent

    int lambda_l = 0;
    double ccost;

    int m, n;

    int j = 0;
    int size_b = this->params.minibatchsize;

    // if(this->minibatchsize == 0)
    // {
    //     cv::error(Error::StsDivByZero, "compute_mini_batch_gradient: set CvLR::MINI_BATCH value to a non-zero number (and less than number of samples in a given class) ", "cv::ml::CvLR::compute_mini_batch_gradient", __FILE__, __LINE__);
    // }

    if(this->params.minibatchsize <= 0 || this->params.alpha == 0)
    {
        cv::error(Error::StsBadArg, "compute_mini_batch_gradient: check training parameters for the classifier","cv::ml::CvLR::compute_mini_batch_gradient", __FILE__, __LINE__);
    }

    if(this->params.num_iters <= 0)
    {
        cv::error(Error::StsBadArg,"compute_mini_batch_gradient: number of iterations cannot be zero or a negative number","cv::ml::CvLR::compute_mini_batch_gradient",__FILE__,__LINE__);
    }

    cv::Mat pcal_a;
    cv::Mat pcal_b;
    cv::Mat pcal_ab;
    cv::Mat gradient;
    cv::Mat theta_p = _init_theta.clone();
    cv::Mat data_d;
    cv::Mat labels_l;

    if(this->params.regularized > 0)
    {
        lambda_l = 1;
    }

    for(int i = 0;this->params.term_crit.max_iter;i++)
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
        n = data_d.cols;

        ccost = compute_cost(data_d, labels_l, theta_p);


        if( cvIsNaN( ccost ) == 1)
        {
            cv::error(Error::StsBadArg, "compute_mini_batch_gradient: check training parameters. Invalid training classifier","cv::ml::CvLR::compute_mini_batch_gradient", __FILE__, __LINE__);
        }

        ///////////////////////////////////////////////////
        // if(this->params.debug == 1 && i%(this->params.term_crit.max_iter/2)==0)
        // {
        //     cout<<"iter: "<<i<<endl;
        //     cout<<"cost: "<<ccost<<endl;
        //     cout<<"alpha"<<this->params.alpha<<endl;
        //     cout<<"num_iters"<<this->params.num_iters<<endl;
        //     cout<<"norm"<<this->params.norm<<endl;
        //     cout<<"debug"<<this->params.debug<<endl;
        //     cout<<"regularized"<<this->params.regularized<<endl;
        //     cout<<"train_method"<<this->params.train_method<<endl;
        //     cout<< "minibatchsize"<<this->params.minibatchsize<<endl;
        // }
        ///////////////////////////////////////////////////

        pcal_b = calc_sigmoid((data_d*theta_p) - labels_l);

        pcal_a = (static_cast<double>(1/m)) * data_d.t();

        gradient = pcal_a * pcal_b;

        pcal_a = calc_sigmoid(data_d*theta_p) - labels_l;

        pcal_b = data_d(Range::all(), Range(0,1));

        cv::multiply(pcal_a, pcal_b, pcal_ab, 1);

        gradient.row(0) = ((float)1/m) * sum(pcal_ab)[0];

        pcal_b = data_d(Range::all(), Range(1,n));

        for(int k = 1;k<gradient.rows;k++)
        {
            pcal_b = data_d(Range::all(), Range(k,k+1));

            cv::multiply(pcal_a, pcal_b, pcal_ab, 1);

            gradient.row(k) = (1.0/m)*cv::sum(pcal_ab)[0] + (lambda_l/m) * theta_p.row(k);
        }


        theta_p = theta_p - ( static_cast<double>(this->params.alpha)/m)*gradient;

        j+=this->params.minibatchsize;

        if(j+size_b>_data.rows)
        {
            // if parsed through all data variables
            break;
        }
    }

    return theta_p;
}


std::map<int, int> CvLR::get_label_map(const cv::Mat& _labels_i)
{
    // this function creates two maps to map user defined labels to program friendsly labels
    // two ways.

    cv::Mat labels;
    int ii = 0;

    _labels_i.convertTo(labels, CV_32S);

    for(int i = 0;i<labels.rows;i++)
    {
        this->forward_mapper[labels.at<int>(i)] += 1;
    }

    for(map<int,int>::iterator it = this->forward_mapper.begin(); it != this->forward_mapper.end(); ++it)
    {
        this->forward_mapper[it->first] = ii;
        ii += 1;
    }

    for(map<int,int>::iterator it = this->forward_mapper.begin(); it != this->forward_mapper.end(); ++it)
    {
        this->reverse_mapper[it->second] = it->first;
    }

    return this->forward_mapper;
}

bool CvLR::set_label_map(const cv::Mat& _labels_i)
{
    // this function creates two maps to map user defined labels to program friendsly labels
    // two ways.

    int ii = 0;
    cv::Mat labels;
    bool ok = false;

    this->labels_o = cv::Mat(0,1, CV_8U);
    this->labels_n = cv::Mat(0,1, CV_8U);

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
    ok = true;

    return ok;
}

cv::Mat CvLR::remap_labels(const Mat& _labels_i, std::map<int, int> lmap)
{
    cv::Mat labels;
    _labels_i.convertTo(labels, CV_32S);

    cv::Mat new_labels = cv::Mat::zeros(labels.rows, labels.cols, labels.type());

    CV_Assert( lmap.size() > 0 );

    for(int i =0;i<labels.rows;i++)
    {
        new_labels.at<int>(i,0) = lmap[labels.at<int>(i,0)];
    }

    return new_labels;
}


bool CvLR::set_default_params()
{
    // set default parameters for the Logisitic Regression classifier
    this->params.alpha = 1.0;
    this->params.term_crit.max_iter = 10000;
    this->params.norm = CvLR::REG_L2;
    ///////////////////////////////////////////////////
    // this->params.debug = 1;
    ///////////////////////////////////////////////////
    this->params.regularized = 1;
    this->params.train_method = CvLR::MINI_BATCH;
    this->params.minibatchsize = 10;

    return true;
}

void CvLR::clear()
{
    this->learnt_thetas.release();
    this->labels_o.release();
    this->labels_n.release();
}

void CvLR::read( CvFileStorage* fs, CvFileNode* node )
{
    CvMat *newData;
    CvMat *o_labels;
    CvMat *n_labels;


    this->params.alpha = cvReadRealByName(fs, node,"alpha", 1.0);
    this->params.num_iters = cvReadIntByName(fs, node,"iterations", 1000);
    this->params.norm = cvReadIntByName(fs, node,"norm", 1);
    // this->params.debug = cvReadIntByName(fs, node,"debug", 1);
    this->params.regularized = cvReadIntByName(fs, node,"regularized", 1);
    this->params.train_method = cvReadIntByName(fs, node,"train_method", 0);

    if(this->params.train_method == CvLR::MINI_BATCH)
    {
        this->params.minibatchsize = cvReadIntByName(fs, node,"mini_batch_size", 1);
    }

    newData = (CvMat*)cvReadByName( fs, node, "learnt_thetas" );
    o_labels = (CvMat*)cvReadByName( fs, node, "o_labels" );
    n_labels = (CvMat*)cvReadByName( fs, node, "n_labels" );

    this->learnt_thetas = cv::Mat(newData->rows, newData->cols, CV_32F, newData->data.db);
    this->labels_o = cv::Mat(o_labels->rows, o_labels->cols, CV_32S, o_labels->data.ptr);
    this->labels_n = cv::Mat(n_labels->rows, n_labels->cols, CV_32S, n_labels->data.ptr);

    for(int ii =0;ii<labels_o.rows;ii++)
    {
        this->forward_mapper[labels_o.at<int>(ii,0)] = labels_n.at<int>(ii,0);
        this->reverse_mapper[labels_n.at<int>(ii,0)] = labels_o.at<int>(ii,0);
    }

}

void CvLR::write( CvFileStorage* fs, const char* name ) const
{
    string desc = "Logisitic Regression Classifier";

    cvStartWriteStruct( fs, name, CV_NODE_MAP, CV_TYPE_NAME_ML_LR );

    cvWriteString( fs, "classifier", desc.c_str());
    cvWriteReal(fs,"alpha",this->params.alpha);
    cvWriteInt(fs,"iterations",this->params.num_iters);
    cvWriteInt(fs,"norm",this->params.norm);
    // cvWriteInt(fs,"debug",this->params.debug);
    cvWriteInt(fs,"regularized",this->params.regularized);
    cvWriteInt(fs,"train_method",this->params.train_method);

    if(this->params.train_method == CvLR::MINI_BATCH)
    {
        cvWriteInt(fs,"mini_batch_size",this->params.minibatchsize);
    }

    CvMat mat_learnt_thetas = this->learnt_thetas;
    CvMat o_labels = this->labels_o;
    CvMat n_labels = this->labels_n;

    cvWrite(fs, "learnt_thetas", &mat_learnt_thetas );
    cvWrite(fs, "n_labels", &n_labels);
    cvWrite(fs, "o_labels", &o_labels);

    cvEndWriteStruct(fs);
}


cv::Mat CvLR::get_learnt_mat()
{
    return this->learnt_thetas;
}

/* End of file. */
