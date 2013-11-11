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

LogisticRegressionParams::LogisticRegressionParams()
{
    term_crit = cv::TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 1000, 0.001);
    alpha = 0.001;
    num_iters = 1000;
    norm = LogisticRegression::REG_L2;
    regularized = 1;
    train_method = LogisticRegression::BATCH;
    mini_batch_size = 1;
}
LogisticRegressionParams::LogisticRegressionParams( double learning_rate, int iters, int train_algo = LogisticRegression::BATCH, int normlization = LogisticRegression::REG_L2, int reg = 1, int mb_size = 5)
{
    term_crit = cv::TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, iters, learning_rate);
    alpha = learning_rate;
    num_iters = iters;
    norm = normlization;
    regularized = reg;
    train_method = train_algo;
    mini_batch_size = mb_size;
}

LogisticRegression::LogisticRegression(const LogisticRegressionParams& pms)
{
    default_model_name = "my_lr";
    this->params = pms;
}

LogisticRegression::LogisticRegression(cv::InputArray data, cv::InputArray labels, const LogisticRegressionParams& pms)
{
    default_model_name = "my_lr";
    this->params = pms;
    train(data, labels);
}

LogisticRegression::~LogisticRegression()
{
    clear();
}

bool LogisticRegression::train(cv::InputArray data_ip, cv::InputArray labels_ip)
{
    clear();
    cv::Mat _data_i = data_ip.getMat();
    cv::Mat _labels_i = labels_ip.getMat();

    CV_Assert( !_labels_i.empty() && !_data_i.empty());

    // check the number of columns
    if(_labels_i.cols != 1)
    {
        CV_Error( CV_StsBadArg, "_labels_i should be a column matrix" );
    }

    // check data type.
    // data should be of floating type CV_32FC1

    if((_data_i.type() != CV_32FC1) || (_labels_i.type() != CV_32FC1))
    {
        CV_Error( CV_StsBadArg, "data and labels must be a floating point matrix" );
    }

    bool ok = false;

    cv::Mat labels;

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
        CV_Error( CV_StsBadArg, "data should have atleast 2 classes" );
    }

    if(_labels_i.rows != _data_i.rows)
    {
        CV_Error( CV_StsBadArg, "number of rows in data and labels should be the equal" );
    }


    cv::Mat thetas = cv::Mat::zeros(num_classes, data_t.cols, CV_32F);
    cv::Mat init_theta = cv::Mat::zeros(data_t.cols, 1, CV_32F);

    cv::Mat labels_l = remap_labels(_labels_i, this->forward_mapper);
    cv::Mat new_local_labels;

    int ii=0;
    cv::Mat new_theta;

    if(num_classes == 2)
    {
        labels_l.convertTo(labels, CV_32F);
        if(this->params.train_method == LogisticRegression::BATCH)
            new_theta = compute_batch_gradient(data_t, labels, init_theta);
        else
            new_theta = compute_mini_batch_gradient(data_t, labels, init_theta);
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
            new_local_labels.convertTo(labels, CV_32F);
            if(this->params.train_method == LogisticRegression::BATCH)
                new_theta = compute_batch_gradient(data_t, labels, init_theta);
            else
                new_theta = compute_mini_batch_gradient(data_t, labels, init_theta);
            hconcat(new_theta.t(), thetas.row(ii));
            ii += 1;
        }
    }

    this->learnt_thetas = thetas.clone();
    if( cvIsNaN( (double)cv::sum(this->learnt_thetas)[0] ) )
    {
        CV_Error( CV_StsBadArg, "check training parameters. Invalid training classifier" );
    }
    ok = true;
    return ok;
}


void LogisticRegression::predict( cv::InputArray _ip_data, cv::OutputArray _output_predicted_labels ) const
{
    /* returns a class of the predicted class
    class names can be 1,2,3,4, .... etc */
    cv::Mat thetas, data, pred_labs;
    data = _ip_data.getMat();

    // check if learnt_mats array is populated
    if(this->learnt_thetas.total()<=0)
    {
        CV_Error( CV_StsBadArg, "classifier should be trained first" );
    }
    if(data.type() != CV_32F)
    {
        CV_Error( CV_StsBadArg, "data must be of floating type" );
    }

    // add a column of ones
    cv::Mat data_t = cv::Mat::zeros(data.rows, data.cols+1, CV_32F);
    for (int i=0;i<data_t.cols;i++)
    {
        if(i==0)
        {
            vconcat(cv::Mat(data.rows, 1, data.type(), Scalar::all(1.0)), data_t.col(i));
            continue;
        }
        vconcat(data.col(i-1), data_t.col(i));
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
    cv::Mat pred_m = cv::Mat::zeros(data_t.rows, thetas.rows, data.type());

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
    pred_labs = remap_labels(labels_c, this->reverse_mapper);
    // convert pred_labs to integer type
    pred_labs.convertTo(pred_labs, CV_32S);
    pred_labs.copyTo(_output_predicted_labels);
}

cv::Mat LogisticRegression::calc_sigmoid(const Mat& data)
{
    cv::Mat dest;
    cv::exp(-data, dest);
    return 1.0/(1.0+dest);
}

double LogisticRegression::compute_cost(const cv::Mat& _data, const cv::Mat& _labels, const cv::Mat& _init_theta)
{

    int llambda = 0;
    int m;
    int n;
    double cost = 0;
    double rparameter = 0;
    cv::Mat gradient;
    cv::Mat theta_b;
    cv::Mat theta_c;
    cv::Mat d_a;
    cv::Mat d_b;

    m = _data.rows;
    n = _data.cols;

    gradient = cv::Mat::zeros( _init_theta.rows, _init_theta.cols, _init_theta.type());
    theta_b = _init_theta(Range(1, n), Range::all());
    cv::multiply(theta_b, theta_b, theta_c, 1);

    if(this->params.regularized > 0)
    {
        llambda = 1;
    }

    if(this->params.norm == LogisticRegression::REG_L1)
    {
        rparameter = (llambda/(2*m)) * cv::sum(theta_b)[0];
    }
    else
    {
        // assuming it to be L2 by default
        rparameter = (llambda/(2*m)) * cv::sum(theta_c)[0];
    }

    d_a = calc_sigmoid(_data* _init_theta);


    cv::log(d_a, d_a);
    cv::multiply(d_a, _labels, d_a);

    d_b = 1 - calc_sigmoid(_data * _init_theta);
    cv::log(d_b, d_b);
    cv::multiply(d_b, 1-_labels, d_b);

    cost = (-1.0/m) * (cv::sum(d_a)[0] + cv::sum(d_b)[0]);
    cost = cost + rparameter;

    return cost;
}

cv::Mat LogisticRegression::compute_batch_gradient(const cv::Mat& _data, const cv::Mat& _labels, const cv::Mat& _init_theta)
{
    // implements batch gradient descent
    if(this->params.alpha<=0)
    {
        CV_Error( CV_StsBadArg, "check training parameters for the classifier" );
    }

    if(this->params.num_iters <= 0)
    {
        CV_Error( CV_StsBadArg, "number of iterations cannot be zero or a negative number" );
    }

    int llambda = 0;
    double ccost;
    int m, n;
    cv::Mat pcal_a;
    cv::Mat pcal_b;
    cv::Mat pcal_ab;
    cv::Mat gradient;
    cv::Mat theta_p = _init_theta.clone();
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
            CV_Error( CV_StsBadArg, "check training parameters. Invalid training classifier" );
        }

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
    }
    return theta_p;
}

cv::Mat LogisticRegression::compute_mini_batch_gradient(const cv::Mat& _data, const cv::Mat& _labels, const cv::Mat& _init_theta)
{
    // implements batch gradient descent
    int lambda_l = 0;
    double ccost;
    int m, n;
    int j = 0;
    int size_b = this->params.mini_batch_size;

    if(this->params.mini_batch_size <= 0 || this->params.alpha == 0)
    {
        CV_Error( CV_StsBadArg, "check training parameters for the classifier" );
    }

    if(this->params.num_iters <= 0)
    {
        CV_Error( CV_StsBadArg, "number of iterations cannot be zero or a negative number" );
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

    for(int i = 0;this->params.term_crit.maxCount;i++)
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
            CV_Error( CV_StsBadArg, "check training parameters. Invalid training classifier" );
        }

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

        j+=this->params.mini_batch_size;

        if(j+size_b>_data.rows)
        {
            // if parsed through all data variables
            break;
        }
    }
    return theta_p;
}

bool LogisticRegression::set_label_map(const cv::Mat& _labels_i)
{
    // this function creates two maps to map user defined labels to program friendly labels two ways.
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

cv::Mat LogisticRegression::remap_labels(const Mat& _labels_i, const std::map<int, int>& lmap)
{
    cv::Mat labels;
    _labels_i.convertTo(labels, CV_32S);

    cv::Mat new_labels = cv::Mat::zeros(labels.rows, labels.cols, labels.type());

    CV_Assert( lmap.size() > 0 );

    for(int i =0;i<labels.rows;i++)
    {
        new_labels.at<int>(i,0) = lmap.find(labels.at<int>(i,0))->second;
    }
    return new_labels;
}

void LogisticRegression::clear()
{
    this->learnt_thetas.release();
    this->labels_o.release();
    this->labels_n.release();
}

void LogisticRegression::write(FileStorage& fs) const
{
    // check if open
    if(fs.isOpened() == 0)
    {
        CV_Error(CV_StsBadArg,"file can't open. Check file path");
    }
    string desc = "Logisitic Regression Classifier";
    fs<<"classifier"<<desc.c_str();
    fs<<"alpha"<<this->params.alpha;
    fs<<"iterations"<<this->params.num_iters;
    fs<<"norm"<<this->params.norm;
    fs<<"regularized"<<this->params.regularized;
    fs<<"train_method"<<this->params.train_method;
    if(this->params.train_method == LogisticRegression::MINI_BATCH)
    {
        fs<<"mini_batch_size"<<this->params.mini_batch_size;
    }
    fs<<"learnt_thetas"<<this->learnt_thetas;
    fs<<"n_labels"<<this->labels_n;
    fs<<"o_labels"<<this->labels_o;
}

void LogisticRegression::read(const FileNode& fn )
{
    // check if empty
    if(fn.empty())
    {
        CV_Error( CV_StsBadArg, "empty FileNode object" );
    }

    this->params.alpha = (double)fn["alpha"];
    this->params.num_iters = (int)fn["iterations"];
    this->params.norm = (int)fn["norm"];
    this->params.regularized = (int)fn["regularized"];
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

const cv::Mat LogisticRegression::get_learnt_thetas() const
{
    return this->learnt_thetas;
}
/* End of file. */
