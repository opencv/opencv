// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "test_precomp.hpp"

namespace opencv_test {

void defaultDistribs( Mat& means, vector<Mat>& covs, int type)
{
    float mp0[] = {0.0f, 0.0f}, cp0[] = {0.67f, 0.0f, 0.0f, 0.67f};
    float mp1[] = {5.0f, 0.0f}, cp1[] = {1.0f, 0.0f, 0.0f, 1.0f};
    float mp2[] = {1.0f, 5.0f}, cp2[] = {1.0f, 0.0f, 0.0f, 1.0f};
    means.create(3, 2, type);
    Mat m0( 1, 2, CV_32FC1, mp0 ), c0( 2, 2, CV_32FC1, cp0 );
    Mat m1( 1, 2, CV_32FC1, mp1 ), c1( 2, 2, CV_32FC1, cp1 );
    Mat m2( 1, 2, CV_32FC1, mp2 ), c2( 2, 2, CV_32FC1, cp2 );
    means.resize(3), covs.resize(3);

    Mat mr0 = means.row(0);
    m0.convertTo(mr0, type);
    c0.convertTo(covs[0], type);

    Mat mr1 = means.row(1);
    m1.convertTo(mr1, type);
    c1.convertTo(covs[1], type);

    Mat mr2 = means.row(2);
    m2.convertTo(mr2, type);
    c2.convertTo(covs[2], type);
}

// generate points sets by normal distributions
void generateData( Mat& data, Mat& labels, const vector<int>& sizes, const Mat& _means, const vector<Mat>& covs, int dataType, int labelType )
{
    vector<int>::const_iterator sit = sizes.begin();
    int total = 0;
    for( ; sit != sizes.end(); ++sit )
        total += *sit;
    CV_Assert( _means.rows == (int)sizes.size() && covs.size() == sizes.size() );
    CV_Assert( !data.empty() && data.rows == total );
    CV_Assert( data.type() == dataType );

    labels.create( data.rows, 1, labelType );

    randn( data, Scalar::all(-1.0), Scalar::all(1.0) );
    vector<Mat> means(sizes.size());
    for(int i = 0; i < _means.rows; i++)
        means[i] = _means.row(i);
    vector<Mat>::const_iterator mit = means.begin(), cit = covs.begin();
    int bi, ei = 0;
    sit = sizes.begin();
    for( int p = 0, l = 0; sit != sizes.end(); ++sit, ++mit, ++cit, l++ )
    {
        bi = ei;
        ei = bi + *sit;
        CV_Assert( mit->rows == 1 && mit->cols == data.cols );
        CV_Assert( cit->rows == data.cols && cit->cols == data.cols );
        for( int i = bi; i < ei; i++, p++ )
        {
            Mat r = data.row(i);
            r =  r * (*cit) + *mit;
            if( labelType == CV_32FC1 )
                labels.at<float>(p, 0) = (float)l;
            else if( labelType == CV_32SC1 )
                labels.at<int>(p, 0) = l;
            else
            {
                CV_DbgAssert(0);
            }
        }
    }
}

int maxIdx( const vector<int>& count )
{
    int idx = -1;
    int maxVal = -1;
    vector<int>::const_iterator it = count.begin();
    for( int i = 0; it != count.end(); ++it, i++ )
    {
        if( *it > maxVal)
        {
            maxVal = *it;
            idx = i;
        }
    }
    CV_Assert( idx >= 0);
    return idx;
}

bool getLabelsMap( const Mat& labels, const vector<int>& sizes, vector<int>& labelsMap, bool checkClusterUniq)
{
    size_t total = 0, nclusters = sizes.size();
    for(size_t i = 0; i < sizes.size(); i++)
        total += sizes[i];

    CV_Assert( !labels.empty() );
    CV_Assert( labels.total() == total && (labels.cols == 1 || labels.rows == 1));
    CV_Assert( labels.type() == CV_32SC1 || labels.type() == CV_32FC1 );

    bool isFlt = labels.type() == CV_32FC1;

    labelsMap.resize(nclusters);

    vector<bool> buzy(nclusters, false);
    int startIndex = 0;
    for( size_t clusterIndex = 0; clusterIndex < sizes.size(); clusterIndex++ )
    {
        vector<int> count( nclusters, 0 );
        for( int i = startIndex; i < startIndex + sizes[clusterIndex]; i++)
        {
            int lbl = isFlt ? (int)labels.at<float>(i) : labels.at<int>(i);
            CV_Assert(lbl < (int)nclusters);
            count[lbl]++;
            CV_Assert(count[lbl] < (int)total);
        }
        startIndex += sizes[clusterIndex];

        int cls = maxIdx( count );
        CV_Assert( !checkClusterUniq || !buzy[cls] );

        labelsMap[clusterIndex] = cls;

        buzy[cls] = true;
    }

    if(checkClusterUniq)
    {
        for(size_t i = 0; i < buzy.size(); i++)
            if(!buzy[i])
                return false;
    }

    return true;
}

bool calcErr( const Mat& labels, const Mat& origLabels, const vector<int>& sizes, float& err, bool labelsEquivalent, bool checkClusterUniq)
{
    err = 0;
    CV_Assert( !labels.empty() && !origLabels.empty() );
    CV_Assert( labels.rows == 1 || labels.cols == 1 );
    CV_Assert( origLabels.rows == 1 || origLabels.cols == 1 );
    CV_Assert( labels.total() == origLabels.total() );
    CV_Assert( labels.type() == CV_32SC1 || labels.type() == CV_32FC1 );
    CV_Assert( origLabels.type() == labels.type() );

    vector<int> labelsMap;
    bool isFlt = labels.type() == CV_32FC1;
    if( !labelsEquivalent )
    {
        if( !getLabelsMap( labels, sizes, labelsMap, checkClusterUniq ) )
            return false;

        for( int i = 0; i < labels.rows; i++ )
            if( isFlt )
                err += labels.at<float>(i) != labelsMap[(int)origLabels.at<float>(i)] ? 1.f : 0.f;
            else
                err += labels.at<int>(i) != labelsMap[origLabels.at<int>(i)] ? 1.f : 0.f;
    }
    else
    {
        for( int i = 0; i < labels.rows; i++ )
            if( isFlt )
                err += labels.at<float>(i) != origLabels.at<float>(i) ? 1.f : 0.f;
            else
                err += labels.at<int>(i) != origLabels.at<int>(i) ? 1.f : 0.f;
    }
    err /= (float)labels.rows;
    return true;
}

bool calculateError( const Mat& _p_labels, const Mat& _o_labels, float& error)
{
    error = 0.0f;
    float accuracy = 0.0f;
    Mat _p_labels_temp;
    Mat _o_labels_temp;
    _p_labels.convertTo(_p_labels_temp, CV_32S);
    _o_labels.convertTo(_o_labels_temp, CV_32S);

    CV_Assert(_p_labels_temp.total() == _o_labels_temp.total());
    CV_Assert(_p_labels_temp.rows == _o_labels_temp.rows);

    accuracy = (float)countNonZero(_p_labels_temp == _o_labels_temp)/_p_labels_temp.rows;
    error = 1 - accuracy;
    return true;
}

} // namespace
