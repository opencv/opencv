//*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
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
//
//M*/
#include "precomp.hpp"
#include <cstdio>
#include <iostream>

using namespace std;

const int progressBarSize = 50;
namespace cv
{

CalonderClassifier::CalonderClassifier()
{
    verbose = false;
    clear();
}

CalonderClassifier::~CalonderClassifier()
{}

CalonderClassifier::CalonderClassifier( const vector<vector<Point2f> >& points, const vector<Mat>& refimgs,
                                        const vector<vector<int> >& labels, int _numClasses,
                                        int _pathSize, int _numTrees, int _treeDepth,
                                        int _numViews, int _compressedDim, int _compressType, int _numQuantBits,
                                        const PatchGenerator &patchGenerator )
{
    verbose = false;
    train( points, refimgs, labels, _numClasses, _pathSize, _numTrees, _treeDepth, _numViews,
           _compressedDim, _compressType, _numQuantBits, patchGenerator );
}

int CalonderClassifier::getPatchSize() const
{ return patchSize; }

int CalonderClassifier::getNumTrees() const
{ return numTrees; }

int CalonderClassifier::getTreeDepth() const
{ return treeDepth; }

int CalonderClassifier::getNumViews() const
{ return numViews; }

int CalonderClassifier::getSignatureSize() const
{ return signatureSize; }

int CalonderClassifier::getCompressType() const
{ return compressType; }

int CalonderClassifier::getNumQuantBits() const
{ return numQuantBits; }

int CalonderClassifier::getOrigNumClasses() const
{ return origNumClasses; }

void CalonderClassifier::setVerbose( bool _verbose )
{
    verbose = _verbose;
}

void CalonderClassifier::clear()
{
    patchSize = numTrees = origNumClasses = signatureSize = treeDepth = numViews = numQuantBits = 0;
    compressType = COMPRESS_NONE;

    nodes.clear();
    posteriors.clear();
#if QUANTIZATION_AVAILABLE
    quantizedPosteriors.clear();
#endif
}

bool CalonderClassifier::empty() const
{
    return posteriors.empty() && quantizedPosteriors.empty();
}

void CalonderClassifier::prepare( int _patchSize, int _signatureSize, int _numTrees, int _treeDepth, int _numViews )
{
    clear();

    patchSize = _patchSize;
    signatureSize = _signatureSize;
    numTrees = _numTrees;
    treeDepth = _treeDepth;
    numViews = _numViews;

    numLeavesPerTree = 1 << treeDepth;      // 2^d
    numNodesPerTree = numLeavesPerTree - 1; // 2^d - 1

    nodes = vector<Node>( numTrees*numNodesPerTree );
    posteriors = vector<float>( numTrees*numLeavesPerTree*signatureSize, 0.f );
}

static int calcNumPoints( const vector<vector<Point2f> >& points )
{
    int count = 0;
    for( size_t i = 0; i < points.size(); i++ )
        count += points[i].size();
    return count;
}

void CalonderClassifier::train( const vector<vector<Point2f> >& points, const vector<Mat>& refimgs,
                                const vector<vector<int> >& labels, int _numClasses,
                                int _patchSize, int _numTrees, int _treeDepth, int _numViews,
                                int _compressedDim, int _compressType, int _numQuantBits,
                                const PatchGenerator &patchGenerator )
{
    if( points.empty() || refimgs.size() != points.size() )
        CV_Error( CV_StsBadSize, "points vector must be no empty and refimgs must have the same size as points" );
    if( _patchSize < 5 || _patchSize >= 256 )
        CV_Error( CV_StsBadArg, "patchSize must be in [5, 255]");
    if( _numTrees <= 0 || _treeDepth <= 0 )
        CV_Error( CV_StsBadArg, "numTrees, treeDepth, numViews must be positive");
    int numPoints = calcNumPoints( points );
    if( !labels.empty() && ( labels.size() != points.size() || _numClasses <=0 || _numClasses > numPoints ) )
        CV_Error( CV_StsBadArg, "labels has incorrect size or _numClasses is not in [1, numPoints]");
    _numViews = std::max( 1, _numViews );

    int _origNumClasses = labels.empty() ? numPoints : _numClasses;

    if( verbose )
    {
        cout << "Using train parameters:" << endl;
        cout << "   patchSize=" << _patchSize << endl;
        cout << "   numTrees=" << _numTrees << endl;
        cout << "   treeDepth=" << _treeDepth << endl;
        cout << "   numViews=" << _numViews << endl;
        cout << "   compressedDim=" << _compressedDim << endl;
        cout << "   compressType=" << _compressType << endl;
        cout << "   numQuantBits=" << _numQuantBits << endl;
        cout << endl
             << "   numPoints=" << numPoints << endl;
        cout << "   origNumClasses=" << _origNumClasses << endl;
    }

    prepare( _patchSize, _origNumClasses, _numTrees, _treeDepth, _numViews );

    origNumClasses = _origNumClasses;
    vector<int> leafSampleCounters = vector<int>( numTrees*numLeavesPerTree, 0 );
    // generate nodes
    RNG rng = theRNG();
    for( int i = 0; i < numTrees*numNodesPerTree; i++ )
    {
        uchar x1 = rng(_patchSize);
        uchar y1 = rng(_patchSize);
        uchar x2 = rng(_patchSize);
        uchar y2 = rng(_patchSize);
        nodes[i] = Node(x1, y1, x2, y2);
    }

    Size size( patchSize, patchSize );
    Mat patch;
    if( verbose ) cout << "START training..." << endl;
    for( size_t treeIdx = 0; treeIdx < (size_t)numTrees; treeIdx++ )
    {
        if( verbose ) cout << "< tree " << treeIdx << endl;
        int globalPointIdx = 0;
        int* treeLeafSampleCounters = &leafSampleCounters[treeIdx*numLeavesPerTree];
        float* treePosteriors = &posteriors[treeIdx*numLeavesPerTree*signatureSize];
        for( size_t imgIdx = 0; imgIdx < points.size(); imgIdx++ )
        {
            const Point2f* imgPoints = &points[imgIdx][0];
            const int* imgLabels = labels.empty() ? 0 : &labels[imgIdx][0];
            int last = -1, cur;
            for( size_t pointIdx = 0; pointIdx < points[imgIdx].size(); pointIdx++, globalPointIdx++ )
            {
                int classID = imgLabels==0 ? globalPointIdx : imgLabels[pointIdx];
                Point2f pt = imgPoints[pointIdx];
                const Mat& src = refimgs[imgIdx];

                if( verbose && (cur = (int)((float)globalPointIdx/numPoints*progressBarSize)) != last )
                {
                    last = cur;
                    cout << ".";
                    cout.flush();
                }

                CV_Assert( classID >= 0 && classID < signatureSize );
                for( int v = 0; v < numViews; v++ )
                {
                    patchGenerator( src, pt, patch, size, rng );
                    // add sample
                    int leafIdx = getLeafIdx( treeIdx, patch );
                    treeLeafSampleCounters[leafIdx]++;
                    treePosteriors[leafIdx*signatureSize + classID]++;
                }
            }
        }

        if( verbose ) cout << endl << ">" << endl;
    }

    _compressedDim = std::max( 0, std::min(signatureSize, _compressedDim) );
    _numQuantBits = std::max( 0, std::min((int)MAX_NUM_QUANT_BITS, _numQuantBits) );
    finalize( _compressedDim, _compressType, _numQuantBits, leafSampleCounters );

    if( verbose ) cout << "END training." << endl;
}

int CalonderClassifier::getLeafIdx( int treeIdx, const Mat& patch ) const
{
    const Node* treeNodes = &nodes[treeIdx*numNodesPerTree];
    int idx = 0;
    for( int d = 0; d < treeDepth-1; d++ )
    {
        int offset = treeNodes[idx](patch);
        idx = 2*idx + 1 + offset;
    }
    return idx;
}

void CalonderClassifier::finalize( int _compressedDim, int _compressType, int _numQuantBits,
                                   const vector<int>& leafSampleCounters )
{
    for( int ti = 0; ti < numTrees; ti++ )
    {
        const int* treeLeafSampleCounters = &leafSampleCounters[ti*numLeavesPerTree];
        float* treePosteriors = &posteriors[ti*numLeavesPerTree*signatureSize];
        // Normalize by number of patches to reach each leaf
        for( int li = 0; li < numLeavesPerTree; li++ )
        {
            int sampleCount = treeLeafSampleCounters[li];
            if( sampleCount != 0 )
            {
                float normalizer = 1.0f / sampleCount;
                int leafPosteriorIdx = li*signatureSize;
                for( int ci = 0; ci < signatureSize; ci++ )
                   treePosteriors[leafPosteriorIdx + ci] *= normalizer;
             }
        }
    }

    // apply compressive sensing
    if( _compressedDim > 0 && _compressedDim < signatureSize )
        compressLeaves( _compressedDim, _compressType );
    else
    {
        if( verbose )
            cout << endl << "[WARNING] NO compression to leaves applied, because _compressedDim=" << _compressedDim << endl;
    }

    // convert float-posteriors to uchar-posteriors (quantization step)
#if QUANTIZATION_AVAILABLE
    if( _numQuantBits > 0 )
        quantizePosteriors( _numQuantBits );
    else
    {
        if( verbose )
            cout << endl << "[WARNING] NO quantization to posteriors, because _numQuantBits=" << _numQuantBits << endl;
    }
#endif
}

Mat createCompressionMatrix( int rows, int cols, int distrType )
{
    Mat mtr( rows, cols, CV_32FC1 );
    assert( rows <= cols );

    RNG rng(23);

    if( distrType == CalonderClassifier::COMPRESS_DISTR_GAUSS )
    {
        float sigma = 1./rows;
        for( int y = 0; y < rows; y++ )
            for( int x = 0; x < cols; x++ )
                mtr.at<float>(y,x) = rng.gaussian( sigma );
    }
    else if( distrType == CalonderClassifier::COMPRESS_DISTR_BERNOULLI )
    {
        float par = (float)(1./sqrt((float)rows));
        for( int y = 0; y < rows; y++ )
            for( int x = 0; x < cols; x++ )
                mtr.at<float>(y,x) = rng(2)==0 ? par : -par;
    }
    else if( distrType == CalonderClassifier::COMPRESS_DISTR_DBFRIENDLY )
    {
        float par = (float)sqrt(3./rows);
        for( int y = 0; y < rows; y++ )
            for( int x = 0; x < cols; x++ )
            {
                int rng6 = rng(6);
                mtr.at<float>(y,x) = rng6==0 ? par : (rng6==1 ? -par : 0.f);
            }
    }
    else
        CV_Assert( 0 );

    return mtr;
}

void CalonderClassifier::compressLeaves( int _compressedDim, int _compressType )
{
    if( verbose )
        cout << endl << "[OK] compressing leaves with matrix " << _compressedDim << " x " << signatureSize << endl;

    Mat compressionMtrT = (createCompressionMatrix( _compressedDim, signatureSize, _compressType )).t();

    vector<float> comprPosteriors( numTrees*numLeavesPerTree*_compressedDim, 0);
    Mat( numTrees*numLeavesPerTree, _compressedDim, CV_32FC1, &comprPosteriors[0] ) =
            Mat( numTrees*numLeavesPerTree, signatureSize, CV_32FC1, &posteriors[0]) * compressionMtrT;

    posteriors.resize( comprPosteriors.size() );
    copy( comprPosteriors.begin(), comprPosteriors.end(), posteriors.begin() );

    signatureSize = _compressedDim;
    compressType = _compressType;
}

#if QUANTIZATION_AVAILABLE
static float percentile( const float* data, int n, float p )
{
   assert( n>0 );
   assert( p>=0 && p<=1 );

   vector<float> vec( data, data+n );
   sort(vec.begin(), vec.end());
   int ix = (int)(p*(n-1));
   return vec[ix];
}

void quantizeVector( const float* src, int dim, float fbounds[2], uchar ubounds[2], uchar* dst )
{
    assert( fbounds[0] < fbounds[1] );
    assert( ubounds[0] < ubounds[1] );

    float normFactor = 1.f/(fbounds[1] - fbounds[0]);
    for( int i = 0; i < dim; i++ )
    {
        float part = (src[i] - fbounds[0]) * normFactor;
        assert( 0 <= part && part <= 1 ) ;
        uchar val = ubounds[0] +  (uchar)( part*ubounds[1] );
        dst[i] = std::max( 0, (int)std::min(ubounds[1], val) );
    }
}

void CalonderClassifier::quantizePosteriors( int _numQuantBits, bool isClearFloatPosteriors )
{
    uchar ubounds[] = { 0, (uchar)((1<<_numQuantBits)-1) };
    float fbounds[] = { 0.f, 0.f };

    int totalLeavesCount = numTrees*numLeavesPerTree;
    for( int li = 0; li < totalLeavesCount; li++ ) // TODO for some random choosen leaves !
    {
        fbounds[0] += percentile( &posteriors[li*signatureSize], signatureSize, GET_LOWER_QUANT_PERC() );
        fbounds[1] += percentile( &posteriors[li*signatureSize], signatureSize, GET_UPPER_QUANT_PERC() );
    }
    fbounds[0] /= totalLeavesCount;
    fbounds[1] /= totalLeavesCount;

    quantizedPosteriors.resize( posteriors.size() );
    quantizeVector( &posteriors[0], posteriors.size(), fbounds, ubounds, &quantizedPosteriors[0] );

    if( isClearFloatPosteriors )
        clearFloatPosteriors();
}

void CalonderClassifier::clearFloatPosteriors()
{
    quantizedPosteriors.clear();
}

#endif

void CalonderClassifier::operator()( const Mat& img, Point2f pt, vector<float>& signature, float thresh ) const
{
    if( img.empty() || img.type() != CV_8UC1 )
        return;

    Mat patch;
    getRectSubPix(img, Size(patchSize,patchSize), pt, patch, img.type());
    (*this)( patch, signature, thresh );
}

void CalonderClassifier::operator()( const Mat& patch, vector<float>& signature, float thresh ) const
{
    if( posteriors.empty() || patch.empty() || patch.type() != CV_8UC1 || patch.cols < patchSize || patch.rows < patchSize )
        return;

    int treePostSize = numLeavesPerTree*signatureSize;

    signature.resize( signatureSize, 0.f );
    float* sig = &signature[0];
    for( int ti = 0; ti < numTrees; ti++ )
    {
        int leafIdx = getLeafIdx( ti, patch );
        const float* post = &posteriors[ti*treePostSize + leafIdx*signatureSize];
        for( int ci = 0; ci < signatureSize; ci++ )
            sig[ci] += post[ci];
    }
    float coef = 1.f/numTrees;
    for( int ci = 0; ci < signatureSize; ci++ )
    {
        sig[ci] *= coef;
        if( sig[ci] < thresh )
            sig[ci] = 0;
    }
}

#if QUANTIZATION_AVAILABLE
void CalonderClassifier::operator()( const Mat& img, Point2f pt, vector<uchar>& signature, uchar thresh ) const
{
    if( img.empty() || img.type() != CV_8UC1 )
        return;

    Mat patch;
    getRectSubPix(img, Size(patchSize,patchSize), pt, patch, img.type());
    (*this)(patch, signature, thresh );
}

void CalonderClassifier::operator()( const Mat& patch, vector<uchar>& signature, uchar thresh ) const
{
    if( quantizedPosteriors.empty() || patch.empty() || patch.type() != CV_8UC1 || patch.cols > patchSize || patch.rows > patchSize )
        return;

    int treePostSize = numLeavesPerTree*signatureSize;

    vector<float> sum( signatureSize, 0.f );
    for( int ti = 0; ti < numTrees; ti++ )
    {
        int leafIdx = getLeafIdx( ti, patch );
        const uchar* post = &quantizedPosteriors[ti*treePostSize + leafIdx*signatureSize];
        for( int ci = 0; ci < signatureSize; ci++ )
            sum[ci] += post[ci];
    }
    float coef = 1.f/numTrees;
    signature.resize( signatureSize );
    uchar* sig = &signature[0];
    for( int ci = 0; ci < signatureSize; ci++ )
    {
        sig[ci] = (uchar)(sum[ci]*coef);
        if( sig[ci] < thresh )
            sig[ci] = 0;
    }
}
#endif

void CalonderClassifier::read( const FileNode& fn )
{
    prepare( fn["patchSize"], fn["signatureSize"], fn["numTrees"], fn["treeDepth"], fn["numViews"] );
    origNumClasses = fn["origNumClasses"];
    compressType = fn["compressType"];
    int _numQuantBits = fn["numQuantBits"];

    for( int ti = 0; ti < numTrees; ti++ )
    {
        stringstream treeName;
        treeName << "tree" << ti;
        FileNode treeFN = fn["trees"][treeName.str()];

        Node* treeNodes = &nodes[ti*numNodesPerTree];
        FileNodeIterator nodesFNIter = treeFN["nodes"].begin();
        for( int ni = 0; ni < numNodesPerTree; ni++ )
        {
            Node* node = treeNodes + ni;
            nodesFNIter >> node->x1 >> node->y1 >> node->x2 >> node->y2;
        }

        FileNode posteriorsFN = treeFN["posteriors"];
        for( int li = 0; li < numLeavesPerTree; li++ )
        {
            stringstream leafName;
            leafName << "leaf" << li;
            float* post = &posteriors[ti*numLeavesPerTree*signatureSize + li*signatureSize];
            FileNodeIterator leafFNIter = posteriorsFN[leafName.str()].begin();
            for( int ci = 0; ci < signatureSize; ci++ )
                leafFNIter >> post[ci];
        }
    }
#if QUANTIZATION_AVAILABLE
    if( _numQuantBits )
        quantizePosteriors(_numQuantBits);
#endif
}

void CalonderClassifier::write( FileStorage& fs ) const
{
    if( !fs.isOpened() )
        return;
    fs << "patchSize" << patchSize;
    fs << "numTrees" << numTrees;
    fs << "treeDepth" << treeDepth;
    fs << "numViews" << numViews;
    fs << "origNumClasses" << origNumClasses;
    fs << "signatureSize" << signatureSize;
    fs << "compressType" << compressType;
    fs << "numQuantBits" << numQuantBits;

    fs << "trees" << "{";
    for( int ti = 0; ti < numTrees; ti++ )
    {
        stringstream treeName;
        treeName << "tree" << ti;
        fs << treeName.str() << "{";

        fs << "nodes" << "[:";
        const Node* treeNodes = &nodes[ti*numNodesPerTree];
        for( int ni = 0; ni < numNodesPerTree; ni++ )
        {
            const Node* node = treeNodes + ni;
            fs << node->x1 << node->y1 << node->x2 << node->y2;
        }
        fs << "]"; // nodes

        fs << "posteriors" << "{";
        for( int li = 0; li < numLeavesPerTree; li++ )
        {
            stringstream leafName;
            leafName << "leaf" << li;
            fs << leafName.str() << "[:";
            const float* post = &posteriors[ti*numLeavesPerTree*signatureSize + li*signatureSize];
            for( int ci = 0; ci < signatureSize; ci++ )
            {
                fs << post[ci];
            }
            fs << "]"; // leaf
        }
        fs << "}"; // posteriors
        fs << "}"; // tree
    }
    fs << "}"; // trees
}

}
