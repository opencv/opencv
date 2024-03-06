/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
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
//   * The name of Intel Corporation may not be used to endorse or promote products
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


namespace cv
{

BOWTrainer::BOWTrainer() : size(0)
{}

BOWTrainer::~BOWTrainer()
{}

void BOWTrainer::add( const Mat& _descriptors )
{
    CV_Assert( !_descriptors.empty() );
    if( !descriptors.empty() )
    {
        CV_Assert( descriptors[0].cols == _descriptors.cols );
        CV_Assert( descriptors[0].type() == _descriptors.type() );
        size += _descriptors.rows;
    }
    else
    {
        size = _descriptors.rows;
    }

    descriptors.push_back(_descriptors);
}

const std::vector<Mat>& BOWTrainer::getDescriptors() const
{
    return descriptors;
}

int BOWTrainer::descriptorsCount() const
{
    return descriptors.empty() ? 0 : size;
}

void BOWTrainer::clear()
{
    descriptors.clear();
}

BOWKMeansTrainer::BOWKMeansTrainer( int _clusterCount, const TermCriteria& _termcrit,
                                    int _attempts, int _flags ) :
    clusterCount(_clusterCount), termcrit(_termcrit), attempts(_attempts), flags(_flags)
{}

Mat BOWKMeansTrainer::cluster()
{
    CV_INSTRUMENT_REGION();

    CV_Assert( !descriptors.empty() );

    Mat mergedDescriptors( descriptorsCount(), descriptors[0].cols, descriptors[0].type() );
    for( size_t i = 0, start = 0; i < descriptors.size(); i++ )
    {
        Mat submut = mergedDescriptors.rowRange((int)start, (int)(start + descriptors[i].rows));
        descriptors[i].copyTo(submut);
        start += descriptors[i].rows;
    }
    return cluster( mergedDescriptors );
}

BOWKMeansTrainer::~BOWKMeansTrainer()
{}

Mat BOWKMeansTrainer::cluster( const Mat& _descriptors )
{
    CV_INSTRUMENT_REGION();

    Mat labels, vocabulary;
    kmeans( _descriptors, clusterCount, labels, termcrit, attempts, flags, vocabulary );
    return vocabulary;
}

DBOWTrainer::DBOWTrainer( int _clusterCountPerLevel, int _level, const NormTypes _scoringType,
                            const TermCriteria& _termcrit, int _attempts, int _flags ) :
    clusterCountPerLevel(_clusterCountPerLevel), level(_level), scoringType(_scoringType), termcrit(_termcrit), attempts(_attempts), flags(_flags)
{}

DBOWTrainer::~DBOWTrainer()
{}

Mat DBOWTrainer::cluster()
{
    CV_INSTRUMENT_REGION();

    CV_Assert( !descriptors.empty() );

    Mat mergedDescriptors;
    vconcat(descriptors, mergedDescriptors);
    mergedDescriptors.convertTo(mergedDescriptors, CV_32F);

    return cluster( mergedDescriptors );
}

Mat DBOWTrainer::cluster( const Mat& _descriptors )
{
    nodes.clear();
    int expected_nodes = (int)((pow((double)clusterCountPerLevel, (double)level + 1) - 1) / (clusterCountPerLevel - 1));
    nodes.reserve(expected_nodes);

    // Start clustering descriptors and building tree nodes
    nodes.push_back(Node(0));
    kmeansStep( _descriptors , 0, 1);

    createWords();
    setWeights();

    // returns an empty Mat
    return Mat();
}

void DBOWTrainer::createWords()
{
    // Create words (nodes that are leaves) from nodes
    words.resize(0);
    if (!nodes.empty())
    {
        words.reserve( (int)pow((double)clusterCountPerLevel, (double)level) );
        for (std::vector<Node>::iterator it = nodes.begin() + 1; it != nodes.end(); it++)
        {
            if (it->childs.empty())
            {
                it->wordIdx = (unsigned)words.size();
                words.push_back(&(*it));
            }
        }
    }
}

void DBOWTrainer::kmeansStep( const Mat& _descriptors, int parent, int current_level )
{
    if (_descriptors.empty()) return;

    Mat labels, vocabulary;
    std::vector<std::vector<unsigned> > groups;
    groups.reserve(clusterCountPerLevel);

    if (_descriptors.rows <= clusterCountPerLevel)
    {
        groups.resize(_descriptors.rows);
        for (int i = 0; i < _descriptors.rows; i++)
        {
            groups[i].push_back(i);
            vocabulary.push_back(_descriptors.row(i));
        }
    }
    else
    {
        groups.resize(clusterCountPerLevel);
        kmeans( _descriptors, clusterCountPerLevel, labels, termcrit, attempts, flags, vocabulary );
        for (int i = 0; i < labels.rows; i++)
            groups[labels.at<int>(i)].push_back(i);
    }

    vocabulary.convertTo(vocabulary, descriptors[0].type());
    for (int i = 0; i < vocabulary.rows; i++)
    {
        unsigned idx = (unsigned)nodes.size();
        nodes.push_back(Node(idx, parent, vocabulary.row(i)));
        nodes[parent].childs.push_back(idx);
    }


    if (current_level < level)
    {
        std::vector<unsigned> childs = nodes[parent].childs;
        for (int i = 0; i < vocabulary.rows; i++)
        {
            unsigned child = childs[i];
            std::vector<cv::Mat> childDescriptors;
            childDescriptors.reserve(groups[i].size());

            for (int j = 0; j < (int)groups[i].size(); j++)
                childDescriptors.push_back(_descriptors.row(groups[i][j]));

            if (childDescriptors.size() > 1)
            {
                cv::Mat mergedDescriptors;
                vconcat(childDescriptors, mergedDescriptors);
                kmeansStep(mergedDescriptors, child, current_level + 1);
            }
        }
    }
}

void DBOWTrainer::setWeights()
{
    CV_Assert( !words.empty() );
    const int nDescriptors = (int)descriptors.size();
    const int nWords = (int)words.size();

    std::vector<unsigned> cnt(nWords, 0);
    std::vector<bool> visited(nWords, false);

    for (int i = 0; i < nDescriptors; i++)
    {
        fill(visited.begin(), visited.end(), false);
        for (int j = 0; j < descriptors[i].rows; j++)
        {
            cv::Mat descriptor = descriptors[i].row(j);
            unsigned wordIdx;
            double weight;
            transform(descriptor, wordIdx, weight);

            if (!visited[wordIdx])
            {
                cnt[wordIdx]++;
                visited[wordIdx] = true;
            }
        }
    }

    for (int i = 0; i < nWords; i++)
        if (cnt[i] > 0)
            words[i]->weight = log((double)nDescriptors / (double)cnt[i]);
}

void DBOWTrainer::transform( const Mat& _descriptor, unsigned& wordIdx, double& weight )
{
    CV_Assert( _descriptor.type() == nodes[0].descriptor.type() );

    std::vector<unsigned> childs;
    std::vector<unsigned>::const_iterator child;
    unsigned returnIdx = 0;

    do
    {
        childs = nodes[returnIdx].childs;
        returnIdx = childs[0];

        int minDistance = (int)norm(_descriptor, nodes[returnIdx].descriptor, scoringType);

        for (child = childs.begin() + 1; child != childs.end(); child++)
        {
            // Compute distance between two descriptors with given scoring type
            int distance = (int)norm(_descriptor, nodes[*child].descriptor, scoringType);

            // Update minDistance and returnIdx if this descriptor has shorter distance
            if (distance < minDistance)
            {
                minDistance = distance;
                returnIdx = *child;
            }
        }

    } while (!nodes[returnIdx].childs.empty());

    wordIdx = nodes[returnIdx].wordIdx;
    weight = nodes[returnIdx].weight;
}

DBOWTrainer::BOWVector::BOWVector()
{}

DBOWTrainer::BOWVector::~BOWVector()
{}

void DBOWTrainer::BOWVector::addWeight( int id, double weight )
{
    BOWVector::iterator it = this->lower_bound(id);

    if (it != this->end() && !(this->key_comp()(id, it->first)))
        it->second += weight;
    else
        this->insert(it, BOWVector::value_type(id, weight));
}

void DBOWTrainer::BOWVector::addIfNotExist( int id, double weight )
{
    BOWVector::iterator it = this->lower_bound(id);

    if (it == this->end() || (this->key_comp()(id, it->first)))
        this->insert(it, BOWVector::value_type(id, weight));
}

void DBOWTrainer::BOWVector::normalize( NormTypes normType )
{
    double norm = 0.0;
    BOWVector::iterator it;

    if (normType == 2)
    {
        // L1 norm
        for (it = begin(); it != end(); ++it)
            norm += fabs(it->second);
    }
    else
    {
        // L2 norm
        for (it = begin(); it != end(); ++it)
            norm += it->second * it->second;
        norm = sqrt(norm);
    }

    if (norm > 0.0)
    {
        for(it = begin(); it != end(); ++it)
            it->second /= norm;
    }
}

void DBOWTrainer::transform( const Mat& _descriptor, BOWVector& bowVector )
{
    bowVector.clear();

    if (words.empty()) return;

    for (int i = 0; i < _descriptor.rows; i++)
    {
        unsigned wordIdx;
        double weight;
        transform(_descriptor.row(i), wordIdx, weight);
        if (weight > 0) bowVector.addWeight(wordIdx, weight);
    }

    bowVector.normalize(scoringType);
}

double DBOWTrainer::score( BOWVector& bowVector1, BOWVector& bowVector2 )
{
    std::vector<double> weights1, weights2;
    BOWVector::const_iterator it1, it2;
    it1 = bowVector1.begin();
    it2 = bowVector2.begin();

    // Find common elements between bowVector1 and bowVector2
    while (it1 != bowVector1.end() && it2 != bowVector2.end())
    {
        if (it1->first < it2->first) ++it1;
        else if (it2->first < it1->first) ++it2;
        else
        {
            weights1.push_back(it1->second);
            weights2.push_back(it2->second);
            ++it1; ++it2;
        }
    }

    double score = 0.0;

    switch (scoringType)
    {
        // Please refer to *Scalable Recognition with a Vocabulary Tree*, CVPR 2006.
        case NORM_L1:
            for (int i = 0; i < (int)weights1.size(); i++)
                score += fabs(weights1[i] - weights2[i]) - fabs(weights1[i]) - fabs(weights2[i]);
            score = -score/2.0;
            break;

        case NORM_L2:
        default:
            for (int i = 0; i < (int)weights1.size(); i++)
                score += weights1[i] * weights2[i];
            score = (score >= 1 ? 1.0 : 1.0 - sqrt(1.0 - score));
            break;
    }
    return score;
}

void DBOWTrainer::save( const std::string &fn )
{
    FileStorage fs(fn.c_str(), FileStorage::WRITE);
    if (!fs.isOpened()) throw std::string("Fail to open file ") + fn;

    int desLength = descriptors[0].cols;
    std::vector<unsigned> parents, childs;
    std::vector<unsigned>::const_iterator child;
    std::vector<Node*>::const_iterator word;

    fs << "vocabulary" << "{";
    fs << "clusterCountPerLevel" << clusterCountPerLevel;
    fs << "level" << level;
    fs << "scoringType" << scoringType;
    fs << "desLength" << desLength;

    parents.push_back(0);

    // Write node data
    fs << "nodes" << "[";
    while (!parents.empty())
    {
        const Node& parent = nodes[parents.back()];
        parents.pop_back();
        childs = parent.childs;

        for (child = childs.begin(); child != childs.end(); child++)
        {
            const Node& node = nodes[*child];

            // Convert descriptor to string for faster reading
            std::stringstream ss;
            const uchar *p = node.descriptor.ptr<uchar>();
            for (int i = 0; i < desLength; i++, p++)
                ss << (int)*p << " ";

            fs << "{:";
            fs << "nodeIdx" << (int)node.idx;
            fs << "parentIdx" << (int)parent.idx;
            fs << "weight" << (double)node.weight;
            fs << "descriptor" << ss.str();
            fs << "}";

            if (!node.childs.empty())
                parents.push_back(*child);
        }
    }
    fs << "]";

    // Write word data
    fs << "words" << "[";
    for (word = words.begin(); word != words.end(); word++)
    {
        unsigned idx = (int)(word - words.begin());
        fs << "{:";
        fs << "wordIdx" << (int)idx;
        fs << "nodeIdx" << (int)(*word)->idx;
        fs << "}";
    }

    fs << "]";
    fs << "}";
    fs.release();
}

void DBOWTrainer::load( const std::string &fn )
{
    FileStorage fs(fn.c_str(), FileStorage::READ);
    if (!fs.isOpened()) throw std::string("Fail to open file ") + fn;

    nodes.clear();
    words.clear();

    int desLength;
    FileNode fsVoc, fsNodes, fsWords;
    fsVoc = fs["vocabulary"];
    fsNodes = fsVoc["nodes"];
    fsWords = fsVoc["words"];

    fsVoc["clusterCountPerLevel"] >> clusterCountPerLevel;
    fsVoc["level"] >> level;
    fsVoc["scoringType"] >> scoringType;
    fsVoc["desLength"] >> desLength;

    // Read nodes
    nodes.resize(fsNodes.size() + 1);
    nodes[0].idx = 0;

    for (unsigned i = 0; i < fsNodes.size(); ++i)
    {
        unsigned idx = (int)fsNodes[i]["nodeIdx"];
        unsigned parent = (int)fsNodes[i]["parentIdx"];
        double weight = (double)fsNodes[i]["weight"];
        std::string desStr = fsNodes[i]["descriptor"];

        nodes[idx].idx = idx;
        nodes[idx].parent = parent;
        nodes[idx].weight = weight;
        nodes[parent].childs.push_back(idx);

        // Parse string to descriptor for faster reading
        nodes[idx].descriptor.create(1, desLength, CV_8U);
        uchar *p = nodes[idx].descriptor.ptr<uchar>();
        std::stringstream ss(desStr);
        for (int j = 0; j < desLength; j++, p++)
        {
            int n;
            ss >> n;

            if (!ss.fail()) *p = (uchar)n;
        }
    }

    // Read words
    words.resize(fsWords.size());

    for(unsigned int i = 0; i < fsWords.size(); ++i)
    {
        unsigned idx = (int)fsWords[i]["nodeIdx"];
        unsigned wordIdx = (int)fsWords[i]["wordIdx"];

        nodes[idx].wordIdx = wordIdx;
        words[wordIdx] = &nodes[idx];
    }

    fs.release();
}

BOWImgDescriptorExtractor::BOWImgDescriptorExtractor( const Ptr<DescriptorExtractor>& _dextractor,
                                                      const Ptr<DescriptorMatcher>& _dmatcher ) :
    dextractor(_dextractor), dmatcher(_dmatcher)
{}

BOWImgDescriptorExtractor::BOWImgDescriptorExtractor( const Ptr<DescriptorMatcher>& _dmatcher ) :
    dmatcher(_dmatcher)
{}

BOWImgDescriptorExtractor::~BOWImgDescriptorExtractor()
{}

void BOWImgDescriptorExtractor::setVocabulary( const Mat& _vocabulary )
{
    dmatcher->clear();
    vocabulary = _vocabulary;
    dmatcher->add( std::vector<Mat>(1, vocabulary) );
}

const Mat& BOWImgDescriptorExtractor::getVocabulary() const
{
    return vocabulary;
}

void BOWImgDescriptorExtractor::compute( InputArray image, std::vector<KeyPoint>& keypoints, OutputArray imgDescriptor,
                                         std::vector<std::vector<int> >* pointIdxsOfClusters, Mat* descriptors )
{
    CV_INSTRUMENT_REGION();

    imgDescriptor.release();

    if( keypoints.empty() )
        return;

    // Compute descriptors for the image.
    Mat _descriptors;
    dextractor->compute( image, keypoints, _descriptors );

    compute( _descriptors, imgDescriptor, pointIdxsOfClusters );

    // Add the descriptors of image keypoints
    if (descriptors) {
        *descriptors = _descriptors.clone();
    }
}

int BOWImgDescriptorExtractor::descriptorSize() const
{
    return vocabulary.empty() ? 0 : vocabulary.rows;
}

int BOWImgDescriptorExtractor::descriptorType() const
{
    return CV_32FC1;
}

void BOWImgDescriptorExtractor::compute( InputArray keypointDescriptors, OutputArray _imgDescriptor, std::vector<std::vector<int> >* pointIdxsOfClusters )
{
    CV_INSTRUMENT_REGION();

    CV_Assert( !vocabulary.empty() );
    CV_Assert(!keypointDescriptors.empty());

    int clusterCount = descriptorSize(); // = vocabulary.rows

    // Match keypoint descriptors to cluster center (to vocabulary)
    std::vector<DMatch> matches;
    dmatcher->match( keypointDescriptors, matches );

    // Compute image descriptor
    if( pointIdxsOfClusters )
    {
        pointIdxsOfClusters->clear();
        pointIdxsOfClusters->resize(clusterCount);
    }

    _imgDescriptor.create(1, clusterCount, descriptorType());
    _imgDescriptor.setTo(Scalar::all(0));

    Mat imgDescriptor = _imgDescriptor.getMat();

    float *dptr = imgDescriptor.ptr<float>();
    for( size_t i = 0; i < matches.size(); i++ )
    {
        int queryIdx = matches[i].queryIdx;
        int trainIdx = matches[i].trainIdx; // cluster index
        CV_Assert( queryIdx == (int)i );

        dptr[trainIdx] = dptr[trainIdx] + 1.f;
        if( pointIdxsOfClusters )
            (*pointIdxsOfClusters)[trainIdx].push_back( queryIdx );
    }

    // Normalize image descriptor.
    imgDescriptor /= keypointDescriptors.size().height;
}

}
