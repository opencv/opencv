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
#include <ctype.h>
#include <algorithm>
#include <iterator>

#include <opencv2/core/utils/logger.hpp>

namespace cv { namespace ml {

static const float MISSED_VAL = TrainData::missingValue();
static const int VAR_MISSED = VAR_ORDERED;

TrainData::~TrainData() {}

Mat TrainData::getSubVector(const Mat& vec, const Mat& idx)
{
    if (!(vec.cols == 1 || vec.rows == 1))
        CV_LOG_WARNING(NULL, "'getSubVector(const Mat& vec, const Mat& idx)' call with non-1D input is deprecated. It is not designed to work with 2D matrixes (especially with 'cv::ml::COL_SAMPLE' layout).");
    return getSubMatrix(vec, idx, vec.rows == 1 ? cv::ml::COL_SAMPLE : cv::ml::ROW_SAMPLE);
}

template<typename T>
Mat getSubMatrixImpl(const Mat& m, const Mat& idx, int layout)
{
    int nidx = idx.checkVector(1, CV_32S);
    int dims = m.cols, nsamples = m.rows;

    Mat subm;
    if (layout == COL_SAMPLE)
    {
        std::swap(dims, nsamples);
        subm.create(dims, nidx, m.type());
    }
    else
    {
        subm.create(nidx, dims, m.type());
    }

    for (int i = 0; i < nidx; i++)
    {
        int k = idx.at<int>(i); CV_CheckGE(k, 0, "Bad idx"); CV_CheckLT(k, nsamples, "Bad idx or layout");
        if (dims == 1)
        {
            subm.at<T>(i) = m.at<T>(k);  // at() has "transparent" access for 1D col-based / row-based vectors.
        }
        else if (layout == COL_SAMPLE)
        {
            for (int j = 0; j < dims; j++)
                subm.at<T>(j, i) = m.at<T>(j, k);
        }
        else
        {
            for (int j = 0; j < dims; j++)
                subm.at<T>(i, j) = m.at<T>(k, j);
        }
    }
    return subm;
}

Mat TrainData::getSubMatrix(const Mat& m, const Mat& idx, int layout)
{
    if (idx.empty())
        return m;
    int type = m.type();
    CV_CheckType(type, type == CV_32S || type == CV_32F || type == CV_64F, "");
    if (type == CV_32S || type == CV_32F)  // 32-bit
        return getSubMatrixImpl<int>(m, idx, layout);
    if (type == CV_64F)  // 64-bit
        return getSubMatrixImpl<double>(m, idx, layout);
    CV_Error(Error::StsInternal, "");
}


class TrainDataImpl CV_FINAL : public TrainData
{
public:
    typedef std::map<String, int> MapType;

    TrainDataImpl()
    {
        file = 0;
        clear();
    }

    virtual ~TrainDataImpl() { closeFile(); }

    int getLayout() const CV_OVERRIDE { return layout; }
    int getNSamples() const CV_OVERRIDE
    {
        return !sampleIdx.empty() ? (int)sampleIdx.total() :
               layout == ROW_SAMPLE ? samples.rows : samples.cols;
    }
    int getNTrainSamples() const CV_OVERRIDE
    {
        return !trainSampleIdx.empty() ? (int)trainSampleIdx.total() : getNSamples();
    }
    int getNTestSamples() const CV_OVERRIDE
    {
        return !testSampleIdx.empty() ? (int)testSampleIdx.total() : 0;
    }
    int getNVars() const CV_OVERRIDE
    {
        return !varIdx.empty() ? (int)varIdx.total() : getNAllVars();
    }
    int getNAllVars() const CV_OVERRIDE
    {
        return layout == ROW_SAMPLE ? samples.cols : samples.rows;
    }

    Mat getTestSamples() const CV_OVERRIDE
    {
        Mat idx = getTestSampleIdx();
        return idx.empty() ? Mat() : getSubMatrix(samples, idx, getLayout());
    }

    Mat getSamples() const CV_OVERRIDE { return samples; }
    Mat getResponses() const CV_OVERRIDE { return responses; }
    Mat getMissing() const CV_OVERRIDE { return missing; }
    Mat getVarIdx() const CV_OVERRIDE { return varIdx; }
    Mat getVarType() const CV_OVERRIDE { return varType; }
    int getResponseType() const CV_OVERRIDE
    {
        return classLabels.empty() ? VAR_ORDERED : VAR_CATEGORICAL;
    }
    Mat getTrainSampleIdx() const CV_OVERRIDE { return !trainSampleIdx.empty() ? trainSampleIdx : sampleIdx; }
    Mat getTestSampleIdx() const CV_OVERRIDE { return testSampleIdx; }
    Mat getSampleWeights() const CV_OVERRIDE
    {
        return sampleWeights;
    }
    Mat getTrainSampleWeights() const CV_OVERRIDE
    {
        return getSubVector(sampleWeights, getTrainSampleIdx());  // 1D-vector
    }
    Mat getTestSampleWeights() const CV_OVERRIDE
    {
        Mat idx = getTestSampleIdx();
        return idx.empty() ? Mat() : getSubVector(sampleWeights, idx);  // 1D-vector
    }
    Mat getTrainResponses() const CV_OVERRIDE
    {
        return getSubMatrix(responses, getTrainSampleIdx(), cv::ml::ROW_SAMPLE);  // col-based responses are transposed in setData()
    }
    Mat getTrainNormCatResponses() const CV_OVERRIDE
    {
        return getSubMatrix(normCatResponses, getTrainSampleIdx(), cv::ml::ROW_SAMPLE);  // like 'responses'
    }
    Mat getTestResponses() const CV_OVERRIDE
    {
        Mat idx = getTestSampleIdx();
        return idx.empty() ? Mat() : getSubMatrix(responses, idx, cv::ml::ROW_SAMPLE);  // col-based responses are transposed in setData()
    }
    Mat getTestNormCatResponses() const CV_OVERRIDE
    {
        Mat idx = getTestSampleIdx();
        return idx.empty() ? Mat() : getSubMatrix(normCatResponses, idx, cv::ml::ROW_SAMPLE);  // like 'responses'
    }
    Mat getNormCatResponses() const CV_OVERRIDE { return normCatResponses; }
    Mat getClassLabels() const CV_OVERRIDE { return classLabels; }
    Mat getClassCounters() const { return classCounters; }
    int getCatCount(int vi) const CV_OVERRIDE
    {
        int n = (int)catOfs.total();
        CV_Assert( 0 <= vi && vi < n );
        Vec2i ofs = catOfs.at<Vec2i>(vi);
        return ofs[1] - ofs[0];
    }

    Mat getCatOfs() const CV_OVERRIDE { return catOfs; }
    Mat getCatMap() const CV_OVERRIDE { return catMap; }

    Mat getDefaultSubstValues() const CV_OVERRIDE { return missingSubst; }

    void closeFile() { if(file) fclose(file); file=0; }
    void clear()
    {
        closeFile();
        samples.release();
        missing.release();
        varType.release();
        varSymbolFlags.release();
        responses.release();
        sampleIdx.release();
        trainSampleIdx.release();
        testSampleIdx.release();
        normCatResponses.release();
        classLabels.release();
        classCounters.release();
        catMap.release();
        catOfs.release();
        nameMap = MapType();
        layout = ROW_SAMPLE;
    }

    typedef std::map<int, int> CatMapHash;

    void setData(InputArray _samples, int _layout, InputArray _responses,
                 InputArray _varIdx, InputArray _sampleIdx, InputArray _sampleWeights,
                 InputArray _varType, InputArray _missing)
    {
        clear();

        CV_Assert(_layout == ROW_SAMPLE || _layout == COL_SAMPLE );
        samples = _samples.getMat();
        layout = _layout;
        responses = _responses.getMat();
        varIdx = _varIdx.getMat();
        sampleIdx = _sampleIdx.getMat();
        sampleWeights = _sampleWeights.getMat();
        varType = _varType.getMat();
        missing = _missing.getMat();

        int nsamples = layout == ROW_SAMPLE ? samples.rows : samples.cols;
        int ninputvars = layout == ROW_SAMPLE ? samples.cols : samples.rows;
        int i, noutputvars = 0;

        CV_Assert( samples.type() == CV_32F || samples.type() == CV_32S );

        if( !sampleIdx.empty() )
        {
            CV_Assert( (sampleIdx.checkVector(1, CV_32S, true) > 0 &&
                       checkRange(sampleIdx, true, 0, 0, nsamples)) ||
                       sampleIdx.checkVector(1, CV_8U, true) == nsamples );
            if( sampleIdx.type() == CV_8U )
                sampleIdx = convertMaskToIdx(sampleIdx);
        }

        if( !sampleWeights.empty() )
        {
            CV_Assert( sampleWeights.checkVector(1, CV_32F, true) == nsamples );
        }
        else
        {
            sampleWeights = Mat::ones(nsamples, 1, CV_32F);
        }

        if( !varIdx.empty() )
        {
            CV_Assert( (varIdx.checkVector(1, CV_32S, true) > 0 &&
                       checkRange(varIdx, true, 0, 0, ninputvars)) ||
                       varIdx.checkVector(1, CV_8U, true) == ninputvars );
            if( varIdx.type() == CV_8U )
                varIdx = convertMaskToIdx(varIdx);
            varIdx = varIdx.clone();
            std::sort(varIdx.ptr<int>(), varIdx.ptr<int>() + varIdx.total());
        }

        if( !responses.empty() )
        {
            CV_Assert( responses.type() == CV_32F || responses.type() == CV_32S );
            if( (responses.cols == 1 || responses.rows == 1) && (int)responses.total() == nsamples )
                noutputvars = 1;
            else
            {
                CV_Assert( (layout == ROW_SAMPLE && responses.rows == nsamples) ||
                           (layout == COL_SAMPLE && responses.cols == nsamples) );
                noutputvars = layout == ROW_SAMPLE ? responses.cols : responses.rows;
            }
            if( !responses.isContinuous() || (layout == COL_SAMPLE && noutputvars > 1) )
            {
                Mat temp;
                transpose(responses, temp);
                responses = temp;
            }
        }

        int nvars = ninputvars + noutputvars;

        if( !varType.empty() )
        {
            CV_Assert( varType.checkVector(1, CV_8U, true) == nvars &&
                       checkRange(varType, true, 0, VAR_ORDERED, VAR_CATEGORICAL+1) );
        }
        else
        {
            varType.create(1, nvars, CV_8U);
            varType = Scalar::all(VAR_ORDERED);
            if( noutputvars == 1 )
                varType.at<uchar>(ninputvars) = (uchar)(responses.type() < CV_32F ? VAR_CATEGORICAL : VAR_ORDERED);
        }

        if( noutputvars > 1 )
        {
            for( i = 0; i < noutputvars; i++ )
                CV_Assert( varType.at<uchar>(ninputvars + i) == VAR_ORDERED );
        }

        catOfs = Mat::zeros(1, nvars, CV_32SC2);
        missingSubst = Mat::zeros(1, nvars, CV_32F);

        vector<int> labels, counters, sortbuf, tempCatMap;
        vector<Vec2i> tempCatOfs;
        CatMapHash ofshash;

        AutoBuffer<uchar> buf(nsamples);
        Mat non_missing(layout == ROW_SAMPLE ? Size(1, nsamples) : Size(nsamples, 1), CV_8U, buf.data());
        bool haveMissing = !missing.empty();
        if( haveMissing )
        {
            CV_Assert( missing.size() == samples.size() && missing.type() == CV_8U );
        }

        // we iterate through all the variables. For each categorical variable we build a map
        // in order to convert input values of the variable into normalized values (0..catcount_vi-1)
        // often many categorical variables are similar, so we compress the map - try to re-use
        // maps for different variables if they are identical
        for( i = 0; i < ninputvars; i++ )
        {
            Mat values_i = layout == ROW_SAMPLE ? samples.col(i) : samples.row(i);

            if( varType.at<uchar>(i) == VAR_CATEGORICAL )
            {
                preprocessCategorical(values_i, 0, labels, 0, sortbuf);
                missingSubst.at<float>(i) = -1.f;
                int j, m = (int)labels.size();
                CV_Assert( m > 0 );
                int a = labels.front(), b = labels.back();
                const int* currmap = &labels[0];
                int hashval = ((unsigned)a*127 + (unsigned)b)*127 + m;
                CatMapHash::iterator it = ofshash.find(hashval);
                if( it != ofshash.end() )
                {
                    int vi = it->second;
                    Vec2i ofs0 = tempCatOfs[vi];
                    int m0 = ofs0[1] - ofs0[0];
                    const int* map0 = &tempCatMap[ofs0[0]];
                    if( m0 == m && map0[0] == a && map0[m0-1] == b )
                    {
                        for( j = 0; j < m; j++ )
                            if( map0[j] != currmap[j] )
                                break;
                        if( j == m )
                        {
                            // re-use the map
                            tempCatOfs.push_back(ofs0);
                            continue;
                        }
                    }
                }
                else
                    ofshash[hashval] = i;
                Vec2i ofs;
                ofs[0] = (int)tempCatMap.size();
                ofs[1] = ofs[0] + m;
                tempCatOfs.push_back(ofs);
                std::copy(labels.begin(), labels.end(), std::back_inserter(tempCatMap));
            }
            else
            {
                tempCatOfs.push_back(Vec2i(0, 0));
                /*Mat missing_i = layout == ROW_SAMPLE ? missing.col(i) : missing.row(i);
                compare(missing_i, Scalar::all(0), non_missing, CMP_EQ);
                missingSubst.at<float>(i) = (float)(mean(values_i, non_missing)[0]);*/
                missingSubst.at<float>(i) = 0.f;
            }
        }

        if( !tempCatOfs.empty() )
        {
            Mat(tempCatOfs).copyTo(catOfs);
            Mat(tempCatMap).copyTo(catMap);
        }

        if( noutputvars > 0 && varType.at<uchar>(ninputvars) == VAR_CATEGORICAL )
        {
            preprocessCategorical(responses, &normCatResponses, labels, &counters, sortbuf);
            Mat(labels).copyTo(classLabels);
            Mat(counters).copyTo(classCounters);
        }
    }

    Mat convertMaskToIdx(const Mat& mask)
    {
        int i, j, nz = countNonZero(mask), n = mask.cols + mask.rows - 1;
        Mat idx(1, nz, CV_32S);
        for( i = j = 0; i < n; i++ )
            if( mask.at<uchar>(i) )
                idx.at<int>(j++) = i;
        return idx;
    }

    struct CmpByIdx
    {
        CmpByIdx(const int* _data, int _step) : data(_data), step(_step) {}
        bool operator ()(int i, int j) const { return data[i*step] < data[j*step]; }
        const int* data;
        int step;
    };

    void preprocessCategorical(const Mat& data, Mat* normdata, vector<int>& labels,
                               vector<int>* counters, vector<int>& sortbuf)
    {
        CV_Assert((data.cols == 1 || data.rows == 1) && (data.type() == CV_32S || data.type() == CV_32F));
        int* odata = 0;
        int ostep = 0;

        if(normdata)
        {
            normdata->create(data.size(), CV_32S);
            odata = normdata->ptr<int>();
            ostep = normdata->isContinuous() ? 1 : (int)normdata->step1();
        }

        int i, n = data.cols + data.rows - 1;
        sortbuf.resize(n*2);
        int* idx = &sortbuf[0];
        int* idata = (int*)data.ptr<int>();
        int istep = data.isContinuous() ? 1 : (int)data.step1();

        if( data.type() == CV_32F )
        {
            idata = idx + n;
            const float* fdata = data.ptr<float>();
            for( i = 0; i < n; i++ )
            {
                if( fdata[i*istep] == MISSED_VAL )
                    idata[i] = -1;
                else
                {
                    idata[i] = cvRound(fdata[i*istep]);
                    CV_Assert( (float)idata[i] == fdata[i*istep] );
                }
            }
            istep = 1;
        }

        for( i = 0; i < n; i++ )
            idx[i] = i;

        std::sort(idx, idx + n, CmpByIdx(idata, istep));

        int clscount = 1;
        for( i = 1; i < n; i++ )
            clscount += idata[idx[i]*istep] != idata[idx[i-1]*istep];

        int clslabel = -1;
        int prev = ~idata[idx[0]*istep];
        int previdx = 0;

        labels.resize(clscount);
        if(counters)
            counters->resize(clscount);

        for( i = 0; i < n; i++ )
        {
            int l = idata[idx[i]*istep];
            if( l != prev )
            {
                clslabel++;
                labels[clslabel] = l;
                int k = i - previdx;
                if( clslabel > 0 && counters )
                    counters->at(clslabel-1) = k;
                prev = l;
                previdx = i;
            }
            if(odata)
                odata[idx[i]*ostep] = clslabel;
        }
        if(counters)
            counters->at(clslabel) = i - previdx;
    }

    bool loadCSV(const String& filename, int headerLines,
                 int responseStartIdx, int responseEndIdx,
                 const String& varTypeSpec, char delimiter, char missch)
    {
        const int M = 1000000;
        const char delimiters[3] = { ' ', delimiter, '\0' };
        int nvars = 0;
        bool varTypesSet = false;

        clear();

        file = fopen( filename.c_str(), "rt" );

        if( !file )
            return false;

        std::vector<char> _buf(M);
        std::vector<float> allresponses;
        std::vector<float> rowvals;
        std::vector<uchar> vtypes, rowtypes;
        std::vector<uchar> vsymbolflags;
        bool haveMissed = false;
        char* buf = &_buf[0];

        int i, ridx0 = responseStartIdx, ridx1 = responseEndIdx;
        int ninputvars = 0, noutputvars = 0;

        Mat tempSamples, tempMissing, tempResponses;
        MapType tempNameMap;
        int catCounter = 1;

        // skip header lines
        int lineno = 0;
        for(;;lineno++)
        {
            if( !fgets(buf, M, file) )
                break;
            if(lineno < headerLines )
                continue;
            // trim trailing spaces
            int idx = (int)strlen(buf)-1;
            while( idx >= 0 && isspace(buf[idx]) )
                buf[idx--] = '\0';
            // skip spaces in the beginning
            char* ptr = buf;
            while( *ptr != '\0' && isspace(*ptr) )
                ptr++;
            // skip commented off lines
            if(*ptr == '#')
                continue;
            rowvals.clear();
            rowtypes.clear();

            char* token = strtok(buf, delimiters);
            if (!token)
                break;

            for(;;)
            {
                float val=0.f; int tp = 0;
                decodeElem( token, val, tp, missch, tempNameMap, catCounter );
                if( tp == VAR_MISSED )
                    haveMissed = true;
                rowvals.push_back(val);
                rowtypes.push_back((uchar)tp);
                token = strtok(NULL, delimiters);
                if (!token)
                    break;
            }

            if( nvars == 0 )
            {
                if( rowvals.empty() )
                    CV_Error(cv::Error::StsBadArg, "invalid CSV format; no data found");
                nvars = (int)rowvals.size();
                if( !varTypeSpec.empty() && varTypeSpec.size() > 0 )
                {
                    setVarTypes(varTypeSpec, nvars, vtypes);
                    varTypesSet = true;
                }
                else
                    vtypes = rowtypes;
                vsymbolflags.resize(nvars);
                for( i = 0; i < nvars; i++ )
                    vsymbolflags[i] = (uchar)(rowtypes[i] == VAR_CATEGORICAL);

                ridx0 = ridx0 >= 0 ? ridx0 : ridx0 == -1 ? nvars - 1 : -1;
                ridx1 = ridx1 >= 0 ? ridx1 : ridx0 >= 0 ? ridx0+1 : -1;
                CV_Assert(ridx1 > ridx0);
                noutputvars = ridx0 >= 0 ? ridx1 - ridx0 : 0;
                ninputvars = nvars - noutputvars;
            }
            else
                CV_Assert( nvars == (int)rowvals.size() );

            // check var types
            for( i = 0; i < nvars; i++ )
            {
                CV_Assert( (!varTypesSet && vtypes[i] == rowtypes[i]) ||
                           (varTypesSet && (vtypes[i] == rowtypes[i] || rowtypes[i] == VAR_ORDERED)) );
                uchar sflag = (uchar)(rowtypes[i] == VAR_CATEGORICAL);
                if( vsymbolflags[i] == VAR_MISSED )
                    vsymbolflags[i] = sflag;
                else
                    CV_Assert(vsymbolflags[i] == sflag || rowtypes[i] == VAR_MISSED);
            }

            if( ridx0 >= 0 )
            {
                for( i = ridx1; i < nvars; i++ )
                    std::swap(rowvals[i], rowvals[i-noutputvars]);
                for( i = ninputvars; i < nvars; i++ )
                    allresponses.push_back(rowvals[i]);
                rowvals.pop_back();
            }
            Mat rmat(1, ninputvars, CV_32F, &rowvals[0]);
            tempSamples.push_back(rmat);
        }

        closeFile();

        int nsamples = tempSamples.rows;
        if( nsamples == 0 )
            return false;

        if( haveMissed )
            compare(tempSamples, MISSED_VAL, tempMissing, CMP_EQ);

        if( ridx0 >= 0 )
        {
            for( i = ridx1; i < nvars; i++ )
                std::swap(vtypes[i], vtypes[i-noutputvars]);
            if( noutputvars > 1 )
            {
                for( i = ninputvars; i < nvars; i++ )
                    if( vtypes[i] == VAR_CATEGORICAL )
                        CV_Error(cv::Error::StsBadArg,
                                 "If responses are vector values, not scalars, they must be marked as ordered responses");
            }
        }

        if( !varTypesSet && noutputvars == 1 && vtypes[ninputvars] == VAR_ORDERED )
        {
            for( i = 0; i < nsamples; i++ )
                if( allresponses[i] != cvRound(allresponses[i]) )
                    break;
            if( i == nsamples )
                vtypes[ninputvars] = VAR_CATEGORICAL;
        }

        //If there are responses in the csv file, save them. If not, responses matrix will contain just zeros
        if (noutputvars != 0){
            Mat(nsamples, noutputvars, CV_32F, &allresponses[0]).copyTo(tempResponses);
            setData(tempSamples, ROW_SAMPLE, tempResponses, noArray(), noArray(),
                    noArray(), Mat(vtypes).clone(), tempMissing);
        }
        else{
            Mat zero_mat(nsamples, 1, CV_32F, Scalar(0));
            zero_mat.copyTo(tempResponses);
            setData(tempSamples, ROW_SAMPLE, tempResponses, noArray(), noArray(),
                    noArray(), noArray(), tempMissing);
        }
        bool ok = !samples.empty();
        if(ok)
        {
            std::swap(tempNameMap, nameMap);
            Mat(vsymbolflags).copyTo(varSymbolFlags);
        }
        return ok;
    }

    void decodeElem( const char* token, float& elem, int& type,
                     char missch, MapType& namemap, int& counter ) const
    {
        char* stopstring = NULL;
        elem = (float)strtod( token, &stopstring );
        if( *stopstring == missch && strlen(stopstring) == 1 ) // missed value
        {
            elem = MISSED_VAL;
            type = VAR_MISSED;
        }
        else if( *stopstring != '\0' )
        {
            MapType::iterator it = namemap.find(token);
            if( it == namemap.end() )
            {
                elem = (float)counter;
                namemap[token] = counter++;
            }
            else
                elem = (float)it->second;
            type = VAR_CATEGORICAL;
        }
        else
            type = VAR_ORDERED;
    }

    void setVarTypes( const String& s, int nvars, std::vector<uchar>& vtypes ) const
    {
        const char* errmsg = "type spec is not correct; it should have format \"cat\", \"ord\" or "
          "\"ord[n1,n2-n3,n4-n5,...]cat[m1-m2,m3,m4-m5,...]\", where n's and m's are 0-based variable indices";
        const char* str = s.c_str();
        int specCounter = 0;

        vtypes.resize(nvars);

        for( int k = 0; k < 2; k++ )
        {
            const char* ptr = strstr(str, k == 0 ? "ord" : "cat");
            int tp = k == 0 ? VAR_ORDERED : VAR_CATEGORICAL;
            if( ptr ) // parse ord/cat str
            {
                char* stopstring = NULL;

                if( ptr[3] == '\0' )
                {
                    for( int i = 0; i < nvars; i++ )
                        vtypes[i] = (uchar)tp;
                    specCounter = nvars;
                    break;
                }

                if ( ptr[3] != '[')
                    CV_Error( cv::Error::StsBadArg, errmsg );

                ptr += 4; // pass "ord["
                do
                {
                    int b1 = (int)strtod( ptr, &stopstring );
                    if( *stopstring == 0 || (*stopstring != ',' && *stopstring != ']' && *stopstring != '-') )
                        CV_Error( cv::Error::StsBadArg, errmsg );
                    ptr = stopstring + 1;
                    if( (stopstring[0] == ',') || (stopstring[0] == ']'))
                    {
                        CV_Assert( 0 <= b1 && b1 < nvars );
                        vtypes[b1] = (uchar)tp;
                        specCounter++;
                    }
                    else
                    {
                        if( stopstring[0] == '-')
                        {
                            int b2 = (int)strtod( ptr, &stopstring);
                            if ( (*stopstring == 0) || (*stopstring != ',' && *stopstring != ']') )
                                CV_Error( cv::Error::StsBadArg, errmsg );
                            ptr = stopstring + 1;
                            CV_Assert( 0 <= b1 && b1 <= b2 && b2 < nvars );
                            for (int i = b1; i <= b2; i++)
                                vtypes[i] = (uchar)tp;
                            specCounter += b2 - b1 + 1;
                        }
                        else
                            CV_Error( cv::Error::StsBadArg, errmsg );

                    }
                }
                while(*stopstring != ']');
            }
        }

        if( specCounter != nvars )
            CV_Error( cv::Error::StsBadArg, "type of some variables is not specified" );
    }

    void setTrainTestSplitRatio(double ratio, bool shuffle) CV_OVERRIDE
    {
        CV_Assert( 0. <= ratio && ratio <= 1. );
        setTrainTestSplit(cvRound(getNSamples()*ratio), shuffle);
    }

    void setTrainTestSplit(int count, bool shuffle) CV_OVERRIDE
    {
        int i, nsamples = getNSamples();
        CV_Assert( 0 <= count && count < nsamples );

        trainSampleIdx.release();
        testSampleIdx.release();

        if( count == 0 )
            trainSampleIdx = sampleIdx;
        else if( count == nsamples )
            testSampleIdx = sampleIdx;
        else
        {
            Mat mask(1, nsamples, CV_8U);
            uchar* mptr = mask.ptr();
            for( i = 0; i < nsamples; i++ )
                mptr[i] = (uchar)(i < count);
            trainSampleIdx.create(1, count, CV_32S);
            testSampleIdx.create(1, nsamples - count, CV_32S);
            int j0 = 0, j1 = 0;
            const int* sptr = !sampleIdx.empty() ? sampleIdx.ptr<int>() : 0;
            int* trainptr = trainSampleIdx.ptr<int>();
            int* testptr = testSampleIdx.ptr<int>();
            for( i = 0; i < nsamples; i++ )
            {
                int idx = sptr ? sptr[i] : i;
                if( mptr[i] )
                    trainptr[j0++] = idx;
                else
                    testptr[j1++] = idx;
            }
            if( shuffle )
                shuffleTrainTest();
        }
    }

    void shuffleTrainTest() CV_OVERRIDE
    {
        if( !trainSampleIdx.empty() && !testSampleIdx.empty() )
        {
            int i, nsamples = getNSamples(), ntrain = getNTrainSamples(), ntest = getNTestSamples();
            int* trainIdx = trainSampleIdx.ptr<int>();
            int* testIdx = testSampleIdx.ptr<int>();
            RNG& rng = theRNG();

            for( i = 0; i < nsamples; i++)
            {
                int a = rng.uniform(0, nsamples);
                int b = rng.uniform(0, nsamples);
                int* ptra = trainIdx;
                int* ptrb = trainIdx;
                if( a >= ntrain )
                {
                    ptra = testIdx;
                    a -= ntrain;
                    CV_Assert( a < ntest );
                }
                if( b >= ntrain )
                {
                    ptrb = testIdx;
                    b -= ntrain;
                    CV_Assert( b < ntest );
                }
                std::swap(ptra[a], ptrb[b]);
            }
        }
    }

    Mat getTrainSamples(int _layout,
                        bool compressSamples,
                        bool compressVars) const CV_OVERRIDE
    {
        if( samples.empty() )
            return samples;

        if( (!compressSamples || (trainSampleIdx.empty() && sampleIdx.empty())) &&
            (!compressVars || varIdx.empty()) &&
            layout == _layout )
            return samples;

        int drows = getNTrainSamples(), dcols = getNVars();
        Mat sidx = getTrainSampleIdx(), vidx = getVarIdx();
        const float* src0 = samples.ptr<float>();
        const int* sptr = !sidx.empty() ? sidx.ptr<int>() : 0;
        const int* vptr = !vidx.empty() ? vidx.ptr<int>() : 0;
        size_t sstep0 = samples.step/samples.elemSize();
        size_t sstep = layout == ROW_SAMPLE ? sstep0 : 1;
        size_t vstep = layout == ROW_SAMPLE ? 1 : sstep0;

        if( _layout == COL_SAMPLE )
        {
            std::swap(drows, dcols);
            std::swap(sptr, vptr);
            std::swap(sstep, vstep);
        }

        Mat dsamples(drows, dcols, CV_32F);

        for( int i = 0; i < drows; i++ )
        {
            const float* src = src0 + (sptr ? sptr[i] : i)*sstep;
            float* dst = dsamples.ptr<float>(i);

            for( int j = 0; j < dcols; j++ )
                dst[j] = src[(vptr ? vptr[j] : j)*vstep];
        }

        return dsamples;
    }

    void getValues( int vi, InputArray _sidx, float* values ) const CV_OVERRIDE
    {
        Mat sidx = _sidx.getMat();
        int i, n = sidx.checkVector(1, CV_32S), nsamples = getNSamples();
        CV_Assert( 0 <= vi && vi < getNAllVars() );
        CV_Assert( n >= 0 );
        const int* s = n > 0 ? sidx.ptr<int>() : 0;
        if( n == 0 )
            n = nsamples;

        size_t step = samples.step/samples.elemSize();
        size_t sstep = layout == ROW_SAMPLE ? step : 1;
        size_t vstep = layout == ROW_SAMPLE ? 1 : step;

        const float* src = samples.ptr<float>() + vi*vstep;
        float subst = missingSubst.at<float>(vi);
        for( i = 0; i < n; i++ )
        {
            int j = i;
            if( s )
            {
                j = s[i];
                CV_Assert( 0 <= j && j < ((layout == ROW_SAMPLE) ? samples.rows : samples.cols) );
            }
            values[i] = src[j*sstep];
            if( values[i] == MISSED_VAL )
                values[i] = subst;
        }
    }

    void getNormCatValues( int vi, InputArray _sidx, int* values ) const CV_OVERRIDE
    {
        float* fvalues = (float*)values;
        getValues(vi, _sidx, fvalues);
        int i, n = (int)_sidx.total();
        Vec2i ofs = catOfs.at<Vec2i>(vi);
        int m = ofs[1] - ofs[0];

        CV_Assert( m > 0 ); // if m==0, vi is an ordered variable
        const int* cmap = &catMap.at<int>(ofs[0]);
        bool fastMap = (m == cmap[m - 1] - cmap[0] + 1);

        if( fastMap )
        {
            for( i = 0; i < n; i++ )
            {
                int val = cvRound(fvalues[i]);
                int idx = val - cmap[0];
                CV_Assert(cmap[idx] == val);
                values[i] = idx;
            }
        }
        else
        {
            for( i = 0; i < n; i++ )
            {
                int val = cvRound(fvalues[i]);
                int a = 0, b = m, c = -1;

                while( a < b )
                {
                    c = (a + b) >> 1;
                    if( val < cmap[c] )
                        b = c;
                    else if( val > cmap[c] )
                        a = c+1;
                    else
                        break;
                }

                CV_DbgAssert( c >= 0 && val == cmap[c] );
                values[i] = c;
            }
        }
    }

    void getSample(InputArray _vidx, int sidx, float* buf) const CV_OVERRIDE
    {
        CV_Assert(buf != 0 && 0 <= sidx && sidx < getNSamples());
        Mat vidx = _vidx.getMat();
        int i, n = vidx.checkVector(1, CV_32S), nvars = getNAllVars();
        CV_Assert( n >= 0 );
        const int* vptr = n > 0 ? vidx.ptr<int>() : 0;
        if( n == 0 )
            n = nvars;

        size_t step = samples.step/samples.elemSize();
        size_t sstep = layout == ROW_SAMPLE ? step : 1;
        size_t vstep = layout == ROW_SAMPLE ? 1 : step;

        const float* src = samples.ptr<float>() + sidx*sstep;
        for( i = 0; i < n; i++ )
        {
            int j = i;
            if( vptr )
            {
                j = vptr[i];
                CV_Assert( 0 <= j && j < nvars );
            }
            buf[i] = src[j*vstep];
        }
    }

    void getNames(std::vector<String>& names) const CV_OVERRIDE
    {
        size_t n = nameMap.size();
        TrainDataImpl::MapType::const_iterator it = nameMap.begin(),
                                               it_end = nameMap.end();
        names.resize(n+1);
        names[0] = "?";
        for( ; it != it_end; ++it )
        {
            String s = it->first;
            int label = it->second;
            CV_Assert( label > 0 && label <= (int)n );
            names[label] = s;
        }
    }

    Mat getVarSymbolFlags() const CV_OVERRIDE
    {
        return varSymbolFlags;
    }

    FILE* file;
    int layout;
    Mat samples, missing, varType, varIdx, varSymbolFlags, responses, missingSubst;
    Mat sampleIdx, trainSampleIdx, testSampleIdx;
    Mat sampleWeights, catMap, catOfs;
    Mat normCatResponses, classLabels, classCounters;
    MapType nameMap;
};


Ptr<TrainData> TrainData::loadFromCSV(const String& filename,
                                      int headerLines,
                                      int responseStartIdx,
                                      int responseEndIdx,
                                      const String& varTypeSpec,
                                      char delimiter, char missch)
{
    CV_TRACE_FUNCTION_SKIP_NESTED();
    Ptr<TrainDataImpl> td = makePtr<TrainDataImpl>();
    if(!td->loadCSV(filename, headerLines, responseStartIdx, responseEndIdx, varTypeSpec, delimiter, missch))
        td.release();
    return td;
}

Ptr<TrainData> TrainData::create(InputArray samples, int layout, InputArray responses,
                                 InputArray varIdx, InputArray sampleIdx, InputArray sampleWeights,
                                 InputArray varType)
{
    CV_TRACE_FUNCTION_SKIP_NESTED();
    Ptr<TrainDataImpl> td = makePtr<TrainDataImpl>();
    td->setData(samples, layout, responses, varIdx, sampleIdx, sampleWeights, varType, noArray());
    return td;
}

}}

/* End of file. */
