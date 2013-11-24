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

/*!  */
class NormHistogramCostExtractorImpl : public NormHistogramCostExtractor
{
public:
    /* Constructors */
    NormHistogramCostExtractorImpl(int _flag, int _nDummies, float _defaultCost)
    {
        flag=_flag;
        nDummies=_nDummies;
        defaultCost=_defaultCost;
        name_ = "HistogramCostExtractor.NOR";
    }

    /* Destructor */
    ~NormHistogramCostExtractorImpl()
    {
    }

    virtual AlgorithmInfo* info() const { return 0; }

    //! the main operator
    virtual void buildCostMatrix(InputArray descriptors1, InputArray descriptors2, OutputArray costMatrix);

    //! Setters/Getters
    void setNDummies(int _nDummies)
    {
        nDummies=_nDummies;
    }

    int getNDummies() const
    {
        return nDummies;
    }

    void setDefaultCost(float _defaultCost)
    {
        defaultCost=_defaultCost;
    }

    float getDefaultCost() const
    {
        return defaultCost;
    }

    virtual void setNormFlag(int _flag)
    {
        flag=_flag;
    }

    virtual int getNormFlag() const
    {
        return flag;
    }

    //! write/read
    virtual void write(FileStorage& fs) const
    {
        fs << "name" << name_
           << "flag" << flag
           << "dummies" << nDummies
           << "default" << defaultCost;
    }

    virtual void read(const FileNode& fn)
    {
        CV_Assert( (String)fn["name"] == name_ );
        flag = (int)fn["flag"];
        nDummies = (int)fn["dummies"];
        defaultCost = (float)fn["default"];
    }

private:
    int flag;
    int nDummies;
    float defaultCost;

protected:
    String name_;
};

void NormHistogramCostExtractorImpl::buildCostMatrix(InputArray _descriptors1, InputArray _descriptors2, OutputArray _costMatrix)
{
    // size of the costMatrix with dummies //
    Mat descriptors1=_descriptors1.getMat();
    Mat descriptors2=_descriptors2.getMat();
    int costrows = std::max(descriptors1.rows, descriptors2.rows)+nDummies;
    _costMatrix.create(costrows, costrows, CV_32F);
    Mat costMatrix=_costMatrix.getMat();


    // Obtain copies of the descriptors //
    cv::Mat scd1 = descriptors1.clone();
    cv::Mat scd2 = descriptors2.clone();

    // row normalization //
    for(int i=0; i<scd1.rows; i++)
    {
        scd1.row(i)/=(sum(scd1.row(i))[0]+FLT_EPSILON);
    }
    for(int i=0; i<scd2.rows; i++)
    {
        scd2.row(i)/=(sum(scd2.row(i))[0]+FLT_EPSILON);
    }

    // Compute the Cost Matrix //
    for(int i=0; i<costrows; i++)
    {
        for(int j=0; j<costrows; j++)
        {
            if (i<scd1.rows && j<scd2.rows)
            {
                Mat columnDiff = scd1.row(i)-scd2.row(j);
                costMatrix.at<float>(i,j)=(float)norm(columnDiff, flag);
            }
            else
            {
                costMatrix.at<float>(i,j)=defaultCost;
            }
        }
    }
}

Ptr <HistogramCostExtractor> createNormHistogramCostExtractor(int flag, int nDummies, float defaultCost)
{
    return Ptr <HistogramCostExtractor>( new NormHistogramCostExtractorImpl(flag, nDummies, defaultCost) );
}

/*!  */
class EMDHistogramCostExtractorImpl : public EMDHistogramCostExtractor
{
public:
    /* Constructors */
    EMDHistogramCostExtractorImpl(int _flag, int _nDummies, float _defaultCost)
    {
        flag=_flag;
        nDummies=_nDummies;
        defaultCost=_defaultCost;
        name_ = "HistogramCostExtractor.EMD";
    }

    /* Destructor */
    ~EMDHistogramCostExtractorImpl()
    {
    }

    virtual AlgorithmInfo* info() const { return 0; }

    //! the main operator
    virtual void buildCostMatrix(InputArray descriptors1, InputArray descriptors2, OutputArray costMatrix);

    //! Setters/Getters
    void setNDummies(int _nDummies)
    {
        nDummies=_nDummies;
    }

    int getNDummies() const
    {
        return nDummies;
    }

    void setDefaultCost(float _defaultCost)
    {
        defaultCost=_defaultCost;
    }

    float getDefaultCost() const
    {
        return defaultCost;
    }

    virtual void setNormFlag(int _flag)
    {
        flag=_flag;
    }

    virtual int getNormFlag() const
    {
        return flag;
    }

    //! write/read
    virtual void write(FileStorage& fs) const
    {
        fs << "name" << name_
           << "flag" << flag
           << "dummies" << nDummies
           << "default" << defaultCost;
    }

    virtual void read(const FileNode& fn)
    {
        CV_Assert( (String)fn["name"] == name_ );
        flag = (int)fn["flag"];
        nDummies = (int)fn["dummies"];
        defaultCost = (float)fn["default"];
    }

private:
    int flag;
    int nDummies;
    float defaultCost;

protected:
    String name_;
};

void EMDHistogramCostExtractorImpl::buildCostMatrix(InputArray _descriptors1, InputArray _descriptors2, OutputArray _costMatrix)
{
    // size of the costMatrix with dummies //
    Mat descriptors1=_descriptors1.getMat();
    Mat descriptors2=_descriptors2.getMat();
    int costrows = std::max(descriptors1.rows, descriptors2.rows)+nDummies;
    _costMatrix.create(costrows, costrows, CV_32F);
    Mat costMatrix=_costMatrix.getMat();

    // Obtain copies of the descriptors //
    cv::Mat scd1=descriptors1.clone();
    cv::Mat scd2=descriptors2.clone();

    // row normalization //
    for(int i=0; i<scd1.rows; i++)
    {
        cv::Mat row = scd1.row(i);
        scd1.row(i)/=(sum(row)[0]+FLT_EPSILON);
    }
    for(int i=0; i<scd2.rows; i++)
    {
        cv::Mat row = scd2.row(i);
        scd2.row(i)/=(sum(row)[0]+FLT_EPSILON);
    }

    // Compute the Cost Matrix //
    for(int i=0; i<costrows; i++)
    {
        for(int j=0; j<costrows; j++)
        {
            if (i<scd1.rows && j<scd2.rows)
            {
                cv::Mat sig1(scd1.cols,2,CV_32F), sig2(scd2.cols,2,CV_32F);
                sig1.col(0)=scd1.row(i).t();
                sig2.col(0)=scd2.row(j).t();
                for (int k=0; k<sig1.rows; k++)
                {
                    sig1.at<float>(k,1)=float(k);
                }
                for (int k=0; k<sig2.rows; k++)
                {
                    sig2.at<float>(k,1)=float(k);
                }

                costMatrix.at<float>(i,j) = cv::EMD(sig1, sig2, flag);
            }
            else
            {
                costMatrix.at<float>(i,j) = defaultCost;
            }
        }
    }
}

Ptr <HistogramCostExtractor> createEMDHistogramCostExtractor(int flag, int nDummies, float defaultCost)
{
    return Ptr <HistogramCostExtractor>( new EMDHistogramCostExtractorImpl(flag, nDummies, defaultCost) );
}

/*!  */
class ChiHistogramCostExtractorImpl : public ChiHistogramCostExtractor
{
public:
    /* Constructors */
    ChiHistogramCostExtractorImpl(int _nDummies, float _defaultCost)
    {
        name_ = "HistogramCostExtractor.CHI";
        nDummies=_nDummies;
        defaultCost=_defaultCost;
    }

    /* Destructor */
    ~ChiHistogramCostExtractorImpl()
    {
    }

    virtual AlgorithmInfo* info() const { return 0; }

    //! the main operator
    virtual void buildCostMatrix(InputArray descriptors1, InputArray descriptors2, OutputArray costMatrix);

    //! setters / getters
    void setNDummies(int _nDummies)
    {
        nDummies=_nDummies;
    }

    int getNDummies() const
    {
        return nDummies;
    }

    void setDefaultCost(float _defaultCost)
    {
        defaultCost=_defaultCost;
    }

    float getDefaultCost() const
    {
        return defaultCost;
    }

    //! write/read
    virtual void write(FileStorage& fs) const
    {
        fs << "name" << name_
           << "dummies" << nDummies
           << "default" << defaultCost;
    }

    virtual void read(const FileNode& fn)
    {
        CV_Assert( (String)fn["name"] == name_ );
        nDummies = (int)fn["dummies"];
        defaultCost = (float)fn["default"];
    }

protected:
    String name_;
    int nDummies;
    float defaultCost;
};

void ChiHistogramCostExtractorImpl::buildCostMatrix(InputArray _descriptors1, InputArray _descriptors2, OutputArray _costMatrix)
{
    // size of the costMatrix with dummies //
    Mat descriptors1=_descriptors1.getMat();
    Mat descriptors2=_descriptors2.getMat();
    int costrows = std::max(descriptors1.rows, descriptors2.rows)+nDummies;
    _costMatrix.create(costrows, costrows, CV_32FC1);
    Mat costMatrix=_costMatrix.getMat();

    // Obtain copies of the descriptors //
    cv::Mat scd1=descriptors1.clone();
    cv::Mat scd2=descriptors2.clone();

    // row normalization //
    for(int i=0; i<scd1.rows; i++)
    {
        cv::Mat row = scd1.row(i);
        scd1.row(i)/=(sum(row)[0]+FLT_EPSILON);
    }
    for(int i=0; i<scd2.rows; i++)
    {
       cv::Mat row = scd2.row(i);
        scd2.row(i)/=(sum(row)[0]+FLT_EPSILON);
    }

    // Compute the Cost Matrix //
    for(int i=0; i<costrows; i++)
    {
        for(int j=0; j<costrows; j++)
        {
            if (i<scd1.rows && j<scd2.rows)
            {
                float csum = 0;
                for(int k=0; k<scd2.cols; k++)
                {
                    float resta=scd1.at<float>(i,k)-scd2.at<float>(j,k);
                    float suma=scd1.at<float>(i,k)+scd2.at<float>(j,k);
                    csum += resta*resta/(FLT_EPSILON+suma);
                }
                costMatrix.at<float>(i,j)=csum/2;
            }
            else
            {
                costMatrix.at<float>(i,j)=defaultCost;
            }
        }
    }
}

Ptr <HistogramCostExtractor> createChiHistogramCostExtractor(int nDummies, float defaultCost)
{
    return Ptr <HistogramCostExtractor>( new ChiHistogramCostExtractorImpl(nDummies, defaultCost) );
}

/*!  */
class EMDL1HistogramCostExtractorImpl : public EMDL1HistogramCostExtractor
{
public:
    /* Constructors */
    EMDL1HistogramCostExtractorImpl(int _nDummies, float _defaultCost)
    {
        name_ = "HistogramCostExtractor.CHI";
        nDummies=_nDummies;
        defaultCost=_defaultCost;
    }

    /* Destructor */
    ~EMDL1HistogramCostExtractorImpl()
    {
    }

    virtual AlgorithmInfo* info() const { return 0; }

    //! the main operator
    virtual void buildCostMatrix(InputArray descriptors1, InputArray descriptors2, OutputArray costMatrix);

    //! setters / getters
    void setNDummies(int _nDummies)
    {
        nDummies=_nDummies;
    }

    int getNDummies() const
    {
        return nDummies;
    }

    void setDefaultCost(float _defaultCost)
    {
        defaultCost=_defaultCost;
    }

    float getDefaultCost() const
    {
        return defaultCost;
    }

    //! write/read
    virtual void write(FileStorage& fs) const
    {
        fs << "name" << name_
           << "dummies" << nDummies
           << "default" << defaultCost;
    }

    virtual void read(const FileNode& fn)
    {
        CV_Assert( (String)fn["name"] == name_ );
        nDummies = (int)fn["dummies"];
        defaultCost = (float)fn["default"];
    }

protected:
    String name_;
    int nDummies;
    float defaultCost;
};

void EMDL1HistogramCostExtractorImpl::buildCostMatrix(InputArray _descriptors1, InputArray _descriptors2, OutputArray _costMatrix)
{
    // size of the costMatrix with dummies //
    Mat descriptors1=_descriptors1.getMat();
    Mat descriptors2=_descriptors2.getMat();
    int costrows = std::max(descriptors1.rows, descriptors2.rows)+nDummies;
    _costMatrix.create(costrows, costrows, CV_32F);
    Mat costMatrix=_costMatrix.getMat();

    // Obtain copies of the descriptors //
    cv::Mat scd1=descriptors1.clone();
    cv::Mat scd2=descriptors2.clone();

    // row normalization //
    for(int i=0; i<scd1.rows; i++)
    {
        cv::Mat row = scd1.row(i);
        scd1.row(i)/=(sum(row)[0]+FLT_EPSILON);
    }
    for(int i=0; i<scd2.rows; i++)
    {
        cv::Mat row = scd2.row(i);
        scd2.row(i)/=(sum(row)[0]+FLT_EPSILON);
    }

    // Compute the Cost Matrix //
    for(int i=0; i<costrows; i++)
    {
        for(int j=0; j<costrows; j++)
        {
            if (i<scd1.rows && j<scd2.rows)
            {
                cv::Mat sig1(scd1.cols,1,CV_32F), sig2(scd2.cols,1,CV_32F);
                sig1.col(0)=scd1.row(i).t();
                sig2.col(0)=scd2.row(j).t();
                costMatrix.at<float>(i,j) = cv::EMDL1(sig1, sig2);
            }
            else
            {
                costMatrix.at<float>(i,j) = defaultCost;
            }
        }
    }
}

Ptr <HistogramCostExtractor> createEMDL1HistogramCostExtractor(int nDummies, float defaultCost)
{
    return Ptr <HistogramCostExtractor>( new EMDL1HistogramCostExtractorImpl(nDummies, defaultCost) );
}

} // cv
