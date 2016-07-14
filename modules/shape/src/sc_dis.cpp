/*M///////////////////////////////////////////////////////////////////////////////////////
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

/*
 * Implementation of the paper Shape Matching and Object Recognition Using Shape Contexts
 * Belongie et al., 2002 by Juan Manuel Perez for GSoC 2013.
 */

#include "precomp.hpp"
#include "opencv2/core.hpp"
#include "scd_def.hpp"
#include <limits>

namespace cv
{
class ShapeContextDistanceExtractorImpl : public ShapeContextDistanceExtractor
{
public:
    /* Constructors */
    ShapeContextDistanceExtractorImpl(int _nAngularBins, int _nRadialBins, float _innerRadius, float _outerRadius, int _iterations,
                                      const Ptr<HistogramCostExtractor> &_comparer, const Ptr<ShapeTransformer> &_transformer)
    {
        nAngularBins=_nAngularBins;
        nRadialBins=_nRadialBins;
        innerRadius=_innerRadius;
        outerRadius=_outerRadius;
        rotationInvariant=false;
        comparer=_comparer;
        iterations=_iterations;
        transformer=_transformer;
        bendingEnergyWeight=0.3f;
        imageAppearanceWeight=0.0f;
        shapeContextWeight=1.0f;
        sigma=10.0f;
        name_ = "ShapeDistanceExtractor.SCD";
    }

    /* Destructor */
    ~ShapeContextDistanceExtractorImpl()
    {
    }

    //! the main operator
    virtual float computeDistance(InputArray contour1, InputArray contour2);

    //! Setters/Getters
    virtual void setAngularBins(int _nAngularBins){CV_Assert(_nAngularBins>0); nAngularBins=_nAngularBins;}
    virtual int getAngularBins() const {return nAngularBins;}

    virtual void setRadialBins(int _nRadialBins){CV_Assert(_nRadialBins>0); nRadialBins=_nRadialBins;}
    virtual int getRadialBins() const {return nRadialBins;}

    virtual void setInnerRadius(float _innerRadius) {CV_Assert(_innerRadius>0); innerRadius=_innerRadius;}
    virtual float getInnerRadius() const {return innerRadius;}

    virtual void setOuterRadius(float _outerRadius) {CV_Assert(_outerRadius>0); outerRadius=_outerRadius;}
    virtual float getOuterRadius() const {return outerRadius;}

    virtual void setRotationInvariant(bool _rotationInvariant) {rotationInvariant=_rotationInvariant;}
    virtual bool getRotationInvariant() const {return rotationInvariant;}

    virtual void setCostExtractor(Ptr<HistogramCostExtractor> _comparer) { comparer = _comparer; }
    virtual Ptr<HistogramCostExtractor> getCostExtractor() const { return comparer; }

    virtual void setShapeContextWeight(float _shapeContextWeight) {shapeContextWeight=_shapeContextWeight;}
    virtual float getShapeContextWeight() const {return shapeContextWeight;}

    virtual void setImageAppearanceWeight(float _imageAppearanceWeight) {imageAppearanceWeight=_imageAppearanceWeight;}
    virtual float getImageAppearanceWeight() const {return imageAppearanceWeight;}

    virtual void setBendingEnergyWeight(float _bendingEnergyWeight) {bendingEnergyWeight=_bendingEnergyWeight;}
    virtual float getBendingEnergyWeight() const {return bendingEnergyWeight;}

    virtual void setStdDev(float _sigma) {sigma=_sigma;}
    virtual float getStdDev() const {return sigma;}

    virtual void setImages(InputArray _image1, InputArray _image2)
    {
        Mat image1_=_image1.getMat(), image2_=_image2.getMat();
        CV_Assert((image1_.depth()==0) && (image2_.depth()==0));
        image1=image1_;
        image2=image2_;
    }

    virtual void getImages(OutputArray _image1, OutputArray _image2) const
    {
        CV_Assert((!image1.empty()) && (!image2.empty()));
        image1.copyTo(_image1);
        image2.copyTo(_image2);
    }

    virtual void setIterations(int _iterations) {CV_Assert(_iterations>0); iterations=_iterations;}
    virtual int getIterations() const {return iterations;}

    virtual void setTransformAlgorithm(Ptr<ShapeTransformer> _transformer) {transformer=_transformer;}
    virtual Ptr<ShapeTransformer> getTransformAlgorithm() const {return transformer;}

    //! write/read
    virtual void write(FileStorage& fs) const
    {
        writeFormat(fs);
        fs << "name" << name_
           << "nRads" << nRadialBins
           << "nAngs" << nAngularBins
           << "iters" << iterations
           << "img_1" << image1
           << "img_2" << image2
           << "beWei" << bendingEnergyWeight
           << "scWei" << shapeContextWeight
           << "iaWei" << imageAppearanceWeight
           << "costF" << costFlag
           << "rotIn" << rotationInvariant
           << "sigma" << sigma;
    }

    virtual void read(const FileNode& fn)
    {
        CV_Assert( (String)fn["name"] == name_ );
        nRadialBins = (int)fn["nRads"];
        nAngularBins = (int)fn["nAngs"];
        iterations = (int)fn["iters"];
        bendingEnergyWeight = (float)fn["beWei"];
        shapeContextWeight = (float)fn["scWei"];
        imageAppearanceWeight = (float)fn["iaWei"];
        costFlag = (int)fn["costF"];
        sigma = (float)fn["sigma"];
    }

protected:
    int nAngularBins;
    int nRadialBins;
    float innerRadius;
    float outerRadius;
    bool rotationInvariant;
    int costFlag;
    int iterations;
    Ptr<ShapeTransformer> transformer;
    Ptr<HistogramCostExtractor> comparer;
    Mat image1;
    Mat image2;
    float bendingEnergyWeight;
    float imageAppearanceWeight;
    float shapeContextWeight;
    float sigma;
    String name_;
};

float ShapeContextDistanceExtractorImpl::computeDistance(InputArray contour1, InputArray contour2)
{
    // Checking //
    Mat sset1=contour1.getMat(), sset2=contour2.getMat(), set1, set2;
    if (set1.type() != CV_32F)
        sset1.convertTo(set1, CV_32F);
    else
        sset1.copyTo(set1);

    if (set2.type() != CV_32F)
        sset2.convertTo(set2, CV_32F);
    else
        sset2.copyTo(set2);

    CV_Assert((set1.channels()==2) && (set1.cols>0));
    CV_Assert((set2.channels()==2) && (set2.cols>0));
    if (imageAppearanceWeight!=0)
    {
        CV_Assert((!image1.empty()) && (!image2.empty()));
    }

    // Initializing Extractor, Descriptor structures and Matcher //
    SCD set1SCE(nAngularBins, nRadialBins, innerRadius, outerRadius, rotationInvariant);
    Mat set1SCD;
    SCD set2SCE(nAngularBins, nRadialBins, innerRadius, outerRadius, rotationInvariant);
    Mat set2SCD;
    SCDMatcher matcher;
    std::vector<DMatch> matches;

    // Distance components (The output is a linear combination of these 3) //
    float sDistance=0, bEnergy=0, iAppearance=0;
    float beta;

    // Initializing some variables //
    std::vector<int> inliers1, inliers2;

    Ptr<ThinPlateSplineShapeTransformer> transDown = transformer.dynamicCast<ThinPlateSplineShapeTransformer>();

    Mat warpedImage;
    int ii, jj, pt;

    for (ii=0; ii<iterations; ii++)
    {
        // Extract SCD descriptor in the set1 //
        set1SCE.extractSCD(set1, set1SCD, inliers1);

        // Extract SCD descriptor of the set2 (TARGET) //
        set2SCE.extractSCD(set2, set2SCD, inliers2, set1SCE.getMeanDistance());

        // regularization parameter with annealing rate annRate //
        beta=set1SCE.getMeanDistance();
        beta *= beta;

        // match //
        matcher.matchDescriptors(set1SCD, set2SCD, matches, comparer, inliers1, inliers2);

        // apply TPS transform //
        if ( !transDown.empty() )
            transDown->setRegularizationParameter(beta);
        transformer->estimateTransformation(set1, set2, matches);
        bEnergy += transformer->applyTransformation(set1, set1);

        // Image appearance //
        if (imageAppearanceWeight!=0)
        {
            // Have to accumulate the transformation along all the iterations
            if (ii==0)
            {
                if ( !transDown.empty() )
                {
                    image2.copyTo(warpedImage);
                }
                else
                {
                    image1.copyTo(warpedImage);
                }
            }
            transformer->warpImage(warpedImage, warpedImage);
        }
    }

    Mat gaussWindow, diffIm;
    if (imageAppearanceWeight!=0)
    {
        // compute appearance cost
        if ( !transDown.empty() )
        {
            resize(warpedImage, warpedImage, image1.size());
            Mat temp=(warpedImage-image1);
            multiply(temp, temp, diffIm);
        }
        else
        {
            resize(warpedImage, warpedImage, image2.size());
            Mat temp=(warpedImage-image2);
            multiply(temp, temp, diffIm);
        }
        gaussWindow = Mat::zeros(warpedImage.rows, warpedImage.cols, CV_32F);
        for (pt=0; pt<sset1.cols; pt++)
        {
            Point2f p = sset1.at<Point2f>(0,pt);
            for (ii=0; ii<diffIm.rows; ii++)
            {
                for (jj=0; jj<diffIm.cols; jj++)
                {
                    float val = float(std::exp( -float( (p.x-jj)*(p.x-jj) + (p.y-ii)*(p.y-ii) )/(2*sigma*sigma) ) / (sigma*sigma*2*CV_PI));
                    gaussWindow.at<float>(ii,jj) += val;
                }
            }
        }

        Mat appIm(diffIm.rows, diffIm.cols, CV_32F);
        for (ii=0; ii<diffIm.rows; ii++)
        {
            for (jj=0; jj<diffIm.cols; jj++)
            {
                float elema=float( diffIm.at<uchar>(ii,jj) )/255;
                float elemb=gaussWindow.at<float>(ii,jj);
                appIm.at<float>(ii,jj) = elema*elemb;
            }
        }
        iAppearance = float(cv::sum(appIm)[0]/sset1.cols);
    }
    sDistance = matcher.getMatchingCost();

    return (sDistance*shapeContextWeight+bEnergy*bendingEnergyWeight+iAppearance*imageAppearanceWeight);
}

Ptr <ShapeContextDistanceExtractor> createShapeContextDistanceExtractor(int nAngularBins, int nRadialBins, float innerRadius, float outerRadius, int iterations,
                                                                        const Ptr<HistogramCostExtractor> &comparer, const Ptr<ShapeTransformer> &transformer)
{
    return Ptr <ShapeContextDistanceExtractor> ( new ShapeContextDistanceExtractorImpl(nAngularBins, nRadialBins, innerRadius,
                                                                                       outerRadius, iterations, comparer, transformer) );
}

//! SCD
void SCD::extractSCD(cv::Mat &contour, cv::Mat &descriptors, const std::vector<int> &queryInliers, const float _meanDistance)
{
    cv::Mat contourMat = contour;
    cv::Mat disMatrix = cv::Mat::zeros(contourMat.cols, contourMat.cols, CV_32F);
    cv::Mat angleMatrix = cv::Mat::zeros(contourMat.cols, contourMat.cols, CV_32F);

    std::vector<double> logspaces, angspaces;
    logarithmicSpaces(logspaces);
    angularSpaces(angspaces);
    buildNormalizedDistanceMatrix(contourMat, disMatrix, queryInliers, _meanDistance);
    buildAngleMatrix(contourMat, angleMatrix);

    // Now, build the descriptor matrix (each row is a point) //
    descriptors = cv::Mat::zeros(contourMat.cols, descriptorSize(), CV_32F);

    for (int ptidx=0; ptidx<contourMat.cols; ptidx++)
    {
        for (int cmp=0; cmp<contourMat.cols; cmp++)
        {
            if (ptidx==cmp) continue;
            if ((int)queryInliers.size()>0)
            {
                if (queryInliers[ptidx]==0 || queryInliers[cmp]==0) continue; //avoid outliers
            }

            int angidx=-1, radidx=-1;
            for (int i=0; i<nRadialBins; i++)
            {
                if (disMatrix.at<float>(ptidx, cmp)<logspaces[i])
                {
                    radidx=i;
                    break;
                }
            }
            for (int i=0; i<nAngularBins; i++)
            {
                if (angleMatrix.at<float>(ptidx, cmp)<angspaces[i])
                {
                    angidx=i;
                    break;
                }
            }
            if (angidx!=-1 && radidx!=-1)
            {
                int idx = angidx+radidx*nAngularBins;
                descriptors.at<float>(ptidx, idx)++;
            }
        }
    }
}

void SCD::logarithmicSpaces(std::vector<double> &vecSpaces) const
{
    double logmin=log10(innerRadius);
    double logmax=log10(outerRadius);
    double delta=(logmax-logmin)/(nRadialBins-1);
    double accdelta=0;

    for (int i=0; i<nRadialBins; i++)
    {
        double val = std::pow(10,logmin+accdelta);
        vecSpaces.push_back(val);
        accdelta += delta;
    }
}

void SCD::angularSpaces(std::vector<double> &vecSpaces) const
{
    double delta=2*CV_PI/nAngularBins;
    double val=0;

    for (int i=0; i<nAngularBins; i++)
    {
        val += delta;
        vecSpaces.push_back(val);
    }
}

void SCD::buildNormalizedDistanceMatrix(cv::Mat &contour, cv::Mat &disMatrix, const std::vector<int> &queryInliers, const float _meanDistance)
{
    cv::Mat contourMat = contour;
    cv::Mat mask(disMatrix.rows, disMatrix.cols, CV_8U);

    for (int i=0; i<contourMat.cols; i++)
    {
      for (int j=0; j<contourMat.cols; j++)
      {
          disMatrix.at<float>(i,j) = (float)norm( cv::Mat(contourMat.at<cv::Point2f>(0,i)-contourMat.at<cv::Point2f>(0,j)), cv::NORM_L2 );
          if (_meanDistance<0)
          {
              if (queryInliers.size()>0)
              {
                  mask.at<char>(i,j)=char(queryInliers[j] && queryInliers[i]);
              }
              else
              {
                  mask.at<char>(i,j)=1;
              }
          }
      }
    }

    if (_meanDistance<0)
    {
      meanDistance=(float)mean(disMatrix, mask)[0];
    }
    else
    {
      meanDistance=_meanDistance;
    }
    disMatrix/=meanDistance+FLT_EPSILON;
}

void SCD::buildAngleMatrix(cv::Mat &contour, cv::Mat &angleMatrix) const
{
    cv::Mat contourMat = contour;

    // if descriptor is rotationInvariant compute massCenter //
    cv::Point2f massCenter(0,0);
    if (rotationInvariant)
    {
        for (int i=0; i<contourMat.cols; i++)
        {
            massCenter+=contourMat.at<cv::Point2f>(0,i);
        }
        massCenter.x=massCenter.x/(float)contourMat.cols;
        massCenter.y=massCenter.y/(float)contourMat.cols;
    }


    for (int i=0; i<contourMat.cols; i++)
    {
        for (int j=0; j<contourMat.cols; j++)
        {
            if (i==j)
            {
                angleMatrix.at<float>(i,j)=0.0;
            }
            else
            {
                cv::Point2f dif = contourMat.at<cv::Point2f>(0,i) - contourMat.at<cv::Point2f>(0,j);
                angleMatrix.at<float>(i,j) = std::atan2(dif.y, dif.x);

                if (rotationInvariant)
                {
                    cv::Point2f refPt = contourMat.at<cv::Point2f>(0,i) - massCenter;
                    float refAngle = atan2(refPt.y, refPt.x);
                    angleMatrix.at<float>(i,j) -= refAngle;
                }
                angleMatrix.at<float>(i,j) = float(fmod(double(angleMatrix.at<float>(i,j)+(double)FLT_EPSILON),2*CV_PI)+CV_PI);
            }
        }
    }
}

//! SCDMatcher
void SCDMatcher::matchDescriptors(cv::Mat &descriptors1, cv::Mat &descriptors2, std::vector<cv::DMatch> &matches,
                                  cv::Ptr<cv::HistogramCostExtractor> &comparer, std::vector<int> &inliers1, std::vector<int> &inliers2)
{
    matches.clear();

    // Build the cost Matrix between descriptors //
    cv::Mat costMat;
    buildCostMatrix(descriptors1, descriptors2, costMat, comparer);

    // Solve the matching problem using the hungarian method //
    hungarian(costMat, matches, inliers1, inliers2, descriptors1.rows, descriptors2.rows);
}

void SCDMatcher::buildCostMatrix(const cv::Mat &descriptors1, const cv::Mat &descriptors2,
                                 cv::Mat &costMatrix, cv::Ptr<cv::HistogramCostExtractor> &comparer) const
{
    comparer->buildCostMatrix(descriptors1, descriptors2, costMatrix);
}

void SCDMatcher::hungarian(cv::Mat &costMatrix, std::vector<cv::DMatch> &outMatches, std::vector<int> &inliers1,
                           std::vector<int> &inliers2, int sizeScd1, int sizeScd2)
{
    std::vector<int> free(costMatrix.rows, 0), collist(costMatrix.rows, 0);
    std::vector<int> matches(costMatrix.rows, 0), colsol(costMatrix.rows), rowsol(costMatrix.rows);
    std::vector<float> d(costMatrix.rows), pred(costMatrix.rows), v(costMatrix.rows);

    const float LOWV = 1e-10f;
    bool unassignedfound;
    int  i=0, imin=0, numfree=0, prvnumfree=0, f=0, i0=0, k=0, freerow=0;
    int  j=0, j1=0, j2=0, endofpath=0, last=0, low=0, up=0;
    float min=0, h=0, umin=0, usubmin=0, v2=0;

    // COLUMN REDUCTION //
    for (j = costMatrix.rows-1; j >= 0; j--)
    {
        // find minimum cost over rows.
        min = costMatrix.at<float>(0,j);
        imin = 0;
        for (i = 1; i < costMatrix.rows; i++)
        if (costMatrix.at<float>(i,j) < min)
        {
            min = costMatrix.at<float>(i,j);
            imin = i;
        }
        v[j] = min;

        if (++matches[imin] == 1)
        {
            rowsol[imin] = j;
            colsol[j] = imin;
        }
        else
        {
            colsol[j]=-1;
        }
    }

    // REDUCTION TRANSFER //
    for (i=0; i<costMatrix.rows; i++)
    {
        if (matches[i] == 0)
        {
            free[numfree++] = i;
        }
        else
        {
            if (matches[i] == 1)
            {
                j1=rowsol[i];
                min=std::numeric_limits<float>::max();
                for (j=0; j<costMatrix.rows; j++)
                {
                    if (j!=j1)
                    {
                        if (costMatrix.at<float>(i,j)-v[j] < min)
                        {
                            min=costMatrix.at<float>(i,j)-v[j];
                        }
                    }
                }
                v[j1] = v[j1]-min;
            }
        }
    }
    // AUGMENTING ROW REDUCTION //
    int loopcnt = 0;
    do
    {
        loopcnt++;
        k=0;
        prvnumfree=numfree;
        numfree=0;
        while (k < prvnumfree)
        {
            i=free[k];
            k++;
            umin = costMatrix.at<float>(i,0)-v[0];
            j1=0;
            usubmin = std::numeric_limits<float>::max();
            for (j=1; j<costMatrix.rows; j++)
            {
                h = costMatrix.at<float>(i,j)-v[j];
                if (h < usubmin)
                {
                    if (h >= umin)
                    {
                        usubmin = h;
                        j2 = j;
                    }
                    else
                    {
                        usubmin = umin;
                        umin = h;
                        j2 = j1;
                        j1 = j;
                    }
                }
            }
            i0 = colsol[j1];

            if (fabs(umin-usubmin) > LOWV) //if( umin < usubmin )
            {
                v[j1] = v[j1] - (usubmin - umin);
            }
            else // minimum and subminimum equal.
            {
                if (i0 >= 0) // minimum column j1 is assigned.
                {
                    j1 = j2;
                    i0 = colsol[j2];
                }
            }
            // (re-)assign i to j1, possibly de-assigning an i0.
            rowsol[i]=j1;
            colsol[j1]=i;

            if (i0 >= 0)
            {
                //if( umin < usubmin )
                if (fabs(umin-usubmin) > LOWV)
                {
                    free[--k] = i0;
                }
                else
                {
                    free[numfree++] = i0;
                }
            }
        }
    }while (loopcnt<2); // repeat once.

    // AUGMENT SOLUTION for each free row //
    for (f = 0; f<numfree; f++)
    {
        freerow = free[f];       // start row of augmenting path.
        // Dijkstra shortest path algorithm.
        // runs until unassigned column added to shortest path tree.
        for (j = 0; j < costMatrix.rows; j++)
        {
            d[j] = costMatrix.at<float>(freerow,j) - v[j];
            pred[j] = float(freerow);
            collist[j] = j;        // init column list.
        }

        low=0; // columns in 0..low-1 are ready, now none.
        up=0;  // columns in low..up-1 are to be scanned for current minimum, now none.
        unassignedfound = false;
        do
        {
            if (up == low)
            {
                last=low-1;
                min = d[collist[up++]];
                for (k = up; k < costMatrix.rows; k++)
                {
                    j = collist[k];
                    h = d[j];
                    if (h <= min)
                    {
                        if (h < min) // new minimum.
                        {
                            up = low; // restart list at index low.
                            min = h;
                        }
                        collist[k] = collist[up];
                        collist[up++] = j;
                    }
                }
                for (k=low; k<up; k++)
                {
                    if (colsol[collist[k]] < 0)
                    {
                        endofpath = collist[k];
                        unassignedfound = true;
                        break;
                    }
                }
            }

            if (!unassignedfound)
            {
                // update 'distances' between freerow and all unscanned columns, via next scanned column.
                j1 = collist[low];
                low++;
                i = colsol[j1];
                h = costMatrix.at<float>(i,j1)-v[j1]-min;

                for (k = up; k < costMatrix.rows; k++)
                {
                    j = collist[k];
                    v2 = costMatrix.at<float>(i,j) - v[j] - h;
                    if (v2 < d[j])
                    {
                        pred[j] = float(i);
                        if (v2 == min)
                        {
                            if (colsol[j] < 0)
                            {
                                // if unassigned, shortest augmenting path is complete.
                                endofpath = j;
                                unassignedfound = true;
                                break;
                            }
                            else
                            {
                                collist[k] = collist[up];
                                collist[up++] = j;
                            }
                        }
                        d[j] = v2;
                    }
                }
            }
        }while (!unassignedfound);

        // update column prices.
        for (k = 0; k <= last; k++)
        {
            j1 = collist[k];
            v[j1] = v[j1] + d[j1] - min;
        }

        // reset row and column assignments along the alternating path.
        do
        {
            i = int(pred[endofpath]);
            colsol[endofpath] = i;
            j1 = endofpath;
            endofpath = rowsol[i];
            rowsol[i] = j1;
        }while (i != freerow);
    }

    // calculate symmetric shape context cost
    cv::Mat trueCostMatrix(costMatrix, cv::Rect(0,0,sizeScd1, sizeScd2));
    float leftcost = 0;
    for (int nrow=0; nrow<trueCostMatrix.rows; nrow++)
    {
        double minval;
        minMaxIdx(trueCostMatrix.row(nrow), &minval);
        leftcost+=float(minval);
    }
    leftcost /= trueCostMatrix.rows;

    float rightcost = 0;
    for (int ncol=0; ncol<trueCostMatrix.cols; ncol++)
    {
        double minval;
        minMaxIdx(trueCostMatrix.col(ncol), &minval);
        rightcost+=float(minval);
    }
    rightcost /= trueCostMatrix.cols;

    minMatchCost = std::max(leftcost,rightcost);

    // Save in a DMatch vector
    for (i=0;i<costMatrix.cols;i++)
    {
        cv::DMatch singleMatch(colsol[i],i,costMatrix.at<float>(colsol[i],i));//queryIdx,trainIdx,distance
        outMatches.push_back(singleMatch);
    }

    // Update inliers
    inliers1.reserve(sizeScd1);
    for (size_t kc = 0; kc<inliers1.size(); kc++)
    {
        if (rowsol[kc]<sizeScd1) // if a real match
            inliers1[kc]=1;
        else
            inliers1[kc]=0;
    }
    inliers2.reserve(sizeScd2);
    for (size_t kc = 0; kc<inliers2.size(); kc++)
    {
        if (colsol[kc]<sizeScd2) // if a real match
            inliers2[kc]=1;
        else
            inliers2[kc]=0;
    }
}

}
