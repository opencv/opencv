// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "../precomp.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/flann.hpp>

#include "opencv2/objdetect/fractal_detector.hpp"
#include "opencv2/objdetect/fractal_marker.hpp"

#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace cv {
namespace aruco {


void FractalMarkerDetector::setParams(const std::string& fractal_config, float markerSize) {
    fractalMarkerSet = FractalMarkerSet(fractal_config);
    if (markerSize != -1) {
        fractalMarkerSet.convertToMeters(markerSize);
    }
}


std::vector<FractalMarker>  FractalMarkerDetector::detect(const cv::Mat &img){

    cv::Mat bwimage,thresImage;

    std::vector<std::pair<int, std::vector<cv::Point2f>>> candidates;

    std::vector<FractalMarker> DetectedFractalMarkers;

    //first, convert to bw
    if(img.channels()==3)
        cv::cvtColor(img,bwimage,cv::COLOR_BGR2GRAY);
    else bwimage=img;


    ///////////////////////////////////////////////////
    // Adaptive Threshold to detect border
    int adaptiveWindowSize=std::max(int(3),int(15*float(bwimage.cols)/1920.));
    if( adaptiveWindowSize%2==0) adaptiveWindowSize++;
    cv::adaptiveThreshold(bwimage, thresImage, 255.,cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY_INV, adaptiveWindowSize, 7);

    ///////////////////////////////////////////////////
    // compute marker candidates by detecting contours
    //if image is eroded, minSize must be adapted
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Point> approxCurve;
    cv::findContours(thresImage, contours, cv::noArray(), cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

    //analyze  it is a paralelepiped likely to be the marker
    for (unsigned int i = 0; i < contours.size(); i++)
    {
        // check it is a possible element by first checking that is is large enough
        if (120 > int(contours[i].size())  ) continue;
        // can approximate to a convex rect?
        cv::approxPolyDP(contours[i], approxCurve, double(contours[i].size()) * 0.05, true);

        if (approxCurve.size() != 4 || !cv::isContourConvex(approxCurve)) continue;
        // add the points
        std::vector<cv::Point2f> markerCandidate;
        for (int j = 0; j < 4; j++)
            markerCandidate.push_back( cv::Point2f( approxCurve[j].x,approxCurve[j].y));

        //sort corner in clockwise direction
        markerCandidate=sort(markerCandidate);

        //extract the code
        //obtain the intensities of the bits using homography

        std::vector<cv::Point2f> in = {cv::Point2f(0,0), cv::Point2f(1,0), cv::Point2f(1,1), cv::Point2f(0,1)};
        cv::Mat H = cv::getPerspectiveTransform(in, markerCandidate);

        for(auto b_vm:fractalMarkerSet.bits_ids)
        {
            int nbitsWithBorder = sqrt(b_vm.first)+2;
            cv::Mat bits(nbitsWithBorder,nbitsWithBorder,CV_8UC1);
            int pixelSum=0;

            for(int r=0;r<bits.rows;r++){
                for(int c=0;c<bits.cols;c++){
                    float x = float(c+0.5f) / float(bits.cols);
                    float y = float(r+0.5f) / float(bits.rows);
                    double* m = H.ptr<double>(0);
                    double a = m[0]*x + m[1]*y + m[2];
                    double b = m[3]*x + m[4]*y + m[5];
                    double c_ = m[6]*x + m[7]*y + m[8];
                    cv::Point2f mapped(a/c_, b/c_);
                    auto pixelValue = uchar(0.5 + getSubpixelValue(bwimage, mapped));
                    bits.at<uchar>(r,c) = pixelValue;
                    pixelSum += pixelValue;
                }
            }

            //threshold by the average value
            double mean=double(pixelSum)/double(bits.cols*bits.rows);
            cv::threshold(bits,bits,mean,255,cv::THRESH_BINARY);

            //now, analyze the inner code to see if is a marker.
            //  If so, rotate to have the points properly sorted
            int nrotations=0;

            int id=getMarkerId(bits, nrotations, b_vm.second, fractalMarkerSet);

            if(id==-1) continue;//not a marker
            std::rotate(markerCandidate.begin(),markerCandidate.begin() + 4 - nrotations,markerCandidate.end());
            candidates.push_back(std::make_pair(id,markerCandidate));
        }
    }

    ////////////////////////////////////////////
    //remove duplicates
    // sort by id and within same id set the largest first
    std::sort(candidates.begin(), candidates.end(),[](const std::pair<int, std::vector<cv::Point2f>> &a,const std::pair<int, std::vector<cv::Point2f>> &b){
        if( a.first<b.first) return true;
        else if( a.first==b.first) return perimeter(a.second)>perimeter(b.second);
        else return false;
    });

     // Using std::unique remove duplicates
       auto ip = std::unique(candidates.begin(), candidates.end(),[](const std::pair<int, std::vector<cv::Point2f>> &a,const std::pair<int, std::vector<cv::Point2f>> &b){return a.first==b.first;});
       candidates.resize(std::distance(candidates.begin(), ip));

       if(candidates.size()>0){
           ////////////////////////////////////////////
           //finally subpixel corner refinement
           int halfwsize= 4*float(bwimage.cols)/float(bwimage.cols) +0.5 ;
           std::vector<cv::Point2f> Corners;
           for (const auto &m:candidates)
               Corners.insert(Corners.end(), m.second.begin(),m.second.end());
           cv::cornerSubPix(bwimage, Corners, cv::Size(halfwsize,halfwsize), cv::Size(-1, -1),cv::TermCriteria( cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS, 12, 0.005));
           // copy back to the markers
           for (unsigned int i = 0; i < candidates.size(); i++)
           {
               DetectedFractalMarkers.push_back(fractalMarkerSet.fractalMarkerCollection[candidates[i].first]);
               for (int c = 0; c < 4; c++) DetectedFractalMarkers[i].push_back(Corners[i * 4 + c]);
           }
       }

    //Done
    return DetectedFractalMarkers;
}


std::vector<FractalMarker> FractalMarkerDetector::detect(const cv::Mat &img, std::vector<cv::Point3f>& p3d,
                                                  std::vector<cv::Point2f>& p2d)
{
    // Convert to grayscale if needed
    cv::Mat bwimage;
    if(img.channels()==3)
        cv::cvtColor(img, bwimage, cv::COLOR_BGR2GRAY);
    else 
        bwimage = img;

    // Fractal marker detection
    std::vector<FractalMarker> detected = detect(bwimage);

    if(detected.size() > 0) {
        // Prepare points for homography
        std::vector<cv::Point2f> imgpoints;
        std::vector<cv::Point3f> objpoints;
        for(auto marker : detected) {
            for(auto p : marker)
                imgpoints.push_back(p);

            for(int c = 0; c < 4; c++) {
                cv::KeyPoint kpt = fractalMarkerSet.fractalMarkerCollection[marker.id].getKeypts()[c];
                objpoints.push_back(cv::Point3f(kpt.pt.x, kpt.pt.y, 0));
            }
        }

        // FAST feature detection
        std::vector<cv::KeyPoint> kpoints;
        cv::Ptr<cv::FastFeatureDetector> fd = cv::FastFeatureDetector::create();
        fd->detect(bwimage, kpoints);
        
        // Filter keypoints
        kfilter(kpoints);
        assignClass(bwimage, kpoints);

        // Build FLANN index
        cv::Mat kpointsMat(kpoints.size(), 2, CV_32F);
        for (size_t i = 0; i < kpoints.size(); ++i)
        {
            kpointsMat.at<float>(i, 0) = kpoints[i].pt.x;
            kpointsMat.at<float>(i, 1) = kpoints[i].pt.y;
        }
        cv::flann::Index Kdtree;
        Kdtree.build(kpointsMat, cv::flann::KDTreeIndexParams(1), cvflann::FLANN_DIST_EUCLIDEAN);

        // Compute homography
        cv::Mat H = cv::findHomography(objpoints, imgpoints);

        // Process each marker
        std::vector<int> nearestIdxList;
        std::vector<float> distsList;
        for (auto &fm : fractalMarkerSet.fractalMarkerCollection) {
            std::vector<cv::Point2f> imgPoints;
            std::vector<cv::Point2f> objPoints;
            std::vector<cv::KeyPoint> objKeyPoints = fm.second.getKeypts();
        
            for (auto kpt : objKeyPoints)
                objPoints.push_back(cv::Point2f(kpt.pt.x, kpt.pt.y));
        
            cv::perspectiveTransform(objPoints, imgPoints, H);
        
            // We consider only markers whose internal points are separated by a specific distance.
            bool consider = true;
            for (size_t i = 0; i < imgPoints.size() - 1 && consider; i++)
                for (size_t j = i + 1; j < imgPoints.size() && consider; j++)
                    if (pow(imgPoints[i].x - imgPoints[j].x, 2) + pow(imgPoints[i].y - imgPoints[j].y, 2) < 150)
                        consider = false;
        
            if (consider) {
                for (size_t idx = 0; idx < imgPoints.size(); idx++) {
                    if (imgPoints[idx].x > 0 && imgPoints[idx].x < img.cols &&
                        imgPoints[idx].y > 0 && imgPoints[idx].y < img.rows) {
                        std::vector<float> query = {imgPoints[idx].x, imgPoints[idx].y};
                        std::vector<int> indices;
                        std::vector<float> dists;
                        
                        Kdtree.radiusSearch(query, indices, dists, 400.0, 1, cv::flann::SearchParams());
                        
                        int nearestIdx = indices[0];

                        float newDist = cv::norm(cv::Point2f(kpoints[nearestIdx].pt) - cv::Point2f(imgPoints[idx]));
                        
                        // This is my next step, adjusting the distance threshold
                        // -to reach a good performance on different images
                        if (kpoints[nearestIdx].class_id != objKeyPoints[idx].class_id||dists[0] > 320||dists[0] == 0) {
                            continue;
                        }
                        if (nearestIdx != -1) {
                            bool duplicateFound = false;
                            for (size_t i = 0; i < nearestIdxList.size(); ++i) {
                                if (nearestIdxList[i] == nearestIdx) {
                                    duplicateFound = true;
                                    float existingDist = distsList[i];
                                    if (newDist < existingDist) {
                                        p2d[i] = kpoints[nearestIdx].pt;
                                        p3d[i] = cv::Point3f(objPoints[idx].x, objPoints[idx].y, 0);
                                        distsList[i] = newDist; // update distsList
                                    }
                                    break;
                                }
                            }
                        
                            if (!duplicateFound) {
                                nearestIdxList.push_back(nearestIdx);
                                distsList.push_back(newDist);              

                                p2d.push_back(kpoints[nearestIdx].pt);
                                p3d.push_back(cv::Point3f(objPoints[idx].x, objPoints[idx].y, 0));
                            }
                        }
                    }
                }
            } else {
                // If a marker is detected and it is not possible to take all their corners,
                // at least take the external one!
                for (auto markerDetected : detected) {
                    if (markerDetected.id == fm.first) {
                        for (int c = 0; c < 4; c++) {
                            cv::Point2f pt = markerDetected.keypts[c].pt;
                            p3d.push_back(cv::Point3f(pt.x, pt.y, 0));
                            p2d.push_back(markerDetected[c]);
                        }
                        break;
                    }
                }
            }
        }
        // Subpixel refinement
        if(p2d.size() > 0) {
            cv::Size winSize(4, 4);
            cv::Size zeroZone(-1, -1);
            cv::TermCriteria criteria(cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS, 12, 0.005);
            cornerSubPix(bwimage, p2d, winSize, zeroZone, criteria);
        }
    }
    return detected;
}


std::vector<cv::Point2f> FractalMarkerDetector::sort(const std::vector<cv::Point2f>& marker) {
    std::vector<cv::Point2f> res_marker = marker;


    double dx1 = res_marker[1].x - res_marker[0].x;
    double dy1 = res_marker[1].y - res_marker[0].y;
    double dx2 = res_marker[2].x - res_marker[0].x;
    double dy2 = res_marker[2].y - res_marker[0].y;
    double o = (dx1 * dy2) - (dy1 * dx2);

    if (o < 0.0) {
        std::swap(res_marker[1], res_marker[3]);
    }
    return res_marker;
}


float FractalMarkerDetector::getSubpixelValue(const cv::Mat& im_grey, const cv::Point2f& p) {
    float intpartX, intpartY;
    float decpartX = std::modf(p.x, &intpartX);
    float decpartY = std::modf(p.y, &intpartY);

    cv::Point tl;

    if (decpartX>0.5) {
        if (decpartY>0.5) tl=cv::Point(intpartX,intpartY);
        else tl=cv::Point(intpartX,intpartY-1);
    }
    else{
        if (decpartY>0.5) tl=cv::Point(intpartX-1,intpartY);
        else tl=cv::Point(intpartX-1,intpartY-1);
    }
    if(tl.x<0) tl.x=0;
    if(tl.y<0) tl.y=0;
    if(tl.x>=im_grey.cols)tl.x=im_grey.cols-1;
    if(tl.y>=im_grey.cols)tl.y=im_grey.rows-1;
    return (1.f-decpartY)*(1.-decpartX)*float(im_grey.at<uchar>(tl.y,tl.x))+
            decpartX*(1-decpartY)*float(im_grey.at<uchar>(tl.y,tl.x+1))+
            (1-decpartX)*decpartY*float(im_grey.at<uchar>(tl.y+1,tl.x))+
            decpartX*decpartY*float(im_grey.at<uchar>(tl.y+1,tl.x+1));
}


int FractalMarkerDetector::getMarkerId(const cv::Mat& bits, int& nrotations, const std::vector<int>& markersId, const FractalMarkerSet& markerSet) {
    auto rotate = [](const cv::Mat& in) {
        cv::Mat out(in.size(), in.type());
        for (int i = 0; i < in.rows; i++) {
            for (int j = 0; j < in.cols; j++) {
                out.at<uchar>(i, j) = in.at<uchar>(in.cols - j - 1, i);
            }
        }
        return out;
    };


    for (int x = 0; x < bits.cols; x++) {
        if (bits.at<uchar>(0, x) != 0 || bits.at<uchar>(bits.rows - 1, x) != 0 ||
            bits.at<uchar>(x, 0) != 0 || bits.at<uchar>(x, bits.cols - 1) != 0) {
            return -1;
        }
    }


    cv::Mat bit_inner(bits.rows - 2, bits.cols - 2, CV_8UC1);
    for (int r = 0; r < bit_inner.rows; r++) {
        for (int c = 0; c < bit_inner.cols; c++) {
            bit_inner.at<uchar>(r, c) = bits.at<uchar>(r + 1, c + 1);
        }
    }

    nrotations = 0;
    do {
        for (auto idx : markersId) {
            FractalMarker fm = markerSet.fractalMarkerCollection.at(idx);

            cv::Mat masked;
            bit_inner.copyTo(masked, fm.mask());

            if (cv::countNonZero(masked != fm.mat() * 255) == 0) {
                return idx;
            }
        }
        bit_inner = rotate(bit_inner);
        nrotations++;
    } while (nrotations < 4);

    return -1;
}


int FractalMarkerDetector::perimeter(const std::vector<cv::Point2f>& a) {
    int sum = 0;
    for (size_t i = 0; i < a.size(); i++) {
        sum += cv::norm(a[i] - a[(i + 1) % a.size()]);
    }
    return sum;
}

void FractalMarkerDetector::kfilter(std::vector<cv::KeyPoint>& kpoints) {
    float minResp = kpoints[0].response;
    float maxResp = kpoints[0].response;
    for (auto& p : kpoints) {
        p.size = 40;
        if (p.response < minResp) minResp = p.response;
        if (p.response > maxResp) maxResp = p.response;
    }
    float thresoldResp = (maxResp - minResp) * 0.20f + minResp;

    for (uint32_t xi = 0; xi < kpoints.size(); xi++) {
        if (kpoints[xi].response < thresoldResp) {
            kpoints[xi].size = -1;
            continue;
        }

        for (uint32_t xj = xi + 1; xj < kpoints.size(); xj++) {
            if (pow(kpoints[xi].pt.x - kpoints[xj].pt.x, 2) + pow(kpoints[xi].pt.y - kpoints[xj].pt.y, 2) < 100) {
                if (kpoints[xj].response > kpoints[xi].response)
                    kpoints[xi] = kpoints[xj];

                kpoints[xj].size = -1;
            }
        }
    }
    kpoints.erase(std::remove_if(kpoints.begin(), kpoints.end(), [](const cv::KeyPoint& kpt) { return kpt.size == -1; }), kpoints.end());
}

void FractalMarkerDetector::assignClass(const cv::Mat& im, std::vector<cv::KeyPoint>& kpoints, float sizeNorm, int wsize) {
    if (im.type() != CV_8UC1)
        throw std::runtime_error("assignClass Input image must be 8UC1");
    int wsizeFull = wsize * 2 + 1;

    cv::Mat labels = cv::Mat::zeros(wsizeFull, wsizeFull, CV_8UC1);
    cv::Mat thresIm = cv::Mat(wsizeFull, wsizeFull, CV_8UC1);

    for (auto& kp : kpoints) {
        float ptX = kp.pt.x;
        float ptY = kp.pt.y;

        if (sizeNorm > 0) {
            ptX = im.cols * (ptX / sizeNorm + 0.5f);
            ptY = im.rows * (-ptY / sizeNorm + 0.5f);
        }

        int centerX = int(ptX + 0.5f);
        int centerY = int(ptY + 0.5f);

        cv::Rect r = cv::Rect(centerX - wsize, centerY - wsize, wsizeFull, wsizeFull);
        if (r.x < 0 || r.x + r.width > im.cols || r.y < 0 || r.y + r.height > im.rows) continue;

        int endX = r.x + r.width;
        int endY = r.y + r.height;
        uchar minV = 255, maxV = 0;
        for (int row = r.y; row < endY; row++) {
            const uchar* ptr = im.ptr<uchar>(row);
            for (int col = r.x; col < endX; col++) {
                if (minV > ptr[col]) minV = ptr[col];
                if (maxV < ptr[col]) maxV = ptr[col];
            }
        }

        if ((maxV - minV) < 25) {
            kp.class_id = 0;
            continue;
        }

        double thres = (maxV + minV) / 2.0;

        unsigned int nZ = 0;
        for (int row = 0; row < wsizeFull; row++) {
            const uchar* ptr = im.ptr<uchar>(r.y + row) + r.x;
            uchar* thresPtr = thresIm.ptr<uchar>(row);
            for (int col = 0; col < wsizeFull; col++) {
                if (ptr[col] > thres) {
                    nZ++;
                    thresPtr[col] = 255;
                } else
                    thresPtr[col] = 0;
            }
        }

        for (int row = 0; row < thresIm.rows; row++) {
            uchar* labelsPtr = labels.ptr<uchar>(row);
            for (int col = 0; col < thresIm.cols; col++) labelsPtr[col] = 0;
        }

        uchar newLab = 1;
        std::map<uchar, uchar> unions;
        for (int row = 0; row < thresIm.rows; row++) {
            uchar* thresPtr = thresIm.ptr<uchar>(row);
            uchar* labelsPtr = labels.ptr<uchar>(row);
            for (int col = 0; col < thresIm.cols; col++) {
                uchar reg = thresPtr[col];
                uchar lleft_px = 0;
                uchar ltop_px = 0;

                if (col - 1 > -1 && reg == thresPtr[col - 1])
                    lleft_px = labelsPtr[col - 1];

                if (row - 1 > -1 && reg == thresIm.ptr<uchar>(row - 1)[col])
                    ltop_px = labels.at<uchar>(row - 1, col);

                if (lleft_px == 0 && ltop_px == 0)
                    labelsPtr[col] = newLab++;

                else if (lleft_px != 0 && ltop_px != 0) {
                    if (lleft_px < ltop_px) {
                        labelsPtr[col] = lleft_px;
                        unions[ltop_px] = lleft_px;
                    } else if (lleft_px > ltop_px) {
                        labelsPtr[col] = ltop_px;
                        unions[lleft_px] = ltop_px;
                    } else
                        labelsPtr[col] = ltop_px;
                } else if (lleft_px != 0)
                    labelsPtr[col] = lleft_px;
                else
                    labelsPtr[col] = ltop_px;
            }
        }

        int nc = newLab - 1 - unions.size();
        if (nc == 2) {
            if (nZ > thresIm.total() - nZ)
                kp.class_id = 0;
            else
                kp.class_id = 1;
        } else if (nc > 2) {
            kp.class_id = 2;
        }
    }
}

} // namespace aruco
} // namespace cv
