#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <map>

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
using namespace std;

static bool readCameraMatrix(const string& filename,
                             Mat& cameraMatrix, Mat& distCoeffs,
                             Size& calibratedImageSize )
{
    FileStorage fs(filename, FileStorage::READ);
    fs["image_width"] >> calibratedImageSize.width;
    fs["image_height"] >> calibratedImageSize.height;
    fs["distortion_coefficients"] >> distCoeffs;
    fs["camera_matrix"] >> cameraMatrix;
    
    if( distCoeffs.type() != CV_64F )
        distCoeffs = Mat_<double>(distCoeffs);
    if( cameraMatrix.type() != CV_64F )
        cameraMatrix = Mat_<double>(cameraMatrix);
    
    return true;
}

static bool readModelViews( const string& filename, vector<Point3f>& box,
                           vector<string>& imagelist,
                           vector<Rect>& roiList, vector<Vec6f>& poseList )
{
    imagelist.resize(0);
    roiList.resize(0);
    poseList.resize(0);
    box.resize(0);
    
    FileStorage fs(filename, FileStorage::READ);
    if( !fs.isOpened() )
        return false;
    fs["box"] >> box;
    
    FileNode all = fs["views"];
    if( all.type() != FileNode::SEQ )
        return false;
    FileNodeIterator it = all.begin(), it_end = all.end();
    
    for(; it != it_end; ++it)
    {
        FileNode n = *it;
        imagelist.push_back((string)n["image"]);
        FileNode nr = n["roi"];
        roiList.push_back(Rect((int)nr[0], (int)nr[1], (int)nr[2], (int)nr[3]));
        FileNode np = n["pose"];
        poseList.push_back(Vec6f((float)np[0], (float)np[1], (float)np[2],
                                 (float)np[3], (float)np[4], (float)np[5]));
    }
    
    return true;
}


struct PointModel
{
    vector<Point3f> points;
    vector<vector<int> > didx;
    Mat descriptors;
    string name;
};


static void writeModel(const string& modelFileName, const string& modelname,
                       const PointModel& model)
{
    FileStorage fs(modelFileName, FileStorage::WRITE);
    
    fs << modelname << "{" <<
        "points" << "[:" << model.points << "]" <<
        "idx" << "[:";
    
    for( size_t i = 0; i < model.didx.size(); i++ )
        fs << "[:" << model.didx[i] << "]";
    fs << "]" << "descriptors" << model.descriptors;
}


static void unpackPose(const Vec6f& pose, Mat& R, Mat& t)
{
    Mat rvec = (Mat_<double>(3,1) << pose[0], pose[1], pose[2]);
    t = (Mat_<double>(3,1) << pose[3], pose[4], pose[5]);
    Rodrigues(rvec, R);
}


static Mat getFundamentalMat( const Mat& R1, const Mat& t1,
                              const Mat& R2, const Mat& t2,
                              const Mat& cameraMatrix )
{
    Mat_<double> R = R2*R1.t(), t = t2 - R*t1;
    double tx = t.at<double>(0,0), ty = t.at<double>(1,0), tz = t.at<double>(2,0);
    Mat E = (Mat_<double>(3,3) << 0, -tz, ty, tz, 0, -tx, -ty, tx, 0)*R;
    Mat iK = cameraMatrix.inv();
    Mat F = iK.t()*E*iK;

#if 0
    static bool checked = false;
    if(!checked)
    {
        vector<Point3f> objpoints(100);
        Mat O(objpoints);
        randu(O, Scalar::all(-10), Scalar::all(10));
        vector<Point2f> imgpoints1, imgpoints2;
        projectPoints(Mat(objpoints), R1, t1, cameraMatrix, Mat(), imgpoints1);
        projectPoints(Mat(objpoints), R2, t2, cameraMatrix, Mat(), imgpoints2);
        double* f = (double*)F.data;
        for( size_t i = 0; i < objpoints.size(); i++ )
        {
            Point2f p1 = imgpoints1[i], p2 = imgpoints2[i];
            double diff = p2.x*(f[0]*p1.x + f[1]*p1.y + f[2]) +
                 p2.y*(f[3]*p1.x + f[4]*p1.y + f[5]) +
                f[6]*p1.x + f[7]*p1.y + f[8];
            CV_Assert(fabs(diff) < 1e-3);
        }
        checked = true;
    }
#endif
    return F;
}


static void findConstrainedCorrespondences(const Mat& _F,
                const vector<KeyPoint>& keypoints1,
                const vector<KeyPoint>& keypoints2,
                const Mat& descriptors1,
                const Mat& descriptors2,
                vector<Vec2i>& matches,
                double eps, double ratio)
{
    float F[9]={0};
    int dsize = descriptors1.cols;
    
    Mat Fhdr = Mat(3, 3, CV_32F, F);
    _F.convertTo(Fhdr, CV_32F);
    matches.clear();
    
    for( size_t i = 0; i < keypoints1.size(); i++ )
    {
        Point2f p1 = keypoints1[i].pt;
        double bestDist1 = DBL_MAX, bestDist2 = DBL_MAX;
        int bestIdx1 = -1, bestIdx2 = -1;
        const float* d1 = descriptors1.ptr<float>(i);
        
        for( size_t j = 0; j < keypoints2.size(); j++ )
        {
            Point2f p2 = keypoints2[j].pt;
            double e = p2.x*(F[0]*p1.x + F[1]*p1.y + F[2]) +
                       p2.y*(F[3]*p1.x + F[4]*p1.y + F[5]) +
                       F[6]*p1.x + F[7]*p1.y + F[8];
            if( fabs(e) > eps )
                continue;
            const float* d2 = descriptors2.ptr<float>(j);
            double dist = 0;
            int k = 0;
            
            for( ; k <= dsize - 8; k += 8 )
            {
                float t0 = d1[k] - d2[k], t1 = d1[k+1] - d2[k+1];
                float t2 = d1[k+2] - d2[k+2], t3 = d1[k+3] - d2[k+3];
                float t4 = d1[k+4] - d2[k+4], t5 = d1[k+5] - d2[k+5];
                float t6 = d1[k+6] - d2[k+6], t7 = d1[k+7] - d2[k+7];
                dist += t0*t0 + t1*t1 + t2*t2 + t3*t3 +
                        t4*t4 + t5*t5 + t6*t6 + t7*t7;
                
                if( dist >= bestDist2 )
                    break;
            }
            
            if( dist < bestDist2 )
            {
                for( ; k < dsize; k++ )
                {
                    float t = d1[k] - d2[k];
                    dist += t*t;
                }
                
                if( dist < bestDist1 )
                {
                    bestDist2 = bestDist1;
                    bestIdx2 = bestIdx1;
                    bestDist1 = dist;
                    bestIdx1 = (int)j;
                }
                else if( dist < bestDist2 )
                {
                    bestDist2 = dist;
                    bestIdx2 = (int)j;
                }
            }
        }
        
        if( bestIdx1 >= 0 && bestDist1 < bestDist2*ratio )
        {
            Point2f p2 = keypoints1[bestIdx1].pt;
            double e = p2.x*(F[0]*p1.x + F[1]*p1.y + F[2]) +
                        p2.y*(F[3]*p1.x + F[4]*p1.y + F[5]) +
                        F[6]*p1.x + F[7]*p1.y + F[8];
            if( e > eps*0.25 )
                continue;
            double threshold = bestDist1/ratio;
            const float* d22 = descriptors2.ptr<float>(bestIdx1);
            size_t i1 = 0;
            for( ; i1 < keypoints1.size(); i1++ )
            {
                if( i1 == i )
                    continue;
                Point2f p1 = keypoints1[i1].pt;
                const float* d11 = descriptors1.ptr<float>(i1);
                double dist = 0;
                
                e = p2.x*(F[0]*p1.x + F[1]*p1.y + F[2]) +
                    p2.y*(F[3]*p1.x + F[4]*p1.y + F[5]) +
                    F[6]*p1.x + F[7]*p1.y + F[8];
                if( fabs(e) > eps )
                    continue;
                
                for( int k = 0; k < dsize; k++ )
                {
                    float t = d11[k] - d22[k];
                    dist += t*t;
                    if( dist >= threshold )
                        break;
                }
                
                if( dist < threshold )
                    break;
            }
            if( i1 == keypoints1.size() )
                matches.push_back(Vec2i(i,bestIdx1));
        }
    }
}


static Point3f findRayIntersection(Point3f k1, Point3f b1, Point3f k2, Point3f b2)
{    
    float a[4], b[2], x[2];
    a[0] = k1.dot(k1);
    a[1] = a[2] = -k1.dot(k2);
    a[3] = k2.dot(k2);
    b[0] = k1.dot(b2 - b1);
    b[1] = k2.dot(b1 - b2);
    Mat_<float> A(2, 2, a), B(2, 1, b), X(2, 1, x);
    solve(A, B, X);
    
    float s1 = X.at<float>(0, 0);
    float s2 = X.at<float>(1, 0);
    return (k1*s1 + b1 + k2*s2 + b2)*0.5f;
}


static Point3f triangulatePoint(const vector<Point2f>& ps,
                                const vector<Mat>& Rs,
                                const vector<Mat>& ts,
                                const Mat& cameraMatrix)
{
    Mat_<double> K(cameraMatrix);
    
    if( ps.size() > 2 )
    {
        Mat_<double> L(ps.size()*3, 4), U, evalues;
        Mat_<double> P(3,4), Rt(3,4), Rt_part1=Rt.colRange(0,3), Rt_part2=Rt.colRange(3,4);
        
        for( size_t i = 0; i < ps.size(); i++ )
        {
            double x = ps[i].x, y = ps[i].y;
            Rs[i].convertTo(Rt_part1, Rt_part1.type());
            ts[i].convertTo(Rt_part2, Rt_part2.type());
            P = K*Rt;
            
            for( int k = 0; k < 4; k++ )
            {
                L(i*3, k) = x*P(2,k) - P(0,k);
                L(i*3+1, k) = y*P(2,k) - P(1,k);
                L(i*3+2, k) = x*P(1,k) - y*P(0,k);
            }
        }
        
        eigen(L.t()*L, evalues, U);
        CV_Assert(evalues(0,0) >= evalues(3,0));
        
        double W = fabs(U(3,3)) > FLT_EPSILON ? 1./U(3,3) : 0;
        return Point3f((float)(U(3,0)*W), (float)(U(3,1)*W), (float)(U(3,2)*W));
    }
    else
    {
        Mat_<float> iK = K.inv();
        Mat_<float> R1t = Mat_<float>(Rs[0]).t();
        Mat_<float> R2t = Mat_<float>(Rs[1]).t();
        Mat_<float> m1 = (Mat_<float>(3,1) << ps[0].x, ps[0].y, 1);
        Mat_<float> m2 = (Mat_<float>(3,1) << ps[1].x, ps[1].y, 1);
        Mat_<float> K1 = R1t*(iK*m1), K2 = R2t*(iK*m2);
        Mat_<float> B1 = -R1t*Mat_<float>(ts[0]);
        Mat_<float> B2 = -R2t*Mat_<float>(ts[1]);
        return findRayIntersection(*K1.ptr<Point3f>(), *B1.ptr<Point3f>(),
                                   *K2.ptr<Point3f>(), *B2.ptr<Point3f>());
    }
}


void triangulatePoint_test(void)
{
    int i, n = 100;
    vector<Point3f> objpt(n), delta1(n), delta2(n);
    Mat rvec1(3,1,CV_32F), tvec1(3,1,CV_64F);
    Mat rvec2(3,1,CV_32F), tvec2(3,1,CV_64F);
    Mat objptmat(objpt), deltamat1(delta1), deltamat2(delta2);
    randu(rvec1, Scalar::all(-10), Scalar::all(10));
    randu(tvec1, Scalar::all(-10), Scalar::all(10));
    randu(rvec2, Scalar::all(-10), Scalar::all(10));
    randu(tvec2, Scalar::all(-10), Scalar::all(10));
    
    randu(objptmat, Scalar::all(-10), Scalar::all(10));
    double eps = 1e-2;
    randu(deltamat1, Scalar::all(-eps), Scalar::all(eps));
    randu(deltamat2, Scalar::all(-eps), Scalar::all(eps));
    vector<Point2f> imgpt1, imgpt2;
    Mat_<float> cameraMatrix(3,3);
    double fx = 1000., fy = 1010., cx = 400.5, cy = 300.5;
    cameraMatrix << fx, 0, cx, 0, fy, cy, 0, 0, 1;
    
    projectPoints(Mat(objpt)+Mat(delta1), rvec1, tvec1, cameraMatrix, Mat(), imgpt1);
    projectPoints(Mat(objpt)+Mat(delta2), rvec2, tvec2, cameraMatrix, Mat(), imgpt2);
    
    vector<Point3f> objptt(n);
    vector<Point2f> pts(2);
    vector<Mat> Rv(2), tv(2);
    Rodrigues(rvec1, Rv[0]);
    Rodrigues(rvec2, Rv[1]);
    tv[0] = tvec1; tv[1] = tvec2;
    for( i = 0; i < n; i++ )
    {
        pts[0] = imgpt1[i]; pts[1] = imgpt2[i];
        objptt[i] = triangulatePoint(pts, Rv, tv, cameraMatrix);
    }
    double err = norm(Mat(objpt), Mat(objptt), CV_C);
    CV_Assert(err < 1e-1);
}

typedef pair<int, int> Pair2i;
typedef map<Pair2i, int> Set2i;

struct EqKeypoints
{
    EqKeypoints(const vector<int>* _dstart, const Set2i* _pairs)
    : dstart(_dstart), pairs(_pairs) {}
    
    bool operator()(const Pair2i& a, const Pair2i& b) const
    {
        return pairs->find(Pair2i(dstart->at(a.first) + a.second,
                                  dstart->at(b.first) + b.second)) != pairs->end();
    }
    
    const vector<int>* dstart;
    const Set2i* pairs;
};

static void build3dmodel( const Ptr<FeatureDetector>& detector,
                          const Ptr<DescriptorExtractor>& descriptorExtractor,
                          const vector<Point3f>& modelBox,
                          const vector<string>& imageList,
                          const vector<Rect>& roiList,
                          const vector<Vec6f>& poseList,
                          const Mat& cameraMatrix,
                          PointModel& model )
{
    int progressBarSize = 10;
    
    const double Feps = 5;
    const double DescriptorRatio = 0.7;
    
    vector<vector<KeyPoint> > allkeypoints;
    vector<int> dstart;
    vector<float> alldescriptorsVec;
    vector<Vec2i> pairwiseMatches;
    vector<Mat> Rs, ts;
    int descriptorSize = 0;
    Mat descriptorbuf;
    Set2i pairs, keypointsIdxMap;
    
    model.points.clear();
    model.didx.clear();
    
    dstart.push_back(0);
    
    size_t nimages = imageList.size();
    size_t nimagePairs = (nimages - 1)*nimages/2 - nimages;
    
    printf("\nComputing descriptors ");
    
    // 1. find all the keypoints and all the descriptors
    for( size_t i = 0; i < nimages; i++ )
    {
        Mat img = imread(imageList[i], 1), gray;
        cvtColor(img, gray, CV_BGR2GRAY);
        
        vector<KeyPoint> keypoints;
        detector->detect(gray, keypoints);
        descriptorExtractor->compute(gray, keypoints, descriptorbuf);
        Point2f roiofs = roiList[i].tl(); 
        for( size_t k = 0; k < keypoints.size(); k++ )
            keypoints[k].pt += roiofs;
        allkeypoints.push_back(keypoints);
        
        Mat buf = descriptorbuf;
        if( !buf.isContinuous() || buf.type() != CV_32F )
        {
            buf.release();
            descriptorbuf.convertTo(buf, CV_32F);
        }
        descriptorSize = buf.cols;
        
        size_t prev = alldescriptorsVec.size();
        size_t delta = buf.rows*buf.cols;
        alldescriptorsVec.resize(prev + delta); 
        std::copy(buf.ptr<float>(), buf.ptr<float>() + delta,
                  alldescriptorsVec.begin() + prev);
        dstart.push_back(dstart.back() + keypoints.size());
        
        Mat R, t;
        unpackPose(poseList[i], R, t);
        Rs.push_back(R);
        ts.push_back(t);
        
        if( (i+1)*progressBarSize/nimages > i*progressBarSize/nimages )
        {
            putchar('.');
            fflush(stdout);
        }
    }
    
    Mat alldescriptors(alldescriptorsVec.size()/descriptorSize, descriptorSize, CV_32F,
                       &alldescriptorsVec[0]);
    
    printf("\nOk. total images = %d. total keypoints = %d\n",
           (int)nimages, alldescriptors.rows);

    printf("\nFinding correspondences ");
    
    int pairsFound = 0;
    
    vector<Point2f> pts_k(2);
    vector<Mat> Rs_k(2), ts_k(2);
    //namedWindow("img1", 1);
    //namedWindow("img2", 1);
    
    // 2. find pairwise correspondences
    for( size_t i = 0; i < nimages; i++ )
        for( size_t j = i+1; j < nimages; j++ )
        {
            const vector<KeyPoint>& keypoints1 = allkeypoints[i];
            const vector<KeyPoint>& keypoints2 = allkeypoints[j];
            Mat descriptors1 = alldescriptors.rowRange(dstart[i], dstart[i+1]);
            Mat descriptors2 = alldescriptors.rowRange(dstart[j], dstart[j+1]);
            
            Mat F = getFundamentalMat(Rs[i], ts[i], Rs[j], ts[j], cameraMatrix);
            
            findConstrainedCorrespondences( F, keypoints1, keypoints2,
                                            descriptors1, descriptors2,
                                            pairwiseMatches, Feps, DescriptorRatio );
            
            //pairsFound += (int)pairwiseMatches.size();
            
            //Mat img1 = imread(format("%s/frame%04d.jpg", model.name.c_str(), (int)i), 1);
            //Mat img2 = imread(format("%s/frame%04d.jpg", model.name.c_str(), (int)j), 1);
            
            //double avg_err = 0;
            for( size_t k = 0; k < pairwiseMatches.size(); k++ )
            {
                int i1 = pairwiseMatches[k][0], i2 = pairwiseMatches[k][1];
                
                pts_k[0] = keypoints1[i1].pt;
                pts_k[1] = keypoints2[i2].pt;
                Rs_k[0] = Rs[i]; Rs_k[1] = Rs[j];
                ts_k[0] = ts[i]; ts_k[1] = ts[j];
                Point3f objpt = triangulatePoint(pts_k, Rs_k, ts_k, cameraMatrix);
                
                vector<Point3f> objpts;
                objpts.push_back(objpt);
                vector<Point2f> imgpts1, imgpts2;
                projectPoints(Mat(objpts), Rs_k[0], ts_k[0], cameraMatrix, Mat(), imgpts1);
                projectPoints(Mat(objpts), Rs_k[1], ts_k[1], cameraMatrix, Mat(), imgpts2);
                
                double e1 = norm(imgpts1[0] - keypoints1[i1].pt);
                double e2 = norm(imgpts2[0] - keypoints2[i2].pt);
                if( e1 + e2 > 10 )
                    continue;
                
                pairsFound++;
                //pts_k[0] = imgpts1[0];
                //pts_k[1] = imgpts2[0];
                //objpt = triangulatePoint(pts_k, Rs_k, ts_k, cameraMatrix);
                //objpts[0] = objpt;
                //projectPoints(Mat(objpts), Rs_k[0], ts_k[0], cameraMatrix, Mat(), imgpts1);
                //projectPoints(Mat(objpts), Rs_k[1], ts_k[1], cameraMatrix, Mat(), imgpts2);
                //double e1 = norm(imgpts1[0] - keypoints1[i1].pt);
                //double e2 = norm(imgpts2[0] - keypoints2[i2].pt);
                
                //model.points.push_back(objpt);   
                pairs[Pair2i(i1+dstart[i], i2+dstart[j])] = 1;
                pairs[Pair2i(i2+dstart[j], i1+dstart[i])] = 1;
                keypointsIdxMap[Pair2i(i,i1)] = 1;
                keypointsIdxMap[Pair2i(j,i2)] = 1;
                //CV_Assert(e1 < 5 && e2 < 5);
                //Scalar color(rand()%256,rand()%256, rand()%256);
                //circle(img1, keypoints1[i1].pt, 2, color, -1, CV_AA);
                //circle(img2, keypoints2[i2].pt, 2, color, -1, CV_AA);
            }
            //printf("avg err = %g\n", pairwiseMatches.size() ? avg_err/(2*pairwiseMatches.size()) : 0.);
            //imshow("img1", img1);
            //imshow("img2", img2);
            //waitKey();
            
            if( (i+1)*progressBarSize/nimagePairs > i*progressBarSize/nimagePairs )
            {
                putchar('.');
                fflush(stdout);
            }
        }
    
    printf("\nOk. Total pairs = %d\n", pairsFound );
    
    // 3. build the keypoint clusters
    vector<Pair2i> keypointsIdx;
    Set2i::iterator kpidx_it = keypointsIdxMap.begin(), kpidx_end = keypointsIdxMap.end();
    
    for( ; kpidx_it != kpidx_end; ++kpidx_it )
        keypointsIdx.push_back(kpidx_it->first);

    printf("\nClustering correspondences ");
    
    vector<int> labels;
    int nclasses = partition( keypointsIdx, labels, EqKeypoints(&dstart, &pairs) );

    printf("\nOk. Total classes (i.e. 3d points) = %d\n", nclasses );
    
    model.descriptors.create(keypointsIdx.size(), descriptorSize, CV_32F);
    model.didx.resize(nclasses);
    model.points.resize(nclasses);
    
    vector<vector<Pair2i> > clusters(nclasses);
    for( size_t i = 0; i < keypointsIdx.size(); i++ )
        clusters[labels[i]].push_back(keypointsIdx[i]);
    
    // 4. now compute 3D points corresponding to each cluster and fill in the model data
    printf("\nComputing 3D coordinates ");
    
    int globalDIdx = 0;
    for( int k = 0; k < nclasses; k++ )
    {
        int i, n = (int)clusters[k].size();
        pts_k.resize(n);
        Rs_k.resize(n);
        ts_k.resize(n);
        model.didx[k].resize(n);
        for( i = 0; i < n; i++ )
        {
            int imgidx = clusters[k][i].first, ptidx = clusters[k][i].second;
            Mat dstrow = model.descriptors.row(globalDIdx);
            alldescriptors.row(dstart[imgidx] + ptidx).copyTo(dstrow);
            
            model.didx[k][i] = globalDIdx++;
            pts_k[i] = allkeypoints[imgidx][ptidx].pt;
            Rs_k[i] = Rs[imgidx];
            ts_k[i] = ts[imgidx];
        }
        Point3f objpt = triangulatePoint(pts_k, Rs_k, ts_k, cameraMatrix);
        model.points[k] = objpt;
        
        if( (i+1)*progressBarSize/nclasses > i*progressBarSize/nclasses )
        {
            putchar('.');
            fflush(stdout);
        }
    }

    Mat img(768, 1024, CV_8UC3);
    vector<Point2f> imagePoints;
    namedWindow("Test", 1);

    // visualize the cloud
    for( size_t i = 0; i < nimages; i++ )
    {
        img = imread(format("%s/frame%04d.jpg", model.name.c_str(), (int)i), 1);
        projectPoints(Mat(model.points), Rs[i], ts[i], cameraMatrix, Mat(), imagePoints);
        
        for( int k = 0; k < (int)imagePoints.size(); k++ )
            circle(img, imagePoints[k], 2, Scalar(0,255,0), -1, CV_AA, 0);
        
        imshow("Test", img);
        int c = waitKey();
        if( c == 'q' || c == 'Q' )
            break;
    }
}


int main(int argc, char** argv)
{
    triangulatePoint_test();
    
    const char* help = "Usage: build3dmodel -i <intrinsics_filename>\n"
        "\t[-d <detector>] [-de <descriptor_extractor>] -m <model_name>\n";
    
    if(argc < 3)
    {
        puts(help);
        return 0;
    }
    const char* intrinsicsFilename = 0;
	const char* modelName = 0;
    const char* detectorName = "SURF";
    const char* descriptorExtractorName = "SURF"; 
    vector<Point3f> modelBox;
    vector<string> imageList;
    vector<Rect> roiList;
    vector<Vec6f> poseList;
    
    for( int i = 1; i < argc; i++ )
    {
        if( strcmp(argv[i], "-i") == 0 )
			intrinsicsFilename = argv[++i];
		else if( strcmp(argv[i], "-m") == 0 )
			modelName = argv[++i];
		else if( strcmp(argv[i], "-d") == 0 )
            detectorName = argv[++i];
        else if( strcmp(argv[i], "-de") == 0 )
            descriptorExtractorName = argv[++i];
		else
		{
			printf("Incorrect option\n");
			puts(help);
			return 0;
		}
    }
    
	if( !intrinsicsFilename || !modelName )
	{
		printf("Some of the required parameters are missing\n");
		puts(help);
		return 0;
	}

    Mat cameraMatrix, distCoeffs;
    Size calibratedImageSize;
    readCameraMatrix(intrinsicsFilename, cameraMatrix, distCoeffs, calibratedImageSize);
    
    Ptr<FeatureDetector> detector = createDetector(detectorName);
    Ptr<DescriptorExtractor> descriptorExtractor = createDescriptorExtractor(descriptorExtractorName);
    
    string modelIndexFilename = format("%s_segm/frame_index.yml", modelName);
    if(!readModelViews( modelIndexFilename, modelBox, imageList, roiList, poseList))
    {
        printf("Can not read the model. Check the parameters and the working directory\n");
        puts(help);
		return 0;
	}
    
    PointModel model;
    model.name = modelName;
    build3dmodel( detector, descriptorExtractor, modelBox,
                  imageList, roiList, poseList, cameraMatrix, model );
    string outputModelName = format("%s_model.yml.gz", modelName);
    
    
    printf("\nDone! Now saving the model ...\n");
    writeModel(outputModelName, modelName, model);
    
    return 0;
}
