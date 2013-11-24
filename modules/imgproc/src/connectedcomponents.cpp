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
// 2011 Jason Newton <nevion@gmail.com>
//M*/
//
#include "precomp.hpp"
#include <vector>

namespace cv{
    namespace connectedcomponents{

    struct NoOp{
        NoOp(){
        }
        void init(int /*labels*/){
        }
        inline
        void operator()(int r, int c, int l){
            (void) r;
            (void) c;
            (void) l;
        }
        void finish(){}
    };
    struct Point2ui64{
        uint64 x, y;
        Point2ui64(uint64 _x, uint64 _y):x(_x), y(_y){}
    };

    struct CCStatsOp{
        const _OutputArray* _mstatsv;
        cv::Mat statsv;
        const _OutputArray* _mcentroidsv;
        cv::Mat centroidsv;
        std::vector<Point2ui64> integrals;

        CCStatsOp(OutputArray _statsv, OutputArray _centroidsv): _mstatsv(&_statsv), _mcentroidsv(&_centroidsv){
        }
        inline
        void init(int nlabels){
            _mstatsv->create(cv::Size(CC_STAT_MAX, nlabels), cv::DataType<int>::type);
            statsv = _mstatsv->getMat();
            _mcentroidsv->create(cv::Size(2, nlabels), cv::DataType<double>::type);
            centroidsv = _mcentroidsv->getMat();

            for(int l = 0; l < (int) nlabels; ++l){
                int *row = (int *) &statsv.at<int>(l, 0);
                row[CC_STAT_LEFT] = INT_MAX;
                row[CC_STAT_TOP] = INT_MAX;
                row[CC_STAT_WIDTH] = INT_MIN;
                row[CC_STAT_HEIGHT] = INT_MIN;
                row[CC_STAT_AREA] = 0;
            }
            integrals.resize(nlabels, Point2ui64(0, 0));
        }
        void operator()(int r, int c, int l){
            int *row = &statsv.at<int>(l, 0);
            row[CC_STAT_LEFT] = MIN(row[CC_STAT_LEFT], c);
            row[CC_STAT_WIDTH] = MAX(row[CC_STAT_WIDTH], c);
            row[CC_STAT_TOP] = MIN(row[CC_STAT_TOP], r);
            row[CC_STAT_HEIGHT] = MAX(row[CC_STAT_HEIGHT], r);
            row[CC_STAT_AREA]++;
            Point2ui64 &integral = integrals[l];
            integral.x += c;
            integral.y += r;
        }
        void finish(){
            for(int l = 0; l < statsv.rows; ++l){
                int *row = &statsv.at<int>(l, 0);
                row[CC_STAT_WIDTH] = row[CC_STAT_WIDTH] - row[CC_STAT_LEFT] + 1;
                row[CC_STAT_HEIGHT] = row[CC_STAT_HEIGHT] - row[CC_STAT_TOP] + 1;

                Point2ui64 &integral = integrals[l];
                double *centroid = &centroidsv.at<double>(l, 0);
                double area = ((unsigned*)row)[CC_STAT_AREA];
                centroid[0] = double(integral.x) / area;
                centroid[1] = double(integral.y) / area;
            }
        }
    };

    //Find the root of the tree of node i
    template<typename LabelT>
    inline static
    LabelT findRoot(const LabelT *P, LabelT i){
        LabelT root = i;
        while(P[root] < root){
            root = P[root];
        }
        return root;
    }

    //Make all nodes in the path of node i point to root
    template<typename LabelT>
    inline static
    void setRoot(LabelT *P, LabelT i, LabelT root){
        while(P[i] < i){
            LabelT j = P[i];
            P[i] = root;
            i = j;
        }
        P[i] = root;
    }

    //Find the root of the tree of the node i and compress the path in the process
    template<typename LabelT>
    inline static
    LabelT find(LabelT *P, LabelT i){
        LabelT root = findRoot(P, i);
        setRoot(P, i, root);
        return root;
    }

    //unite the two trees containing nodes i and j and return the new root
    template<typename LabelT>
    inline static
    LabelT set_union(LabelT *P, LabelT i, LabelT j){
        LabelT root = findRoot(P, i);
        if(i != j){
            LabelT rootj = findRoot(P, j);
            if(root > rootj){
                root = rootj;
            }
            setRoot(P, j, root);
        }
        setRoot(P, i, root);
        return root;
    }

    //Flatten the Union Find tree and relabel the components
    template<typename LabelT>
    inline static
    LabelT flattenL(LabelT *P, LabelT length){
        LabelT k = 1;
        for(LabelT i = 1; i < length; ++i){
            if(P[i] < i){
                P[i] = P[P[i]];
            }else{
                P[i] = k; k = k + 1;
            }
        }
        return k;
    }

    //Based on "Two Strategies to Speed up Connected Components Algorithms", the SAUF (Scan array union find) variant
    //using decision trees
    //Kesheng Wu, et al
    //Note: rows are encoded as position in the "rows" array to save lookup times
    //reference for 4-way: {{-1, 0}, {0, -1}};//b, d neighborhoods
    const int G4[2][2] = {{1, 0}, {0, -1}};//b, d neighborhoods
    //reference for 8-way: {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}};//a, b, c, d neighborhoods
    const int G8[4][2] = {{1, -1}, {1, 0}, {1, 1}, {0, -1}};//a, b, c, d neighborhoods
    template<typename LabelT, typename PixelT, typename StatsOp = NoOp >
    struct LabelingImpl{
    LabelT operator()(const cv::Mat &I, cv::Mat &L, int connectivity, StatsOp &sop){
        CV_Assert(L.rows == I.rows);
        CV_Assert(L.cols == I.cols);
        CV_Assert(connectivity == 8 || connectivity == 4);
        const int rows = L.rows;
        const int cols = L.cols;
        size_t Plength = (size_t(rows + 3 - 1)/3) * (size_t(cols + 3 - 1)/3);
        if(connectivity == 4){
            Plength = 4 * Plength;//a quick and dirty upper bound, an exact answer exists if you want to find it
            //the 4 comes from the fact that a 3x3 block can never have more than 4 unique labels
        }
        LabelT *P = (LabelT *) fastMalloc(sizeof(LabelT) * Plength);
        P[0] = 0;
        LabelT lunique = 1;
        //scanning phase
        for(int r_i = 0; r_i < rows; ++r_i){
            LabelT *Lrow = (LabelT *)(L.data + L.step.p[0] * r_i);
            LabelT *Lrow_prev = (LabelT *)(((char *)Lrow) - L.step.p[0]);
            const PixelT *Irow = (PixelT *)(I.data + I.step.p[0] * r_i);
            const PixelT *Irow_prev = (const PixelT *)(((char *)Irow) - I.step.p[0]);
            LabelT *Lrows[2] = {
                Lrow,
                Lrow_prev
            };
            const PixelT *Irows[2] = {
                Irow,
                Irow_prev
            };
            if(connectivity == 8){
                const int a = 0;
                const int b = 1;
                const int c = 2;
                const int d = 3;
                const bool T_a_r = (r_i - G8[a][0]) >= 0;
                const bool T_b_r = (r_i - G8[b][0]) >= 0;
                const bool T_c_r = (r_i - G8[c][0]) >= 0;
                for(int c_i = 0; Irows[0] != Irow + cols; ++Irows[0], c_i++){
                    if(!*Irows[0]){
                        Lrow[c_i] = 0;
                        continue;
                    }
                    Irows[1] = Irow_prev + c_i;
                    Lrows[0] = Lrow + c_i;
                    Lrows[1] = Lrow_prev + c_i;
                    const bool T_a = T_a_r && (c_i + G8[a][1]) >= 0   && *(Irows[G8[a][0]] + G8[a][1]);
                    const bool T_b = T_b_r                            && *(Irows[G8[b][0]] + G8[b][1]);
                    const bool T_c = T_c_r && (c_i + G8[c][1]) < cols && *(Irows[G8[c][0]] + G8[c][1]);
                    const bool T_d =          (c_i + G8[d][1]) >= 0   && *(Irows[G8[d][0]] + G8[d][1]);

                    //decision tree
                    if(T_b){
                        //copy(b)
                        *Lrows[0] = *(Lrows[G8[b][0]] + G8[b][1]);
                    }else{//not b
                        if(T_c){
                            if(T_a){
                                //copy(c, a)
                                *Lrows[0] = set_union(P, *(Lrows[G8[c][0]] + G8[c][1]), *(Lrows[G8[a][0]] + G8[a][1]));
                            }else{
                                if(T_d){
                                    //copy(c, d)
                                    *Lrows[0] = set_union(P, *(Lrows[G8[c][0]] + G8[c][1]), *(Lrows[G8[d][0]] + G8[d][1]));
                                }else{
                                    //copy(c)
                                    *Lrows[0] = *(Lrows[G8[c][0]] + G8[c][1]);
                                }
                            }
                        }else{//not c
                            if(T_a){
                                //copy(a)
                                *Lrows[0] = *(Lrows[G8[a][0]] + G8[a][1]);
                            }else{
                                if(T_d){
                                    //copy(d)
                                    *Lrows[0] = *(Lrows[G8[d][0]] + G8[d][1]);
                                }else{
                                    //new label
                                    *Lrows[0] = lunique;
                                    P[lunique] = lunique;
                                    lunique = lunique + 1;
                                }
                            }
                        }
                    }
                }
            }else{
                //B & D only
                const int b = 0;
                const int d = 1;
                const bool T_b_r = (r_i - G4[b][0]) >= 0;
                for(int c_i = 0; Irows[0] != Irow + cols; ++Irows[0], c_i++){
                    if(!*Irows[0]){
                        Lrow[c_i] = 0;
                        continue;
                    }
                    Irows[1] = Irow_prev + c_i;
                    Lrows[0] = Lrow + c_i;
                    Lrows[1] = Lrow_prev + c_i;
                    const bool T_b = T_b_r                            && *(Irows[G4[b][0]] + G4[b][1]);
                    const bool T_d =          (c_i + G4[d][1]) >= 0   && *(Irows[G4[d][0]] + G4[d][1]);
                    if(T_b){
                        if(T_d){
                            //copy(d, b)
                            *Lrows[0] = set_union(P, *(Lrows[G4[d][0]] + G4[d][1]), *(Lrows[G4[b][0]] + G4[b][1]));
                        }else{
                            //copy(b)
                            *Lrows[0] = *(Lrows[G4[b][0]] + G4[b][1]);
                        }
                    }else{
                        if(T_d){
                            //copy(d)
                            *Lrows[0] = *(Lrows[G4[d][0]] + G4[d][1]);
                        }else{
                            //new label
                            *Lrows[0] = lunique;
                            P[lunique] = lunique;
                            lunique = lunique + 1;
                        }
                    }
                }
            }
        }

        //analysis
        LabelT nLabels = flattenL(P, lunique);
        sop.init(nLabels);

        for(int r_i = 0; r_i < rows; ++r_i){
            LabelT *Lrow_start = (LabelT *)(L.data + L.step.p[0] * r_i);
            LabelT *Lrow_end = Lrow_start + cols;
            LabelT *Lrow = Lrow_start;
            for(int c_i = 0; Lrow != Lrow_end; ++Lrow, ++c_i){
                const LabelT l = P[*Lrow];
                *Lrow = l;
                sop(r_i, c_i, l);
            }
        }

        sop.finish();
        fastFree(P);

        return nLabels;
    }//End function LabelingImpl operator()

    };//End struct LabelingImpl
}//end namespace connectedcomponents

//L's type must have an appropriate depth for the number of pixels in I
template<typename StatsOp>
static
int connectedComponents_sub1(const cv::Mat &I, cv::Mat &L, int connectivity, StatsOp &sop){
    CV_Assert(L.channels() == 1 && I.channels() == 1);
    CV_Assert(connectivity == 8 || connectivity == 4);

    int lDepth = L.depth();
    int iDepth = I.depth();
    using connectedcomponents::LabelingImpl;
    //warn if L's depth is not sufficient?

    CV_Assert(iDepth == CV_8U || iDepth == CV_8S);

    if(lDepth == CV_8U){
        return (int) LabelingImpl<uchar, uchar, StatsOp>()(I, L, connectivity, sop);
    }else if(lDepth == CV_16U){
        return (int) LabelingImpl<ushort, uchar, StatsOp>()(I, L, connectivity, sop);
    }else if(lDepth == CV_32S){
        //note that signed types don't really make sense here and not being able to use unsigned matters for scientific projects
        //OpenCV: how should we proceed?  .at<T> typechecks in debug mode
        return (int) LabelingImpl<int, uchar, StatsOp>()(I, L, connectivity, sop);
    }

    CV_Error(CV_StsUnsupportedFormat, "unsupported label/image type");
    return -1;
}

}

int cv::connectedComponents(InputArray _img, OutputArray _labels, int connectivity, int ltype){
    const cv::Mat img = _img.getMat();
    _labels.create(img.size(), CV_MAT_DEPTH(ltype));
    cv::Mat labels = _labels.getMat();
    connectedcomponents::NoOp sop;
    if(ltype == CV_16U){
        return connectedComponents_sub1(img, labels, connectivity, sop);
    }else if(ltype == CV_32S){
        return connectedComponents_sub1(img, labels, connectivity, sop);
    }else{
        CV_Error(CV_StsUnsupportedFormat, "the type of labels must be 16u or 32s");
        return 0;
    }
}

int cv::connectedComponentsWithStats(InputArray _img, OutputArray _labels, OutputArray statsv,
                                     OutputArray centroids, int connectivity, int ltype)
{
    const cv::Mat img = _img.getMat();
    _labels.create(img.size(), CV_MAT_DEPTH(ltype));
    cv::Mat labels = _labels.getMat();
    connectedcomponents::CCStatsOp sop(statsv, centroids);
    if(ltype == CV_16U){
        return connectedComponents_sub1(img, labels, connectivity, sop);
    }else if(ltype == CV_32S){
        return connectedComponents_sub1(img, labels, connectivity, sop);
    }else{
        CV_Error(CV_StsUnsupportedFormat, "the type of labels must be 16u or 32s");
        return 0;
    }
}
