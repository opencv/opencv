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

    template<typename LabelT>
    struct NoOp{
        NoOp(){
        }
        void init(const LabelT labels){
            (void) labels;
        }
        inline
        void operator()(int r, int c, LabelT l){
            (void) r;
            (void) c;
            (void) l;
        }
        void finish(){}
    };
    template<typename LabelT>
    struct CCStatsOp{
        std::vector<cv::ConnectedComponentStats> &statsv;
        CCStatsOp(std::vector<cv::ConnectedComponentStats> &_statsv): statsv(_statsv){
        }
        inline
        void init(const LabelT nlabels){
            statsv.clear();
            cv::ConnectedComponentStats stats = cv::ConnectedComponentStats();
            stats.lower_x = std::numeric_limits<LabelT>::max();
            stats.lower_y = std::numeric_limits<LabelT>::max();
            stats.upper_x = std::numeric_limits<LabelT>::min();
            stats.upper_y = std::numeric_limits<LabelT>::min();
            stats.centroid_x = 0;
            stats.centroid_y = 0;
            stats.integral_x = 0;
            stats.integral_y = 0;
            stats.area = 0;
            statsv.resize(nlabels, stats);
        }
        void operator()(int r, int c, LabelT l){
            ConnectedComponentStats &stats = statsv[l];
            if(c > stats.upper_x){
                stats.upper_x = c;
            }else{
                if(c < stats.lower_x){
                    stats.lower_x = c;
                }
            }
            if(r > stats.upper_y){
                stats.upper_y = r;
            }else{
                if(r < stats.lower_y){
                    stats.lower_y = r;
                }
            }
            stats.integral_x += c;
            stats.integral_y += r;
            stats.area++;
        }
        void finish(){
            for(size_t l = 0; l < statsv.size(); ++l){
                ConnectedComponentStats &stats = statsv[l];
                stats.lower_x = std::min(stats.lower_x, stats.upper_x);
                stats.lower_y = std::min(stats.lower_y, stats.upper_y);
                stats.centroid_x = stats.integral_x / double(stats.area);
                stats.centroid_y = stats.integral_y / double(stats.area);
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
    template<typename LabelT, typename PixelT, typename StatsOp = NoOp<LabelT>, int connectivity = 8>
    struct LabelingImpl{
    LabelT operator()(Mat &L, const Mat &I, StatsOp &sop){
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
                assert(connectivity == 4);
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
uint64_t connectedComponents_sub1(Mat &L, const Mat &I, int connectivity, StatsOp &sop){
    CV_Assert(L.rows == I.rows);
    CV_Assert(L.cols == I.cols);
    CV_Assert(L.channels() == 1 && I.channels() == 1);
    CV_Assert(connectivity == 8 || connectivity == 4);

    int lDepth = L.depth();
    int iDepth = I.depth();
    using connectedcomponents::LabelingImpl;
    //warn if L's depth is not sufficient?

    if(lDepth == CV_8U){
        if(iDepth == CV_8U || iDepth == CV_8S){
            if(connectivity == 4){
                return (uint64_t) LabelingImpl<uint8_t, uint8_t, StatsOp, 4>()(L, I, sop);
            }else{
                return (uint64_t) LabelingImpl<uint8_t, uint8_t, StatsOp, 8>()(L, I, sop);
            }
        }else if(iDepth == CV_16U || iDepth == CV_16S){
            if(connectivity == 4){
                return (uint64_t) LabelingImpl<uint8_t, uint16_t, StatsOp, 4>()(L, I, sop);
            }else{
                return (uint64_t) LabelingImpl<uint8_t, uint16_t, StatsOp, 8>()(L, I, sop);
            }
        }else if(iDepth == CV_32S){
            if(connectivity == 4){
                return (uint64_t) LabelingImpl<uint8_t, int32_t, StatsOp, 4>()(L, I, sop);
            }else{
                return (uint64_t) LabelingImpl<uint8_t, int32_t, StatsOp, 8>()(L, I, sop);
            }
        }else if(iDepth == CV_32F){
            if(connectivity == 4){
                return (uint64_t) LabelingImpl<uint8_t, float, StatsOp, 4>()(L, I, sop);
            }else{
                return (uint64_t) LabelingImpl<uint8_t, float, StatsOp, 8>()(L, I, sop);
            }
        }else if(iDepth == CV_64F){
            if(connectivity == 4){
                return (uint64_t) LabelingImpl<uint8_t, double, StatsOp, 4>()(L, I, sop);
            }else{
                return (uint64_t) LabelingImpl<uint8_t, double, StatsOp, 8>()(L, I, sop);
            }
        }
    }else if(lDepth == CV_16U){
        if(iDepth == CV_8U || iDepth == CV_8S){
            if(connectivity == 4){
                return (uint64_t) LabelingImpl<uint16_t, uint8_t, StatsOp, 4>()(L, I, sop);
            }else{
                return (uint64_t) LabelingImpl<uint16_t, uint8_t, StatsOp, 8>()(L, I, sop);
            }
        }else if(iDepth == CV_16U || iDepth == CV_16S){
            if(connectivity == 4){
                return (uint64_t) LabelingImpl<uint16_t, uint16_t, StatsOp, 4>()(L, I, sop);
            }else{
                return (uint64_t) LabelingImpl<uint16_t, uint16_t, StatsOp, 8>()(L, I, sop);
            }
        }else if(iDepth == CV_32S){
            if(connectivity == 4){
                return (uint64_t) LabelingImpl<uint16_t, int32_t, StatsOp, 4>()(L, I, sop);
            }else{
                return (uint64_t) LabelingImpl<uint16_t, int32_t, StatsOp, 8>()(L, I, sop);
            }
        }else if(iDepth == CV_32F){
            if(connectivity == 4){
                return (uint64_t) LabelingImpl<uint16_t, float, StatsOp, 4>()(L, I, sop);
            }else{
                return (uint64_t) LabelingImpl<uint16_t, float, StatsOp, 8>()(L, I, sop);
            }
        }else if(iDepth == CV_64F){
            if(connectivity == 4){
                return (uint64_t) LabelingImpl<uint16_t, double, StatsOp, 4>()(L, I, sop);
            }else{
                return (uint64_t) LabelingImpl<uint16_t, double, StatsOp, 8>()(L, I, sop);
            }
        }
    }else if(lDepth == CV_32S){
        //note that signed types don't really make sense here and not being able to use uint32_t matters for scientific projects
        //OpenCV: how should we proceed?  .at<T> typechecks in debug mode
        if(iDepth == CV_8U || iDepth == CV_8S){
            if(connectivity == 4){
                return (uint64_t) LabelingImpl<int32_t, uint8_t, StatsOp, 4>()(L, I, sop);
            }else{
                return (uint64_t) LabelingImpl<int32_t, uint8_t, StatsOp, 8>()(L, I, sop);
            }
        }else if(iDepth == CV_16U || iDepth == CV_16S){
            if(connectivity == 4){
                return (uint64_t) LabelingImpl<int32_t, uint16_t, StatsOp, 4>()(L, I, sop);
            }else{
                return (uint64_t) LabelingImpl<int32_t, uint16_t, StatsOp, 8>()(L, I, sop);
            }
        }else if(iDepth == CV_32S){
            if(connectivity == 4){
                return (uint64_t) LabelingImpl<int32_t, int32_t, StatsOp, 4>()(L, I, sop);
            }else{
                return (uint64_t) LabelingImpl<int32_t, int32_t, StatsOp, 8>()(L, I, sop);
            }
        }else if(iDepth == CV_32F){
            if(connectivity == 4){
                return (uint64_t) LabelingImpl<int32_t, float, StatsOp, 4>()(L, I, sop);
            }else{
                return (uint64_t) LabelingImpl<int32_t, float, StatsOp, 8>()(L, I, sop);
            }
        }else if(iDepth == CV_64F){
            if(connectivity == 4){
                return (uint64_t) LabelingImpl<int32_t, double, StatsOp, 4>()(L, I, sop);
            }else{
                return (uint64_t) LabelingImpl<int32_t, double, StatsOp, 8>()(L, I, sop);
            }
        }else{
            CV_Assert(false);
        }
    }

    CV_Error(CV_StsUnsupportedFormat, "unsupported label/image type");
    return -1;
}

uint64_t connectedComponents(Mat &L, const Mat &I, int connectivity){
    int lDepth = L.depth();
    if(lDepth == CV_8U){
        connectedcomponents::NoOp<uint8_t> sop; return connectedComponents_sub1(L, I, connectivity, sop);
    }else if(lDepth == CV_16U){
        connectedcomponents::NoOp<uint16_t> sop; return connectedComponents_sub1(L, I, connectivity, sop);
    }else if(lDepth == CV_32S){
        connectedcomponents::NoOp<uint32_t> sop; return connectedComponents_sub1(L, I, connectivity, sop);
    }else{
        CV_Assert(false);
        return 0;
    }
}

uint64_t connectedComponents(Mat &L, const Mat &I, std::vector<ConnectedComponentStats> &statsv, int connectivity){
    int lDepth = L.depth();
    if(lDepth == CV_8U){
        connectedcomponents::CCStatsOp<uint8_t> sop(statsv); return connectedComponents_sub1(L, I, connectivity, sop);
    }else if(lDepth == CV_16U){
        connectedcomponents::CCStatsOp<uint16_t> sop(statsv); return connectedComponents_sub1(L, I, connectivity, sop);
    }else if(lDepth == CV_32S){
        connectedcomponents::CCStatsOp<uint32_t> sop(statsv); return connectedComponents_sub1(L, I, connectivity, sop);
    }else{
        CV_Assert(false);
        return 0;
    }
}

}

