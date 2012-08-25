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

namespace cv{
    namespace connectedcomponents{
    using std::vector;

    //Find the root of the tree of node i
    template<typename LabelT>
    inline static
    LabelT findRoot(const vector<LabelT> &P, LabelT i){
        LabelT root = i;
        while(P[root] < root){
            root = P[root];
        }
        return root;
    }

    //Make all nodes in the path of node i point to root
    template<typename LabelT>
    inline static
    void setRoot(vector<LabelT> &P, LabelT i, LabelT root){
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
    LabelT find(vector<LabelT> &P, LabelT i){
        LabelT root = findRoot(P, i);
        setRoot(P, i, root);
        return root;
    }

    //unite the two trees containing nodes i and j and return the new root
    template<typename LabelT>
    inline static
    LabelT set_union(vector<LabelT> &P, LabelT i, LabelT j){
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
    LabelT flattenL(vector<LabelT> &P){
        LabelT k = 1;
        for(size_t i = 1; i < P.size(); ++i){
            if(P[i] < i){
                P[i] = P[P[i]];
            }else{
                P[i] = k; k = k + 1;
            }
        }
        return k;
    }

    ////Flatten the Union Find tree - inconsistent labels
    //void flatten(int P[], int size){
    //    for(int i = 1; i < size; ++i){
    //        P[i] = P[P[i]];
    //    }
    //}
    const int G4[2][2] = {{-1, 0}, {0, -1}};//b, d neighborhoods
    const int G8[4][2] = {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}};//a, b, c, d neighborhoods
    //Based on "Two Strategies to Speed up Connected Components Algorithms", the SAUF (Scan array union find) variant
    //using decision trees
    //Kesheng Wu, et al
    template<typename LabelT, typename PixelT, int connectivity = 8>
    struct LabelingImpl{
    LabelT operator()(Mat &L, const Mat &I){
        const int rows = L.rows;
        const int cols = L.cols;
        size_t nPixels = size_t(rows) * cols;
        vector<LabelT> P; P.push_back(0);
        LabelT l = 1;
        //scanning phase
        for(int r_i = 0; r_i < rows; ++r_i){
            for(int c_i = 0; c_i < cols; ++c_i){
                if(!I.at<PixelT>(r_i, c_i)){
                    L.at<LabelT>(r_i, c_i) = 0;
                    continue;
                }
                if(connectivity == 8){
                    const int a = 0;
                    const int b = 1;
                    const int c = 2;
                    const int d = 3;

                    bool T[4];

                    for(size_t i = 0; i < 4; ++i){
                        int gr = r_i + G8[i][0];
                        int gc = c_i + G8[i][1];
                        T[i] = false;
                        if(gr >= 0 && gr < rows && gc >= 0 && gc < cols){
                            if(I.at<PixelT>(gr, gc)){
                                T[i] = true;
                            }
                        }
                    }

                    //decision tree
                    if(T[b]){
                        //copy(b)
                        L.at<LabelT>(r_i, c_i) = L.at<LabelT>(r_i + G8[b][0], c_i + G8[b][1]);
                    }else{//not b
                        if(T[c]){
                            if(T[a]){
                                //copy(c, a)
                                L.at<LabelT>(r_i, c_i) = set_union(P, L.at<LabelT>(r_i + G8[c][0], c_i + G8[c][1]), L.at<LabelT>(r_i + G8[a][0], c_i + G8[a][1]));
                            }else{
                                if(T[d]){
                                    //copy(c, d)
                                    L.at<LabelT>(r_i, c_i) = set_union(P, L.at<LabelT>(r_i + G8[c][0], c_i + G8[c][1]), L.at<LabelT>(r_i + G8[d][0], c_i + G8[d][1]));
                                }else{
                                    //copy(c)
                                    L.at<LabelT>(r_i, c_i) = L.at<LabelT>(r_i + G8[c][0], c_i + G8[c][1]);
                                }
                            }
                        }else{//not c
                            if(T[a]){
                                //copy(a)
                                L.at<LabelT>(r_i, c_i) = L.at<LabelT>(r_i + G8[a][0], c_i + G8[a][1]);
                            }else{
                                if(T[d]){
                                    //copy(d)
                                    L.at<LabelT>(r_i, c_i) = L.at<LabelT>(r_i + G8[d][0], c_i + G8[d][1]);
                                }else{
                                    //new label
                                    L.at<LabelT>(r_i, c_i) = l;
                                    P.push_back(l);//P[l] = l;
                                    l = l + 1;
                                }
                            }
                        }
                    }
                }else{
                    //B & D only
                    const int b = 0;
                    const int d = 1;
                    assert(connectivity == 4);
                    bool T[2];
                    for(size_t i = 0; i < 2; ++i){
                        int gr = r_i + G4[i][0];
                        int gc = c_i + G4[i][1];
                        T[i] = false;
                        if(gr >= 0 && gr < rows && gc >= 0 && gc < cols){
                            if(I.at<PixelT>(gr, gc)){
                                T[i] = true;
                            }
                        }
                    }

                    if(T[b]){
                        if(T[d]){
                            //copy(d, b)
                            L.at<LabelT>(r_i, c_i) = set_union(P, L.at<LabelT>(r_i + G4[d][0], c_i + G4[d][1]), L.at<LabelT>(r_i + G4[b][0], c_i + G4[b][1]));
                        }else{
                            //copy(b)
                            L.at<LabelT>(r_i, c_i) = L.at<LabelT>(r_i + G4[b][0], c_i + G4[b][1]);
                        }
                    }else{
                        if(T[d]){
                            //copy(d)
                            L.at<LabelT>(r_i, c_i) = L.at<LabelT>(r_i + G4[d][0], c_i + G4[d][1]);
                        }else{
                            //new label
                            L.at<LabelT>(r_i, c_i) = l;
                            P.push_back(l);//P[l] = l;
                            l = l + 1;
                        }
                    }

                }
            }
        }

        //analysis
        LabelT nLabels = flattenL(P);

        //assign final labels
        for(size_t r = 0; r < rows; ++r){
            for(size_t c = 0; c < cols; ++c){
                L.at<LabelT>(r, c) = P[L.at<LabelT>(r, c)];
            }
        }

        return nLabels;
    }//End function LabelingImpl operator()

    };//End struct LabelingImpl
}//end namespace connectedcomponents

//L's type must have an appropriate depth for the number of pixels in I
uint64_t connectedComponents(Mat &L, const Mat &I, int connectivity){
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
                return (uint64_t) LabelingImpl<uint8_t, uint8_t, 4>()(L, I);
            }else{
                return (uint64_t) LabelingImpl<uint8_t, uint8_t, 8>()(L, I);
            }
        }else if(iDepth == CV_16U || iDepth == CV_16S){
            if(connectivity == 4){
                return (uint64_t) LabelingImpl<uint8_t, uint16_t, 4>()(L, I);
            }else{
                return (uint64_t) LabelingImpl<uint8_t, uint16_t, 8>()(L, I);
            }
        }else if(iDepth == CV_32S){
            if(connectivity == 4){
                return (uint64_t) LabelingImpl<uint8_t, int32_t, 4>()(L, I);
            }else{
                return (uint64_t) LabelingImpl<uint8_t, int32_t, 8>()(L, I);
            }
        }else if(iDepth == CV_32F){
            if(connectivity == 4){
                return (uint64_t) LabelingImpl<uint8_t, float, 4>()(L, I);
            }else{
                return (uint64_t) LabelingImpl<uint8_t, float, 8>()(L, I);
            }
        }else if(iDepth == CV_64F){
            if(connectivity == 4){
                return (uint64_t) LabelingImpl<uint8_t, double, 4>()(L, I);
            }else{
                return (uint64_t) LabelingImpl<uint8_t, double, 8>()(L, I);
            }
        }
    }else if(lDepth == CV_16U){
        if(iDepth == CV_8U || iDepth == CV_8S){
            if(connectivity == 4){
                return (uint64_t) LabelingImpl<uint16_t, uint8_t, 4>()(L, I);
            }else{
                return (uint64_t) LabelingImpl<uint16_t, uint8_t, 8>()(L, I);
            }
        }else if(iDepth == CV_16U || iDepth == CV_16S){
            if(connectivity == 4){
                return (uint64_t) LabelingImpl<uint16_t, uint16_t, 4>()(L, I);
            }else{
                return (uint64_t) LabelingImpl<uint16_t, uint16_t, 8>()(L, I);
            }
        }else if(iDepth == CV_32S){
            if(connectivity == 4){
                return (uint64_t) LabelingImpl<uint16_t, int32_t, 4>()(L, I);
            }else{
                return (uint64_t) LabelingImpl<uint16_t, int32_t, 8>()(L, I);
            }
        }else if(iDepth == CV_32F){
            if(connectivity == 4){
                return (uint64_t) LabelingImpl<uint16_t, float, 4>()(L, I);
            }else{
                return (uint64_t) LabelingImpl<uint16_t, float, 8>()(L, I);
            }
        }else if(iDepth == CV_64F){
            if(connectivity == 4){
                return (uint64_t) LabelingImpl<uint16_t, double, 4>()(L, I);
            }else{
                return (uint64_t) LabelingImpl<uint16_t, double, 8>()(L, I);
            }
        }
    }else if(lDepth == CV_32S){
        if(iDepth == CV_8U || iDepth == CV_8S){
            if(connectivity == 4){
                return (uint64_t) LabelingImpl<int32_t, uint8_t, 4>()(L, I);
            }else{
                return (uint64_t) LabelingImpl<int32_t, uint8_t, 8>()(L, I);
            }
        }else if(iDepth == CV_16U || iDepth == CV_16S){
            if(connectivity == 4){
                return (uint64_t) LabelingImpl<int32_t, uint16_t, 4>()(L, I);
            }else{
                return (uint64_t) LabelingImpl<int32_t, uint16_t, 8>()(L, I);
            }
        }else if(iDepth == CV_32S){
            if(connectivity == 4){
                return (uint64_t) LabelingImpl<int32_t, int32_t, 4>()(L, I);
            }else{
                return (uint64_t) LabelingImpl<int32_t, int32_t, 8>()(L, I);
            }
        }else if(iDepth == CV_32F){
            if(connectivity == 4){
                return (uint64_t) LabelingImpl<int32_t, float, 4>()(L, I);
            }else{
                return (uint64_t) LabelingImpl<int32_t, float, 8>()(L, I);
            }
        }else if(iDepth == CV_64F){
            if(connectivity == 4){
                return (uint64_t) LabelingImpl<int32_t, double, 4>()(L, I);
            }else{
                return (uint64_t) LabelingImpl<int32_t, double, 8>()(L, I);
            }
        }
    }

    CV_Error(CV_StsUnsupportedFormat, "unsupported label/image type");
    return -1;
}


}

