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
// 2016 Costantino Grama <costantino.grana@unimore.it>
// 2016 Federico Bolelli <federico.bolelli@hotmail.com>
// 2016 Lorenzo Baraldi <lorenzo.baraldi@unimore.it>
// 2016 Roberto Vezzani <roberto.vezzani@unimore.it>
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
    struct LabelingWu{
    LabelT operator()(const cv::Mat &I, cv::Mat &L, int connectivity, StatsOp &sop){
        CV_Assert(L.rows == I.rows);
        CV_Assert(L.cols == I.cols);
        CV_Assert(connectivity == 8 || connectivity == 4);
        const int rows = L.rows;
        const int cols = L.cols;
        //A quick and dirty upper bound for the maximimum number of labels.  The 4 comes from
        //the fact that a 3x3 block can never have more than 4 unique labels for both 4 & 8-way
        const size_t Plength = 4 * (size_t(rows + 3 - 1)/3) * (size_t(cols + 3 - 1)/3);
        LabelT *P = (LabelT *) fastMalloc(sizeof(LabelT) * Plength);
        P[0] = 0;
        LabelT lunique = 1;
        //scanning phase
        for(int r_i = 0; r_i < rows; ++r_i){
            LabelT * const Lrow = L.ptr<LabelT>(r_i);
            LabelT * const Lrow_prev = (LabelT *)(((char *)Lrow) - L.step.p[0]);
            const PixelT * const Irow = I.ptr<PixelT>(r_i);
            const PixelT * const Irow_prev = (const PixelT *)(((char *)Irow) - I.step.p[0]);
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
            LabelT *Lrow_start = L.ptr<LabelT>(r_i);
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
    }//End function LabelingWu operator()
    };//End struct LabelingWu

    // Based on "Optimized  Block-based Connected Components Labeling with Decision Trees", Costantino Grana et al
    // Only for 8-connectivity
    template<typename LabelT, typename PixelT, typename StatsOp = NoOp >
    struct LabelingGrana{
    LabelT operator()(const cv::Mat &img, cv::Mat &imgLabels, int connectivity,  StatsOp &sop){
        CV_Assert(img.rows == imgLabels.rows);
        CV_Assert(img.cols == imgLabels.cols);
        CV_Assert(connectivity == 8 || connectivity == 4);

        const int h = img.rows;
        const int w = img.cols;

        //A quick and dirty upper bound for the maximimum number of labels.
        const size_t Plength = img.rows*img.cols / 4;
        LabelT *P = (LabelT *)fastMalloc(sizeof(LabelT)* Plength);
        P[0] = 0;
        LabelT lunique = 1;

        // First scan
        for (int r = 0; r<h; r += 2) {
            // Get rows pointer
            const PixelT* const img_row = img.ptr<PixelT>(r);
            const PixelT* const img_row_prev = (PixelT *)(((char *)img_row) - img.step.p[0]);
            const PixelT* const img_row_prev_prev = (PixelT *)(((char *)img_row_prev) - img.step.p[0]);
            const PixelT* const img_row_fol = (PixelT *)(((char *)img_row) + img.step.p[0]);
            LabelT* const imgLabels_row = imgLabels.ptr<LabelT>(r);
            LabelT* const imgLabels_row_prev_prev = (LabelT *)(((char *)imgLabels_row) - imgLabels.step.p[0] - imgLabels.step.p[0]);
            for (int c = 0; c < w; c += 2) {

                // We work with 2x2 blocks
                // +-+-+-+
                // |P|Q|R|
                // +-+-+-+
                // |S|X|
                // +-+-+

                // The pixels are named as follows
                // +---+---+---+
                // |a b|c d|e f|
                // |g h|i j|k l|
                // +---+---+---+
                // |m n|o p|
                // |q r|s t|
                // +---+---+

                // Pixels a, f, l, q are not needed, since we need to understand the
                // the connectivity between these blocks and those pixels only metter
                // when considering the outer connectivities

                // A bunch of defines used to check if the pixels are foreground,
                // without going outside the image limits.
                #define condition_b c-1>=0 && r-2>=0 && img_row_prev_prev[c-1]>0
                #define condition_c r-2>=0 && img_row_prev_prev[c]>0
                #define condition_d c+1<w && r-2>=0 && img_row_prev_prev[c+1]>0
                #define condition_e c+2<w && r-2>=0 && img_row_prev_prev[c+2]>0

                #define condition_g c-2>=0 && r-1>=0 && img_row_prev[c-2]>0
                #define condition_h c-1>=0 && r-1>=0 && img_row_prev[c-1]>0
                #define condition_i r-1>=0 && img_row_prev[c]>0
                #define condition_j c+1<w && r-1>=0 && img_row_prev[c+1]>0
                #define condition_k c+2<w && r-1>=0 && img_row_prev[c+2]>0

                #define condition_m c-2>=0 && img_row[c-2]>0
                #define condition_n c-1>=0 && img_row[c-1]>0
                #define condition_o img_row[c]>0
                #define condition_p c+1<w && img_row[c+1]>0

                #define condition_r c-1>=0 && r+1<h && img_row_fol[c-1]>0
                #define condition_s r+1<h && img_row_fol[c]>0
                #define condition_t c+1<w && r+1<h && img_row_fol[c+1]>0

                // This is a decision tree which allows to choose which action to
                // perform, checking as few conditions as possible.
                // Actions: the blocks label are provisionally stored in the top left
                // pixel of the block in the labels image

                if (condition_o) {
                    if (condition_n) {
                        if (condition_j) {
                            if (condition_i) {
                                //Action_6: Assign label of block S
                                imgLabels_row[c] = imgLabels_row[c - 2];
                                continue;
                            }
                            else {
                                if (condition_c) {
                                    if (condition_h) {
                                        //Action_6: Assign label of block S
                                        imgLabels_row[c] = imgLabels_row[c - 2];
                                        continue;
                                    }
                                    else {
                                        if (condition_g) {
                                            if (condition_b) {
                                                //Action_6: Assign label of block S
                                                imgLabels_row[c] = imgLabels_row[c - 2];
                                                continue;
                                            }
                                            else {
                                                //Action_11: Merge labels of block Q and S
                                                imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
                                                continue;
                                            }
                                        }
                                        else {
                                            //Action_11: Merge labels of block Q and S
                                            imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
                                            continue;
                                        }
                                    }
                                }
                                else {
                                    //Action_11: Merge labels of block Q and S
                                    imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
                                    continue;
                                }
                            }
                        }
                        else {
                            if (condition_p) {
                                if (condition_k) {
                                    if (condition_d) {
                                        if (condition_i) {
                                            //Action_6: Assign label of block S
                                            imgLabels_row[c] = imgLabels_row[c - 2];
                                            continue;
                                        }
                                        else {
                                            if (condition_c) {
                                                if (condition_h) {
                                                    //Action_6: Assign label of block S
                                                    imgLabels_row[c] = imgLabels_row[c - 2];
                                                    continue;
                                                }
                                                else {
                                                    if (condition_g) {
                                                        if (condition_b) {
                                                            //Action_6: Assign label of block S
                                                            imgLabels_row[c] = imgLabels_row[c - 2];
                                                            continue;
                                                        }
                                                        else {
                                                            //Action_12: Merge labels of block R and S
                                                            imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);
                                                            continue;
                                                        }
                                                    }
                                                    else {
                                                        //Action_12: Merge labels of block R and S
                                                        imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);
                                                        continue;
                                                    }
                                                }
                                            }
                                            else {
                                                //Action_12: Merge labels of block R and S
                                                imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);
                                                continue;
                                            }
                                        }
                                    }
                                    else {
                                        //Action_12: Merge labels of block R and S
                                        imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);
                                        continue;
                                    }
                                }
                                else {
                                    //Action_6: Assign label of block S
                                    imgLabels_row[c] = imgLabels_row[c - 2];
                                    continue;
                                }
                            }
                            else {
                                //Action_6: Assign label of block S
                                imgLabels_row[c] = imgLabels_row[c - 2];
                                continue;
                            }
                        }
                    }
                    else {
                        if (condition_r) {
                            if (condition_j) {
                                if (condition_m) {
                                    if (condition_h) {
                                        if (condition_i) {
                                            //Action_6: Assign label of block S
                                            imgLabels_row[c] = imgLabels_row[c - 2];
                                            continue;
                                        }
                                        else {
                                            if (condition_c) {
                                                //Action_6: Assign label of block S
                                                imgLabels_row[c] = imgLabels_row[c - 2];
                                                continue;
                                            }
                                            else {
                                                //Action_11: Merge labels of block Q and S
                                                imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
                                                continue;
                                            }
                                        }
                                    }
                                    else {
                                        if (condition_g) {
                                            if (condition_b) {
                                                if (condition_i) {
                                                    //Action_6: Assign label of block S
                                                    imgLabels_row[c] = imgLabels_row[c - 2];
                                                    continue;
                                                }
                                                else {
                                                    if (condition_c) {
                                                        //Action_6: Assign label of block S
                                                        imgLabels_row[c] = imgLabels_row[c - 2];
                                                        continue;
                                                    }
                                                    else {
                                                        //Action_11: Merge labels of block Q and S
                                                        imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
                                                        continue;
                                                    }
                                                }
                                            }
                                            else {
                                                //Action_11: Merge labels of block Q and S
                                                imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
                                                continue;
                                            }
                                        }
                                        else {
                                            //Action_11: Merge labels of block Q and S
                                            imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
                                            continue;
                                        }
                                    }
                                }
                                else {
                                    if (condition_i) {
                                        //Action_11: Merge labels of block Q and S
                                        imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
                                        continue;
                                    }
                                    else {
                                        if (condition_h) {
                                            if (condition_c) {
                                                //Action_11: Merge labels of block Q and S
                                                imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
                                                continue;
                                            }
                                            else {
                                                //Action_14: Merge labels of block P, Q and S
                                                imgLabels_row[c] = set_union(P, set_union(P, imgLabels_row_prev_prev[c - 2], imgLabels_row_prev_prev[c]), imgLabels_row[c - 2]);
                                                continue;
                                            }
                                        }
                                        else {
                                            //Action_11: Merge labels of block Q and S
                                            imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
                                            continue;
                                        }
                                    }
                                }
                            }
                            else {
                                if (condition_p) {
                                    if (condition_k) {
                                        if (condition_m) {
                                            if (condition_h) {
                                                if (condition_d) {
                                                    if (condition_i) {
                                                        //Action_6: Assign label of block S
                                                        imgLabels_row[c] = imgLabels_row[c - 2];
                                                        continue;
                                                    }
                                                    else {
                                                        if (condition_c) {
                                                            //Action_6: Assign label of block S
                                                            imgLabels_row[c] = imgLabels_row[c - 2];
                                                            continue;
                                                        }
                                                        else {
                                                            //Action_12: Merge labels of block R and S
                                                            imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);
                                                            continue;
                                                        }
                                                    }
                                                }
                                                else {
                                                    //Action_12: Merge labels of block R and S
                                                    imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);
                                                    continue;
                                                }
                                            }
                                            else {
                                                if (condition_d) {
                                                    if (condition_g) {
                                                        if (condition_b) {
                                                            if (condition_i) {
                                                                //Action_6: Assign label of block S
                                                                imgLabels_row[c] = imgLabels_row[c - 2];
                                                                continue;
                                                            }
                                                            else {
                                                                if (condition_c) {
                                                                    //Action_6: Assign label of block S
                                                                    imgLabels_row[c] = imgLabels_row[c - 2];
                                                                    continue;
                                                                }
                                                                else {
                                                                    //Action_12: Merge labels of block R and S
                                                                    imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);
                                                                    continue;
                                                                }
                                                            }
                                                        }
                                                        else {
                                                            //Action_12: Merge labels of block R and S
                                                            imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);
                                                            continue;
                                                        }
                                                    }
                                                    else {
                                                        //Action_12: Merge labels of block R and S
                                                        imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);
                                                        continue;
                                                    }
                                                }
                                                else {
                                                    if (condition_i) {
                                                        if (condition_g) {
                                                            if (condition_b) {
                                                                //Action_12: Merge labels of block R and S
                                                                imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);
                                                                continue;
                                                            }
                                                            else {
                                                                //Action_16: labels of block Q, R and S
                                                                imgLabels_row[c] = set_union(P, set_union(P, imgLabels_row_prev_prev[c], imgLabels_row_prev_prev[c + 2]), imgLabels_row[c - 2]);
                                                                continue;
                                                            }
                                                        }
                                                        else {
                                                            //Action_16: labels of block Q, R and S
                                                            imgLabels_row[c] = set_union(P, set_union(P, imgLabels_row_prev_prev[c], imgLabels_row_prev_prev[c + 2]), imgLabels_row[c - 2]);
                                                            continue;
                                                        }
                                                    }
                                                    else {
                                                        //Action_12: Merge labels of block R and S
                                                        imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);
                                                        continue;
                                                    }
                                                }
                                            }
                                        }
                                        else {
                                            if (condition_i) {
                                                if (condition_d) {
                                                    //Action_12: Merge labels of block R and S
                                                    imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);
                                                    continue;
                                                }
                                                else {
                                                    //Action_16: labels of block Q, R and S
                                                    imgLabels_row[c] = set_union(P, set_union(P, imgLabels_row_prev_prev[c], imgLabels_row_prev_prev[c + 2]), imgLabels_row[c - 2]);
                                                    continue;
                                                }
                                            }
                                            else {
                                                if (condition_h) {
                                                    if (condition_d) {
                                                        if (condition_c) {
                                                            //Action_12: Merge labels of block R and S
                                                            imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);
                                                            continue;
                                                        }
                                                        else {
                                                            //Action_15: Merge labels of block P, R and S
                                                            imgLabels_row[c] = set_union(P, set_union(P, imgLabels_row_prev_prev[c - 2], imgLabels_row_prev_prev[c + 2]), imgLabels_row[c - 2]);
                                                            continue;
                                                        }
                                                    }
                                                    else {
                                                        //Action_15: Merge labels of block P, R and S
                                                        imgLabels_row[c] = set_union(P, set_union(P, imgLabels_row_prev_prev[c - 2], imgLabels_row_prev_prev[c + 2]), imgLabels_row[c - 2]);
                                                        continue;
                                                    }
                                                }
                                                else {
                                                    //Action_12: Merge labels of block R and S
                                                    imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);
                                                    continue;
                                                }
                                            }
                                        }
                                    }
                                    else {
                                        if (condition_h) {
                                            if (condition_m) {
                                                //Action_6: Assign label of block S
                                                imgLabels_row[c] = imgLabels_row[c - 2];
                                                continue;
                                            }
                                            else {
                                                // ACTION_9 Merge labels of block P and S
                                                imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c - 2], imgLabels_row[c - 2]);
                                                continue;
                                            }
                                        }
                                        else {
                                            if (condition_i) {
                                                if (condition_m) {
                                                    if (condition_g) {
                                                        if (condition_b) {
                                                            //Action_6: Assign label of block S
                                                            imgLabels_row[c] = imgLabels_row[c - 2];
                                                            continue;
                                                        }
                                                        else {
                                                            //Action_11: Merge labels of block Q and S
                                                            imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
                                                            continue;
                                                        }
                                                    }
                                                    else {
                                                        //Action_11: Merge labels of block Q and S
                                                        imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
                                                        continue;
                                                    }
                                                }
                                                else {
                                                    //Action_11: Merge labels of block Q and S
                                                    imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
                                                    continue;
                                                }
                                            }
                                            else {
                                                //Action_6: Assign label of block S
                                                imgLabels_row[c] = imgLabels_row[c - 2];
                                                continue;
                                            }
                                        }
                                    }
                                }
                                else {
                                    if (condition_h) {
                                        if (condition_m) {
                                            //Action_6: Assign label of block S
                                            imgLabels_row[c] = imgLabels_row[c - 2];
                                            continue;
                                        }
                                        else {
                                            // ACTION_9 Merge labels of block P and S
                                            imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c - 2], imgLabels_row[c - 2]);
                                            continue;
                                        }
                                    }
                                    else {
                                        if (condition_i) {
                                            if (condition_m) {
                                                if (condition_g) {
                                                    if (condition_b) {
                                                        //Action_6: Assign label of block S
                                                        imgLabels_row[c] = imgLabels_row[c - 2];
                                                        continue;
                                                    }
                                                    else {
                                                        //Action_11: Merge labels of block Q and S
                                                        imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
                                                        continue;
                                                    }
                                                }
                                                else {
                                                    //Action_11: Merge labels of block Q and S
                                                    imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
                                                    continue;
                                                }
                                            }
                                            else {
                                                //Action_11: Merge labels of block Q and S
                                                imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
                                                continue;
                                            }
                                        }
                                        else {
                                            //Action_6: Assign label of block S
                                            imgLabels_row[c] = imgLabels_row[c - 2];
                                            continue;
                                        }
                                    }
                                }
                            }
                        }
                        else {
                            if (condition_j) {
                                if (condition_i) {
                                    //Action_4: Assign label of block Q
                                    imgLabels_row[c] = imgLabels_row_prev_prev[c];
                                    continue;
                                }
                                else {
                                    if (condition_h) {
                                        if (condition_c) {
                                            //Action_4: Assign label of block Q
                                            imgLabels_row[c] = imgLabels_row_prev_prev[c];
                                            continue;
                                        }
                                        else {
                                            //Action_7: Merge labels of block P and Q
                                            imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c - 2], imgLabels_row_prev_prev[c]);
                                            continue;
                                        }
                                    }
                                    else {
                                        //Action_4: Assign label of block Q
                                        imgLabels_row[c] = imgLabels_row_prev_prev[c];
                                        continue;
                                    }
                                }
                            }
                            else {
                                if (condition_p) {
                                    if (condition_k) {
                                        if (condition_i) {
                                            if (condition_d) {
                                                //Action_5: Assign label of block R
                                                imgLabels_row[c] = imgLabels_row_prev_prev[c + 2];
                                                continue;
                                            }
                                            else {
                                                // ACTION_10 Merge labels of block Q and R
                                                imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row_prev_prev[c + 2]);
                                                continue;
                                            }
                                        }
                                        else {
                                            if (condition_h) {
                                                if (condition_d) {
                                                    if (condition_c) {
                                                        //Action_5: Assign label of block R
                                                        imgLabels_row[c] = imgLabels_row_prev_prev[c + 2];
                                                        continue;
                                                    }
                                                    else {
                                                        //Action_8: Merge labels of block P and R
                                                        imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c - 2], imgLabels_row_prev_prev[c + 2]);
                                                        continue;
                                                    }
                                                }
                                                else {
                                                    //Action_8: Merge labels of block P and R
                                                    imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c - 2], imgLabels_row_prev_prev[c + 2]);
                                                    continue;
                                                }
                                            }
                                            else {
                                                //Action_5: Assign label of block R
                                                imgLabels_row[c] = imgLabels_row_prev_prev[c + 2];
                                                continue;
                                            }
                                        }
                                    }
                                    else {
                                        if (condition_i) {
                                            //Action_4: Assign label of block Q
                                            imgLabels_row[c] = imgLabels_row_prev_prev[c];
                                            continue;
                                        }
                                        else {
                                            if (condition_h) {
                                                //Action_3: Assign label of block P
                                                imgLabels_row[c] = imgLabels_row_prev_prev[c - 2];
                                                continue;
                                            }
                                            else {
                                                //Action_2: New label (the block has foreground pixels and is not connected to anything else)
                                                imgLabels_row[c] = lunique;
                                                P[lunique] = lunique;
                                                lunique = lunique + 1;
                                                continue;
                                            }
                                        }
                                    }
                                }
                                else {
                                    if (condition_i) {
                                        //Action_4: Assign label of block Q
                                        imgLabels_row[c] = imgLabels_row_prev_prev[c];
                                        continue;
                                    }
                                    else {
                                        if (condition_h) {
                                            //Action_3: Assign label of block P
                                            imgLabels_row[c] = imgLabels_row_prev_prev[c - 2];
                                            continue;
                                        }
                                        else {
                                            //Action_2: New label (the block has foreground pixels and is not connected to anything else)
                                            imgLabels_row[c] = lunique;
                                            P[lunique] = lunique;
                                            lunique = lunique + 1;
                                            continue;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                else {
                    if (condition_s) {
                        if (condition_p) {
                            if (condition_n) {
                                if (condition_j) {
                                    if (condition_i) {
                                        //Action_6: Assign label of block S
                                        imgLabels_row[c] = imgLabels_row[c - 2];
                                        continue;
                                    }
                                    else {
                                        if (condition_c) {
                                            if (condition_h) {
                                                //Action_6: Assign label of block S
                                                imgLabels_row[c] = imgLabels_row[c - 2];
                                                continue;
                                            }
                                            else {
                                                if (condition_g) {
                                                    if (condition_b) {
                                                        //Action_6: Assign label of block S
                                                        imgLabels_row[c] = imgLabels_row[c - 2];
                                                        continue;
                                                    }
                                                    else {
                                                        //Action_11: Merge labels of block Q and S
                                                        imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
                                                        continue;
                                                    }
                                                }
                                                else {
                                                    //Action_11: Merge labels of block Q and S
                                                    imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
                                                    continue;
                                                }
                                            }
                                        }
                                        else {
                                            //Action_11: Merge labels of block Q and S
                                            imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
                                            continue;
                                        }
                                    }
                                }
                                else {
                                    if (condition_k) {
                                        if (condition_d) {
                                            if (condition_i) {
                                                //Action_6: Assign label of block S
                                                imgLabels_row[c] = imgLabels_row[c - 2];
                                                continue;
                                            }
                                            else {
                                                if (condition_c) {
                                                    if (condition_h) {
                                                        //Action_6: Assign label of block S
                                                        imgLabels_row[c] = imgLabels_row[c - 2];
                                                        continue;
                                                    }
                                                    else {
                                                        if (condition_g) {
                                                            if (condition_b) {
                                                                //Action_6: Assign label of block S
                                                                imgLabels_row[c] = imgLabels_row[c - 2];
                                                                continue;
                                                            }
                                                            else {
                                                                //Action_12: Merge labels of block R and S
                                                                imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);
                                                                continue;
                                                            }
                                                        }
                                                        else {
                                                            //Action_12: Merge labels of block R and S
                                                            imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);
                                                            continue;
                                                        }
                                                    }
                                                }
                                                else {
                                                    //Action_12: Merge labels of block R and S
                                                    imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);
                                                    continue;
                                                }
                                            }
                                        }
                                        else {
                                            //Action_12: Merge labels of block R and S
                                            imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);
                                            continue;
                                        }
                                    }
                                    else {
                                        //Action_6: Assign label of block S
                                        imgLabels_row[c] = imgLabels_row[c - 2];
                                        continue;
                                    }
                                }
                            }
                            else {
                                if (condition_r) {
                                    if (condition_j) {
                                        if (condition_m) {
                                            if (condition_h) {
                                                if (condition_i) {
                                                    //Action_6: Assign label of block S
                                                    imgLabels_row[c] = imgLabels_row[c - 2];
                                                    continue;
                                                }
                                                else {
                                                    if (condition_c) {
                                                        //Action_6: Assign label of block S
                                                        imgLabels_row[c] = imgLabels_row[c - 2];
                                                        continue;
                                                    }
                                                    else {
                                                        //Action_11: Merge labels of block Q and S
                                                        imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
                                                        continue;
                                                    }
                                                }
                                            }
                                            else {
                                                if (condition_g) {
                                                    if (condition_b) {
                                                        if (condition_i) {
                                                            //Action_6: Assign label of block S
                                                            imgLabels_row[c] = imgLabels_row[c - 2];
                                                            continue;
                                                        }
                                                        else {
                                                            if (condition_c) {
                                                                //Action_6: Assign label of block S
                                                                imgLabels_row[c] = imgLabels_row[c - 2];
                                                                continue;
                                                            }
                                                            else {
                                                                //Action_11: Merge labels of block Q and S
                                                                imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
                                                                continue;
                                                            }
                                                        }
                                                    }
                                                    else {
                                                        //Action_11: Merge labels of block Q and S
                                                        imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
                                                        continue;
                                                    }
                                                }
                                                else {
                                                    //Action_11: Merge labels of block Q and S
                                                    imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
                                                    continue;
                                                }
                                            }
                                        }
                                        else {
                                            //Action_11: Merge labels of block Q and S
                                            imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
                                            continue;
                                        }
                                    }
                                    else {
                                        if (condition_k) {
                                            if (condition_d) {
                                                if (condition_m) {
                                                    if (condition_h) {
                                                        if (condition_i) {
                                                            //Action_6: Assign label of block S
                                                            imgLabels_row[c] = imgLabels_row[c - 2];
                                                            continue;
                                                        }
                                                        else {
                                                            if (condition_c) {
                                                                //Action_6: Assign label of block S
                                                                imgLabels_row[c] = imgLabels_row[c - 2];
                                                                continue;
                                                            }
                                                            else {
                                                                //Action_12: Merge labels of block R and S
                                                                imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);
                                                                continue;
                                                            }
                                                        }
                                                    }
                                                    else {
                                                        if (condition_g) {
                                                            if (condition_b) {
                                                                if (condition_i) {
                                                                    //Action_6: Assign label of block S
                                                                    imgLabels_row[c] = imgLabels_row[c - 2];
                                                                    continue;
                                                                }
                                                                else {
                                                                    if (condition_c) {
                                                                        //Action_6: Assign label of block S
                                                                        imgLabels_row[c] = imgLabels_row[c - 2];
                                                                        continue;
                                                                    }
                                                                    else {
                                                                        //Action_12: Merge labels of block R and S
                                                                        imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);
                                                                        continue;
                                                                    }
                                                                }
                                                            }
                                                            else {
                                                                //Action_12: Merge labels of block R and S
                                                                imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);
                                                                continue;
                                                            }
                                                        }
                                                        else {
                                                            //Action_12: Merge labels of block R and S
                                                            imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);
                                                            continue;
                                                        }
                                                    }
                                                }
                                                else {
                                                    //Action_12: Merge labels of block R and S
                                                    imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);
                                                    continue;
                                                }
                                            }
                                            else {
                                                if (condition_i) {
                                                    if (condition_m) {
                                                        if (condition_h) {
                                                            //Action_12: Merge labels of block R and S
                                                            imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);
                                                            continue;
                                                        }
                                                        else {
                                                            if (condition_g) {
                                                                if (condition_b) {
                                                                    //Action_12: Merge labels of block R and S
                                                                    imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);
                                                                    continue;
                                                                }
                                                                else {
                                                                    //Action_16: labels of block Q, R and S
                                                                    imgLabels_row[c] = set_union(P, set_union(P, imgLabels_row_prev_prev[c], imgLabels_row_prev_prev[c + 2]), imgLabels_row[c - 2]);
                                                                    continue;
                                                                }
                                                            }
                                                            else {
                                                                //Action_16: labels of block Q, R and S
                                                                imgLabels_row[c] = set_union(P, set_union(P, imgLabels_row_prev_prev[c], imgLabels_row_prev_prev[c + 2]), imgLabels_row[c - 2]);
                                                                continue;
                                                            }
                                                        }
                                                    }
                                                    else {
                                                        //Action_16: labels of block Q, R and S
                                                        imgLabels_row[c] = set_union(P, set_union(P, imgLabels_row_prev_prev[c], imgLabels_row_prev_prev[c + 2]), imgLabels_row[c - 2]);
                                                        continue;
                                                    }
                                                }
                                                else {
                                                    //Action_12: Merge labels of block R and S
                                                    imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);
                                                    continue;
                                                }
                                            }
                                        }
                                        else {
                                            if (condition_i) {
                                                if (condition_m) {
                                                    if (condition_h) {
                                                        //Action_6: Assign label of block S
                                                        imgLabels_row[c] = imgLabels_row[c - 2];
                                                        continue;
                                                    }
                                                    else {
                                                        if (condition_g) {
                                                            if (condition_b) {
                                                                //Action_6: Assign label of block S
                                                                imgLabels_row[c] = imgLabels_row[c - 2];
                                                                continue;
                                                            }
                                                            else {
                                                                //Action_11: Merge labels of block Q and S
                                                                imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
                                                                continue;
                                                            }
                                                        }
                                                        else {
                                                            //Action_11: Merge labels of block Q and S
                                                            imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
                                                            continue;
                                                        }
                                                    }
                                                }
                                                else {
                                                    //Action_11: Merge labels of block Q and S
                                                    imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
                                                    continue;
                                                }
                                            }
                                            else {
                                                //Action_6: Assign label of block S
                                                imgLabels_row[c] = imgLabels_row[c - 2];
                                                continue;
                                            }
                                        }
                                    }
                                }
                                else {
                                    if (condition_j) {
                                        //Action_4: Assign label of block Q
                                        imgLabels_row[c] = imgLabels_row_prev_prev[c];
                                        continue;
                                    }
                                    else {
                                        if (condition_k) {
                                            if (condition_i) {
                                                if (condition_d) {
                                                    //Action_5: Assign label of block R
                                                    imgLabels_row[c] = imgLabels_row_prev_prev[c + 2];
                                                    continue;
                                                }
                                                else {
                                                    // ACTION_10 Merge labels of block Q and R
                                                    imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row_prev_prev[c + 2]);
                                                    continue;
                                                }
                                            }
                                            else {
                                                //Action_5: Assign label of block R
                                                imgLabels_row[c] = imgLabels_row_prev_prev[c + 2];
                                                continue;
                                            }
                                        }
                                        else {
                                            if (condition_i) {
                                                //Action_4: Assign label of block Q
                                                imgLabels_row[c] = imgLabels_row_prev_prev[c];
                                                continue;
                                            }
                                            else {
                                                //Action_2: New label (the block has foreground pixels and is not connected to anything else)
                                                imgLabels_row[c] = lunique;
                                                P[lunique] = lunique;
                                                lunique = lunique + 1;
                                                continue;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        else {
                            if (condition_r) {
                                //Action_6: Assign label of block S
                                imgLabels_row[c] = imgLabels_row[c - 2];
                                continue;
                            }
                            else {
                                if (condition_n) {
                                    //Action_6: Assign label of block S
                                    imgLabels_row[c] = imgLabels_row[c - 2];
                                    continue;
                                }
                                else {
                                    //Action_2: New label (the block has foreground pixels and is not connected to anything else)
                                    imgLabels_row[c] = lunique;
                                    P[lunique] = lunique;
                                    lunique = lunique + 1;
                                    continue;
                                }
                            }
                        }
                    }
                    else {
                        if (condition_p) {
                            if (condition_j) {
                                //Action_4: Assign label of block Q
                                imgLabels_row[c] = imgLabels_row_prev_prev[c];
                                continue;
                            }
                            else {
                                if (condition_k) {
                                    if (condition_i) {
                                        if (condition_d) {
                                            //Action_5: Assign label of block R
                                            imgLabels_row[c] = imgLabels_row_prev_prev[c + 2];
                                            continue;
                                        }
                                        else {
                                            // ACTION_10 Merge labels of block Q and R
                                            imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row_prev_prev[c + 2]);
                                            continue;
                                        }
                                    }
                                    else {
                                        //Action_5: Assign label of block R
                                        imgLabels_row[c] = imgLabels_row_prev_prev[c + 2];
                                        continue;
                                    }
                                }
                                else {
                                    if (condition_i) {
                                        //Action_4: Assign label of block Q
                                        imgLabels_row[c] = imgLabels_row_prev_prev[c];
                                        continue;
                                    }
                                    else {
                                        //Action_2: New label (the block has foreground pixels and is not connected to anything else)
                                        imgLabels_row[c] = lunique;
                                        P[lunique] = lunique;
                                        lunique = lunique + 1;
                                        continue;
                                    }
                                }
                            }
                        }
                        else {
                            if (condition_t) {
                                //Action_2: New label (the block has foreground pixels and is not connected to anything else)
                                imgLabels_row[c] = lunique;
                                P[lunique] = lunique;
                                lunique = lunique + 1;
                                continue;
                            }
                            else {
                                // Action_1: No action (the block has no foreground pixels)
                                imgLabels_row[c] = 0;
                                continue;
                            }
                        }
                    }
                }
            }
        }

        // Second scan + analysis
        LabelT nLabels = flattenL(P, lunique);
        sop.init(nLabels);

        if (imgLabels.rows & 1){
            if (imgLabels.cols & 1){
                //Case 1: both rows and cols odd
                for (int r = 0; r<imgLabels.rows; r += 2) {
                    // Get rows pointer
                    const PixelT* const img_row = img.ptr<PixelT>(r);
                    const PixelT* const img_row_fol = (PixelT *)(((char *)img_row) + img.step.p[0]);
                    LabelT* const imgLabels_row = imgLabels.ptr<LabelT>(r);
                    LabelT* const imgLabels_row_fol = (LabelT *)(((char *)imgLabels_row) + imgLabels.step.p[0]);

                    for (int c = 0; c<imgLabels.cols; c += 2) {
                        LabelT iLabel = imgLabels_row[c];
                        if (iLabel>0) {
                            iLabel = P[iLabel];
                            if (img_row[c] > 0){
                                imgLabels_row[c] = iLabel;
                                sop(r, c, iLabel);
                            }
                            else{
                                imgLabels_row[c] = 0;
                                sop(r, c, 0);
                            }
                            if (c + 1<imgLabels.cols) {
                                if (img_row[c + 1] > 0){
                                    imgLabels_row[c + 1] = iLabel;
                                    sop(r, c + 1, iLabel);
                                }
                                else{
                                    imgLabels_row[c + 1] = 0;
                                    sop(r, c + 1, 0);
                                }
                                if (r + 1<imgLabels.rows) {
                                    if (img_row_fol[c] > 0){
                                        imgLabels_row_fol[c] = iLabel;
                                        sop(r + 1, c, iLabel);
                                    } else{
                                        imgLabels_row_fol[c] = 0;
                                        sop(r + 1, c, 0);
                                    }
                                    if (img_row_fol[c + 1]>0){
                                        imgLabels_row_fol[c + 1] = iLabel;
                                        sop(r + 1, c + 1, iLabel);
                                    } else{
                                        imgLabels_row_fol[c + 1] = 0;
                                        sop(r + 1, c + 1, 0);
                                    }
                               }
                            }
                            else if (r + 1<imgLabels.rows) {
                                if (img_row_fol[c]>0){
                                    imgLabels_row_fol[c] = iLabel;
                                    sop(r + 1, c, iLabel);
                                }else{
                                    imgLabels_row_fol[c] = 0;
                                    sop(r + 1, c, 0);
                                }
                            }
                        }
                        else {
                            imgLabels_row[c] = 0;
                            sop(r, c, 0);
                            if (c + 1<imgLabels.cols) {
                                imgLabels_row[c + 1] = 0;
                                sop(r, c + 1, 0);
                                if (r + 1<imgLabels.rows) {
                                    imgLabels_row_fol[c] = 0;
                                    imgLabels_row_fol[c + 1] = 0;
                                    sop(r + 1, c, 0);
                                    sop(r + 1, c + 1, 0);
                                }
                            }else if (r + 1<imgLabels.rows) {
                                imgLabels_row_fol[c] = 0;
                                sop(r + 1, c, 0);
                            }
                        }
                    }
                }
            }//END Case 1
            else{
                //Case 2: only rows odd
                for (int r = 0; r<imgLabels.rows; r += 2) {
                    // Get rows pointer
                    const PixelT* const img_row = img.ptr<PixelT>(r);
                    const PixelT* const img_row_fol = (PixelT *)(((char *)img_row) + img.step.p[0]);
                    LabelT* const imgLabels_row = imgLabels.ptr<LabelT>(r);
                    LabelT* const imgLabels_row_fol = (LabelT *)(((char *)imgLabels_row) + imgLabels.step.p[0]);

                    for (int c = 0; c<imgLabels.cols; c += 2) {
                        LabelT iLabel = imgLabels_row[c];
                        if (iLabel>0) {
                            iLabel = P[iLabel];
                            if (img_row[c]>0){
                                imgLabels_row[c] = iLabel;
                                sop(r, c, iLabel);
                            } else{
                                imgLabels_row[c] = 0;
                                sop(r, c, 0);
                            }
                            if (img_row[c + 1]>0){
                                imgLabels_row[c + 1] = iLabel;
                                sop(r, c + 1, iLabel);
                            }else{
                                imgLabels_row[c + 1] = 0;
                                sop(r, c + 1, 0);
                            }
                            if (r + 1<imgLabels.rows) {
                                if (img_row_fol[c]>0){
                                    imgLabels_row_fol[c] = iLabel;
                                    sop(r + 1, c, iLabel);
                                }else{
                                    imgLabels_row_fol[c] = 0;
                                    sop(r + 1, c, 0);
                                }
                                if (img_row_fol[c + 1]>0){
                                    imgLabels_row_fol[c + 1] = iLabel;
                                    sop(r + 1, c + 1, iLabel);
                                }else{
                                    imgLabels_row_fol[c + 1] = 0;
                                    sop(r + 1, c + 1, 0);
                                }
                            }
                        }
                        else {
                            imgLabels_row[c] = 0;
                            imgLabels_row[c + 1] = 0;
                            sop(r, c, 0);
                            sop(r, c + 1, 0);
                            if (r + 1<imgLabels.rows) {
                                imgLabels_row_fol[c] = 0;
                                imgLabels_row_fol[c + 1] = 0;
                                sop(r + 1, c, 0);
                                sop(r + 1, c + 1, 0);
                            }
                        }
                    }
                }
            }// END Case 2
        }
        else{
            if (imgLabels.cols & 1){
                //Case 3: only cols odd
                for (int r = 0; r<imgLabels.rows; r += 2) {
                    // Get rows pointer
                    const PixelT* const img_row = img.ptr<PixelT>(r);
                    const PixelT* const img_row_fol = (PixelT *)(((char *)img_row) + img.step.p[0]);
                    LabelT* const imgLabels_row = imgLabels.ptr<LabelT>(r);
                    LabelT* const imgLabels_row_fol = (LabelT *)(((char *)imgLabels_row) + imgLabels.step.p[0]);

                    for (int c = 0; c<imgLabels.cols; c += 2) {
                        LabelT iLabel = imgLabels_row[c];
                        if (iLabel>0) {
                            iLabel = P[iLabel];
                            if (img_row[c]>0){
                                imgLabels_row[c] = iLabel;
                                sop(r, c, iLabel);
                            }else{
                                imgLabels_row[c] = 0;
                                sop(r, c, 0);
                            }
                            if (img_row_fol[c]>0){
                                imgLabels_row_fol[c] = iLabel;
                                sop(r + 1, c, iLabel);
                            }else{
                                imgLabels_row_fol[c] = 0;
                                sop(r + 1, c, 0);
                            }
                            if (c + 1<imgLabels.cols) {
                                if (img_row[c + 1]>0){
                                    imgLabels_row[c + 1] = iLabel;
                                    sop(r, c + 1, iLabel);
                                }else{
                                    imgLabels_row[c + 1] = 0;
                                    sop(r, c + 1, 0);
                                }
                                if (img_row_fol[c + 1]>0){
                                    imgLabels_row_fol[c + 1] = iLabel;
                                    sop(r + 1, c + 1, iLabel);
                                }else{
                                    imgLabels_row_fol[c + 1] = 0;
                                    sop(r + 1, c + 1, 0);
                                }
                            }
                        }
                        else{
                            imgLabels_row[c] = 0;
                            imgLabels_row_fol[c] = 0;
                            sop(r, c, 0);
                            sop(r + 1, c, 0);
                            if (c + 1<imgLabels.cols) {
                                imgLabels_row[c + 1] = 0;
                                imgLabels_row_fol[c + 1] = 0;
                                sop(r, c + 1, 0);
                                sop(r + 1, c + 1, 0);
                            }
                        }
                    }
                }
            }// END case 3
            else{
                //Case 4: nothing odd
                for (int r = 0; r < imgLabels.rows; r += 2) {
                    // Get rows pointer
                    const PixelT* const img_row = img.ptr<PixelT>(r);
                    const PixelT* const img_row_fol = (PixelT *)(((char *)img_row) + img.step.p[0]);
                    LabelT* const imgLabels_row = imgLabels.ptr<LabelT>(r);
                    LabelT* const imgLabels_row_fol = (LabelT *)(((char *)imgLabels_row) + imgLabels.step.p[0]);

                    for (int c = 0; c<imgLabels.cols; c += 2) {
                        LabelT iLabel = imgLabels_row[c];
                        if (iLabel>0) {
                            iLabel = P[iLabel];
                            if (img_row[c] > 0){
                                imgLabels_row[c] = iLabel;
                                sop(r, c, iLabel);
                            }else{
                                imgLabels_row[c] = 0;
                                sop(r, c, 0);
                            }
                            if (img_row[c + 1] > 0){
                                imgLabels_row[c + 1] = iLabel;
                                sop(r, c + 1, iLabel);
                            }else{
                                imgLabels_row[c + 1] = 0;
                                sop(r, c + 1, 0);
                            }
                            if (img_row_fol[c] > 0){
                                imgLabels_row_fol[c] = iLabel;
                                sop(r + 1, c, iLabel);
                            }else{
                                imgLabels_row_fol[c] = 0;
                                sop(r + 1, c, 0);
                            }
                            if (img_row_fol[c + 1] > 0){
                                imgLabels_row_fol[c + 1] = iLabel;
                                sop(r + 1, c + 1, iLabel);
                            }else{
                                imgLabels_row_fol[c + 1] = 0;
                                sop(r + 1, c + 1, 0);
                            }
                        }
                        else {
                            imgLabels_row[c] = 0;
                            imgLabels_row[c + 1] = 0;
                            imgLabels_row_fol[c] = 0;
                            imgLabels_row_fol[c + 1] = 0;
                            sop(r, c, 0);
                            sop(r, c + 1, 0);
                            sop(r + 1, c, 0);
                            sop(r + 1, c + 1, 0);
                        }
                    }
                }
            }//END case 4
        }

        sop.finish();
        fastFree(P);

        return nLabels;

    }   //End function LabelingGrana operator()
    }; //End struct LabelingGrana
}//end namespace connectedcomponents

//L's type must have an appropriate depth for the number of pixels in I
template<typename StatsOp>
static
int connectedComponents_sub1(const cv::Mat &I, cv::Mat &L, int connectivity, int ccltype, StatsOp &sop){
    CV_Assert(L.channels() == 1 && I.channels() == 1);
    CV_Assert(connectivity == 8 || connectivity == 4);
    CV_Assert(ccltype == CCL_GRANA || ccltype == CCL_WU || ccltype == CCL_DEFAULT);

    int lDepth = L.depth();
    int iDepth = I.depth();

    CV_Assert(iDepth == CV_8U || iDepth == CV_8S);

    if (ccltype == CCL_WU || connectivity == 4){
        // Wu algorithm is used
        using connectedcomponents::LabelingWu;
        //warn if L's depth is not sufficient?
        if (lDepth == CV_8U){
            return (int)LabelingWu<uchar, uchar, StatsOp>()(I, L, connectivity, sop);
        }
        else if (lDepth == CV_16U){
            return (int)LabelingWu<ushort, uchar, StatsOp>()(I, L, connectivity, sop);
        }
        else if (lDepth == CV_32S){
            //note that signed types don't really make sense here and not being able to use unsigned matters for scientific projects
            //OpenCV: how should we proceed?  .at<T> typechecks in debug mode
            return (int)LabelingWu<int, uchar, StatsOp>()(I, L, connectivity, sop);
        }
    }else if ((ccltype == CCL_GRANA || ccltype == CCL_DEFAULT) && connectivity == 8){
        // Grana algorithm is used
        using connectedcomponents::LabelingGrana;
        //warn if L's depth is not sufficient?
        if (lDepth == CV_8U){
            return (int)LabelingGrana<uchar, uchar, StatsOp>()(I, L, connectivity, sop);
        }
        else if (lDepth == CV_16U){
            return (int)LabelingGrana<ushort, uchar, StatsOp>()(I, L, connectivity, sop);
        }
        else if (lDepth == CV_32S){
            //note that signed types don't really make sense here and not being able to use unsigned matters for scientific projects
            //OpenCV: how should we proceed?  .at<T> typechecks in debug mode
            return (int)LabelingGrana<int, uchar, StatsOp>()(I, L, connectivity, sop);
        }
    }

    CV_Error(CV_StsUnsupportedFormat, "unsupported label/image type");
    return -1;
}

}

// Simple wrapper to ensure binary and source compatibility (ABI)
int cv::connectedComponents(InputArray _img, OutputArray _labels, int connectivity, int ltype){
    return cv::connectedComponents(_img, _labels, connectivity, ltype, CCL_DEFAULT);
}

int cv::connectedComponents(InputArray _img, OutputArray _labels, int connectivity, int ltype, int ccltype){
    const cv::Mat img = _img.getMat();
    _labels.create(img.size(), CV_MAT_DEPTH(ltype));
    cv::Mat labels = _labels.getMat();
    connectedcomponents::NoOp sop;
    if (ltype == CV_16U){
        return connectedComponents_sub1(img, labels, connectivity, ccltype, sop);
    }
    else if (ltype == CV_32S){
        return connectedComponents_sub1(img, labels, connectivity, ccltype, sop);
    }
    else{
        CV_Error(CV_StsUnsupportedFormat, "the type of labels must be 16u or 32s");
        return 0;
    }
}

// Simple wrapper to ensure binary and source compatibility (ABI)
int cv::connectedComponentsWithStats(InputArray _img, OutputArray _labels, OutputArray statsv,
                                     OutputArray centroids, int connectivity, int ltype)
{
    return cv::connectedComponentsWithStats(_img, _labels, statsv, centroids, connectivity, ltype, CCL_DEFAULT);
}

int cv::connectedComponentsWithStats(InputArray _img, OutputArray _labels, OutputArray statsv,
                                     OutputArray centroids, int connectivity, int ltype, int ccltype)
{
    const cv::Mat img = _img.getMat();
    _labels.create(img.size(), CV_MAT_DEPTH(ltype));
    cv::Mat labels = _labels.getMat();
    connectedcomponents::CCStatsOp sop(statsv, centroids);
    if (ltype == CV_16U){
        return connectedComponents_sub1(img, labels, connectivity, ccltype, sop);
    }
    else if (ltype == CV_32S){
        return connectedComponents_sub1(img, labels, connectivity, ccltype, sop);
    }
    else{
        CV_Error(CV_StsUnsupportedFormat, "the type of labels must be 16u or 32s");
        return 0;
    }
}
