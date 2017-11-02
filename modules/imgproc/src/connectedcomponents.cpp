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
// 2016 Michele Cancilla <cancilla.michele@gmail.com>
//M*/
//
#include "precomp.hpp"
#include <vector>

namespace cv{
    namespace connectedcomponents{

    struct NoOp{
        NoOp(){
        }

        inline
        void init(int /*labels*/){
        }

        inline
        void initElement(const int /*nlabels*/){
        }

        inline
        void operator()(int r, int c, int l){
            (void)r;
            (void)c;
            (void)l;
        }

        void finish(){
        }

        inline
        void setNextLoc(const int /*nextLoc*/){
        }

        inline static
        void mergeStats(const cv::Mat& /*imgLabels*/, NoOp * /*sopArray*/, NoOp& /*sop*/, const int& /*nLabels*/){
        }

    };
    struct Point2ui64{
        uint64 x, y;
        Point2ui64(uint64 _x, uint64 _y) :x(_x), y(_y){}
        };

    struct CCStatsOp{
        const _OutputArray *_mstatsv;
        cv::Mat statsv;
        const _OutputArray *_mcentroidsv;
        cv::Mat centroidsv;
        std::vector<Point2ui64> integrals;
        int _nextLoc;

        CCStatsOp() : _mstatsv(0), _mcentroidsv(0), _nextLoc(0) {}
        CCStatsOp(OutputArray _statsv, OutputArray _centroidsv) : _mstatsv(&_statsv), _mcentroidsv(&_centroidsv), _nextLoc(0){}

        inline
        void init(int nlabels){
            _mstatsv->create(cv::Size(CC_STAT_MAX, nlabels), cv::DataType<int>::type);
            statsv = _mstatsv->getMat();
            _mcentroidsv->create(cv::Size(2, nlabels), cv::DataType<double>::type);
            centroidsv = _mcentroidsv->getMat();

            for (int l = 0; l < (int)nlabels; ++l){
                int *row = (int *)&statsv.at<int>(l, 0);
                row[CC_STAT_LEFT] = INT_MAX;
                row[CC_STAT_TOP] = INT_MAX;
                row[CC_STAT_WIDTH] = INT_MIN;
                row[CC_STAT_HEIGHT] = INT_MIN;
                row[CC_STAT_AREA] = 0;
            }
            integrals.resize(nlabels, Point2ui64(0, 0));
        }

        inline
        void initElement(const int nlabels){
            statsv = cv::Mat(nlabels, CC_STAT_MAX, cv::DataType<int>::type);
            for (int l = 0; l < (int)nlabels; ++l){
                int *row = (int *)statsv.ptr(l);
                row[CC_STAT_LEFT] = INT_MAX;
                row[CC_STAT_TOP] = INT_MAX;
                row[CC_STAT_WIDTH] = INT_MIN;
                row[CC_STAT_HEIGHT] = INT_MIN;
                row[CC_STAT_AREA] = 0;
            }
            integrals.resize(nlabels, Point2ui64(0, 0));
        }

        void operator()(int r, int c, int l){
            int *row =& statsv.at<int>(l, 0);
            row[CC_STAT_LEFT] = MIN(row[CC_STAT_LEFT], c);
            row[CC_STAT_WIDTH] = MAX(row[CC_STAT_WIDTH], c);
            row[CC_STAT_TOP] = MIN(row[CC_STAT_TOP], r);
            row[CC_STAT_HEIGHT] = MAX(row[CC_STAT_HEIGHT], r);
            row[CC_STAT_AREA]++;
            Point2ui64& integral = integrals[l];
            integral.x += c;
            integral.y += r;
        }

        void finish(){
            for (int l = 0; l < statsv.rows; ++l){
                int *row =& statsv.at<int>(l, 0);
                row[CC_STAT_WIDTH] = row[CC_STAT_WIDTH] - row[CC_STAT_LEFT] + 1;
                row[CC_STAT_HEIGHT] = row[CC_STAT_HEIGHT] - row[CC_STAT_TOP] + 1;

                Point2ui64& integral = integrals[l];
                double *centroid = &centroidsv.at<double>(l, 0);
                double area = ((unsigned*)row)[CC_STAT_AREA];
                centroid[0] = double(integral.x) / area;
                centroid[1] = double(integral.y) / area;
            }
        }

        inline
        void setNextLoc(const int nextLoc){
            _nextLoc = nextLoc;
        }

        inline static
        void mergeStats(const cv::Mat& imgLabels, CCStatsOp *sopArray, CCStatsOp& sop, const int& nLabels){
            const int  h = imgLabels.rows;

            if (sop._nextLoc != h){
                for (int nextLoc = sop._nextLoc; nextLoc < h; nextLoc = sopArray[nextLoc]._nextLoc){
                    //merge between sopNext and sop
                    for (int l = 0; l < nLabels; ++l){
                        int *rowNext = (int*)sopArray[nextLoc].statsv.ptr(l);
                        if (rowNext[CC_STAT_AREA] > 0){ //if changed merge all the stats
                            int *rowMerged = (int*)sop.statsv.ptr(l);
                            rowMerged[CC_STAT_LEFT] = MIN(rowMerged[CC_STAT_LEFT], rowNext[CC_STAT_LEFT]);
                            rowMerged[CC_STAT_WIDTH] = MAX(rowMerged[CC_STAT_WIDTH], rowNext[CC_STAT_WIDTH]);
                            rowMerged[CC_STAT_TOP] = MIN(rowMerged[CC_STAT_TOP], rowNext[CC_STAT_TOP]);
                            rowMerged[CC_STAT_HEIGHT] = MAX(rowMerged[CC_STAT_HEIGHT], rowNext[CC_STAT_HEIGHT]);
                            rowMerged[CC_STAT_AREA] += rowNext[CC_STAT_AREA];

                            sop.integrals[l].x += sopArray[nextLoc].integrals[l].x;
                            sop.integrals[l].y += sopArray[nextLoc].integrals[l].y;
                        }
                    }
                }
            }
        }
    };

    //Find the root of the tree of node i
    template<typename LabelT>
    inline static
    LabelT findRoot(const LabelT *P, LabelT i){
        LabelT root = i;
        while (P[root] < root){
            root = P[root];
        }
        return root;
    }

    //Make all nodes in the path of node i point to root
    template<typename LabelT>
    inline static
    void setRoot(LabelT *P, LabelT i, LabelT root){
        while (P[i] < i){
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
        if (i != j){
            LabelT rootj = findRoot(P, j);
            if (root > rootj){
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
        for (LabelT i = 1; i < length; ++i){
            if (P[i] < i){
                P[i] = P[P[i]];
            }
            else{
                P[i] = k; k = k + 1;
            }
        }
        return k;
    }

    template<typename LabelT>
    inline static
    void flattenL(LabelT *P, const int start, const int nElem, LabelT& k){
        for (int i = start; i < start + nElem; ++i){
            if (P[i] < i){//node that point to root
                P[i] = P[P[i]];
            }
            else{ //for root node
                P[i] = k;
                k = k + 1;
            }
        }
    }

    //Based on "Two Strategies to Speed up Connected Components Algorithms", the SAUF (Scan array union find) variant
        //using decision trees
        //Kesheng Wu, et al
    template<typename LabelT, typename PixelT, typename StatsOp = NoOp >
    struct LabelingWuParallel{

        class FirstScan8Connectivity : public cv::ParallelLoopBody{
            const cv::Mat& img_;
            cv::Mat& imgLabels_;
            LabelT *P_;
            int *chunksSizeAndLabels_;

        public:
            FirstScan8Connectivity(const cv::Mat& img, cv::Mat& imgLabels, LabelT *P, int *chunksSizeAndLabels)
                : img_(img), imgLabels_(imgLabels), P_(P), chunksSizeAndLabels_(chunksSizeAndLabels){}

            FirstScan8Connectivity&  operator=(const FirstScan8Connectivity& ) { return *this; }

            void operator()(const cv::Range& range) const{

                int r = range.start;
                chunksSizeAndLabels_[r] = range.end;

                LabelT label = LabelT((r + 1) / 2)  * LabelT((imgLabels_.cols + 1) / 2) + 1;

                const LabelT firstLabel = label;
                const int w = img_.cols;
                const int limitLine = r, startR = r;

                // Rosenfeld Mask
                // +-+-+-+
                // |p|q|r|
                // +-+-+-+
                // |s|x|
                // +-+-+
                for (; r != range.end; ++r)
                {
                    PixelT const * const img_row = img_.ptr<PixelT>(r);
                    PixelT const * const img_row_prev = (PixelT *)(((char *)img_row) - img_.step.p[0]);
                    LabelT * const imgLabels_row = imgLabels_.ptr<LabelT>(r);
                    LabelT * const imgLabels_row_prev = (LabelT *)(((char *)imgLabels_row) - imgLabels_.step.p[0]);
                    for (int c = 0; c < w; ++c) {

#define condition_p c > 0 && r > limitLine && img_row_prev[c - 1] > 0
#define condition_q r > limitLine && img_row_prev[c] > 0
#define condition_r c < w - 1 && r > limitLine && img_row_prev[c + 1] > 0
#define condition_s c > 0 && img_row[c - 1] > 0
#define condition_x img_row[c] > 0

                        if (condition_x){
                            if (condition_q){
                                //copy q
                                imgLabels_row[c] = imgLabels_row_prev[c];
                            }
                            else{
                                //not q
                                if (condition_r){
                                    if (condition_p){
                                        //concavity p->x->r. Merge
                                        imgLabels_row[c] = set_union(P_, imgLabels_row_prev[c - 1], imgLabels_row_prev[c + 1]);
                                    }
                                    else{ //not p and q
                                        if (condition_s){
                                            //step s->x->r. Merge
                                            imgLabels_row[c] = set_union(P_, imgLabels_row[c - 1], imgLabels_row_prev[c + 1]);
                                        }
                                        else{ //not p, q and s
                                            //copy r
                                            imgLabels_row[c] = imgLabels_row_prev[c + 1];
                                        }
                                    }
                                }
                                else{
                                    //not r and q
                                    if (condition_p){
                                        //copy p
                                        imgLabels_row[c] = imgLabels_row_prev[c - 1];
                                    }
                                    else{//not r,q and p
                                        if (condition_s){
                                            imgLabels_row[c] = imgLabels_row[c - 1];
                                        }
                                        else{
                                            //new label
                                            imgLabels_row[c] = label;
                                            P_[label] = label;
                                            label = label + 1;
                                        }
                                    }
                                }
                            }
                        }
                        else{
                            //x is a background pixel
                            imgLabels_row[c] = 0;
                        }
                    }
                }
                //write in the follower memory location
                chunksSizeAndLabels_[startR + 1] = label - firstLabel;
            }
#undef condition_p
#undef condition_q
#undef condition_r
#undef condition_s
#undef condition_x
        };

        class FirstScan4Connectivity : public cv::ParallelLoopBody{
            const cv::Mat& img_;
            cv::Mat& imgLabels_;
            LabelT *P_;
            int *chunksSizeAndLabels_;

        public:
            FirstScan4Connectivity(const cv::Mat& img, cv::Mat& imgLabels, LabelT *P, int *chunksSizeAndLabels)
                : img_(img), imgLabels_(imgLabels), P_(P), chunksSizeAndLabels_(chunksSizeAndLabels){}

            FirstScan4Connectivity&  operator=(const FirstScan4Connectivity& ) { return *this; }

            void operator()(const cv::Range& range) const{

                int r = range.start;
                chunksSizeAndLabels_[r] = range.end;

                LabelT label = LabelT((r * imgLabels_.cols + 1) / 2 + 1);

                const LabelT firstLabel = label;
                const int w = img_.cols;
                const int limitLine = r, startR = r;

                // Rosenfeld Mask
                // +-+-+-+
                // |-|q|-|
                // +-+-+-+
                // |s|x|
                // +-+-+
                for (; r != range.end; ++r){
                    PixelT const * const img_row = img_.ptr<PixelT>(r);
                    PixelT const * const img_row_prev = (PixelT *)(((char *)img_row) - img_.step.p[0]);
                    LabelT * const imgLabels_row = imgLabels_.ptr<LabelT>(r);
                    LabelT * const imgLabels_row_prev = (LabelT *)(((char *)imgLabels_row) - imgLabels_.step.p[0]);
                    for (int c = 0; c < w; ++c) {

#define condition_q r > limitLine && img_row_prev[c] > 0
#define condition_s c > 0 && img_row[c - 1] > 0
#define condition_x img_row[c] > 0

                        if (condition_x){
                            if (condition_q){
                                if (condition_s){
                                    //step s->x->q. Merge
                                    imgLabels_row[c] = set_union(P_, imgLabels_row[c - 1], imgLabels_row_prev[c]);
                                }
                                else{
                                    //copy q
                                    imgLabels_row[c] = imgLabels_row_prev[c];
                                }
                            }
                            else{
                                if (condition_s){ // copy s
                                    imgLabels_row[c] = imgLabels_row[c - 1];
                                }
                                else{
                                    //new label
                                    imgLabels_row[c] = label;
                                    P_[label] = label;
                                    label = label + 1;
                                }
                            }
                        }
                        else{
                            //x is a background pixel
                            imgLabels_row[c] = 0;
                        }
                    }
                }
                //write in the following memory location
                chunksSizeAndLabels_[startR + 1] = label - firstLabel;
            }
#undef condition_q
#undef condition_s
#undef condition_x
        };

        class SecondScan : public cv::ParallelLoopBody{
            cv::Mat& imgLabels_;
            const LabelT *P_;
            StatsOp& sop_;
            StatsOp *sopArray_;
            LabelT& nLabels_;
        public:
            SecondScan(cv::Mat& imgLabels, const LabelT *P, StatsOp& sop, StatsOp *sopArray, LabelT& nLabels)
                : imgLabels_(imgLabels), P_(P), sop_(sop), sopArray_(sopArray), nLabels_(nLabels){}

            SecondScan&  operator=(const SecondScan& ) { return *this; }

            void operator()(const cv::Range& range) const{

                int r = range.start;
                const int rowBegin = r;
                const int rowEnd = range.end;

                if (rowBegin > 0){
                    sopArray_[rowBegin].initElement(nLabels_);
                    sopArray_[rowBegin].setNextLoc(rowEnd); //_nextLoc = rowEnd;

                    for (; r < rowEnd; ++r) {
                        LabelT * img_row_start = imgLabels_.ptr<LabelT>(r);
                        LabelT * const img_row_end = img_row_start + imgLabels_.cols;
                        for (int c = 0; img_row_start != img_row_end; ++img_row_start, ++c){
                            *img_row_start = P_[*img_row_start];
                            sopArray_[rowBegin](r, c, *img_row_start);
                        }
                    }
                }
                else{
                    //the first thread uses sop in order to make less merges
                    sop_.setNextLoc(rowEnd);
                    for (; r < rowEnd; ++r) {
                        LabelT * img_row_start = imgLabels_.ptr<LabelT>(r);
                        LabelT * const img_row_end = img_row_start + imgLabels_.cols;
                        for (int c = 0; img_row_start != img_row_end; ++img_row_start, ++c){
                            *img_row_start = P_[*img_row_start];
                            sop_(r, c, *img_row_start);
                        }
                    }
                }
            }
        };

        inline static
        void mergeLabels8Connectivity(cv::Mat& imgLabels, LabelT *P, const int *chunksSizeAndLabels){

            // Merge Mask
            // +-+-+-+
            // |p|q|r|
            // +-+-+-+
            //	 |x|
            //   +-+
            const int w = imgLabels.cols, h = imgLabels.rows;

            for (int r = chunksSizeAndLabels[0]; r < h; r = chunksSizeAndLabels[r]){

                LabelT * const imgLabels_row = imgLabels.ptr<LabelT>(r);
                LabelT * const imgLabels_row_prev = (LabelT *)(((char *)imgLabels_row) - imgLabels.step.p[0]);

                for (int c = 0; c < w; ++c){

#define condition_p c > 0 && imgLabels_row_prev[c - 1] > 0
#define condition_q imgLabels_row_prev[c] > 0
#define condition_r c < w - 1 && imgLabels_row_prev[c + 1] > 0
#define condition_x imgLabels_row[c] > 0

                    if (condition_x){
                        if (condition_p){
                            //merge of two label
                            imgLabels_row[c] = set_union(P, imgLabels_row_prev[c - 1], imgLabels_row[c]);
                        }
                        if (condition_r){
                            //merge of two label
                            imgLabels_row[c] = set_union(P, imgLabels_row_prev[c + 1], imgLabels_row[c]);
                        }
                        if (condition_q){
                            //merge of two label
                            imgLabels_row[c] = set_union(P, imgLabels_row_prev[c], imgLabels_row[c]);
                        }
                    }
                }
            }
#undef condition_p
#undef condition_q
#undef condition_r
#undef condition_x
        }

        inline static
        void mergeLabels4Connectivity(cv::Mat& imgLabels, LabelT *P, const int *chunksSizeAndLabels){

            // Merge Mask
            // +-+-+-+
            // |-|q|-|
            // +-+-+-+
            //	 |x|
            //   +-+
            const int w = imgLabels.cols, h = imgLabels.rows;

            for (int r = chunksSizeAndLabels[0]; r < h; r = chunksSizeAndLabels[r]){

                LabelT * const imgLabels_row = imgLabels.ptr<LabelT>(r);
                LabelT * const imgLabels_row_prev = (LabelT *)(((char *)imgLabels_row) - imgLabels.step.p[0]);

                for (int c = 0; c < w; ++c){

#define condition_q imgLabels_row_prev[c] > 0
#define condition_x imgLabels_row[c] > 0

                    if (condition_x){
                        if (condition_q){
                            //merge of two label
                            imgLabels_row[c] = set_union(P, imgLabels_row_prev[c], imgLabels_row[c]);
                        }
                    }
                }
            }
#undef condition_q
#undef condition_x
        }

        LabelT operator()(const cv::Mat& img, cv::Mat& imgLabels, int connectivity, StatsOp& sop){
            CV_Assert(img.rows == imgLabels.rows);
            CV_Assert(img.cols == imgLabels.cols);
            CV_Assert(connectivity == 8 || connectivity == 4);

            const int nThreads = cv::getNumberOfCPUs();
            cv::setNumThreads(nThreads);

            const int h = img.rows;
            const int w = img.cols;

            //A quick and dirty upper bound for the maximimum number of labels.
            //Following formula comes from the fact that a 2x2 block in 4-way connectivity
            //labeling can never have more than 2 new labels and 1 label for background.
            //Worst case image example pattern:
            //1 0 1 0 1...
            //0 1 0 1 0...
            //1 0 1 0 1...
            //............
            //Obviously, 4-way connectivity upper bound is also good for 8-way connectivity labeling
            const size_t Plength = (size_t(h) * size_t(w) + 1) / 2 + 1;

            //Array used to store info and labeled pixel by each thread.
            //Different threads affect different memory location of chunksSizeAndLabels
            int *chunksSizeAndLabels = (int *)cv::fastMalloc(h * sizeof(int));

            //Tree of labels
            LabelT *P = (LabelT *)cv::fastMalloc(Plength * sizeof(LabelT));
            //First label is for background
            P[0] = 0;

            cv::Range range(0, h);
            LabelT nLabels = 1;

            if (connectivity == 8){
                //First scan, each thread works with chunk of img.rows/nThreads rows
                //e.g. 300 rows, 4 threads -> each chunks is composed of 75 rows
                cv::parallel_for_(range, FirstScan8Connectivity(img, imgLabels, P, chunksSizeAndLabels), nThreads);

                //merge labels of different chunks
                mergeLabels8Connectivity(imgLabels, P, chunksSizeAndLabels);

                for (int i = 0; i < h; i = chunksSizeAndLabels[i]){
                    flattenL(P, int((i + 1) / 2) * int((w + 1) / 2) + 1, chunksSizeAndLabels[i + 1], nLabels);
                }
            }
            else{
                //First scan, each thread works with chunk of img.rows/nThreads rows
                //e.g. 300 rows, 4 threads -> each chunks is composed of 75 rows
                cv::parallel_for_(range, FirstScan4Connectivity(img, imgLabels, P, chunksSizeAndLabels), nThreads);

                //merge labels of different chunks
                mergeLabels4Connectivity(imgLabels, P, chunksSizeAndLabels);

                for (int i = 0; i < h; i = chunksSizeAndLabels[i]){
                    flattenL(P, int(i * w + 1) / 2 + 1, chunksSizeAndLabels[i + 1], nLabels);
                }
            }

            //Array for statistics dataof threads
            StatsOp *sopArray = new StatsOp[h];

            sop.init(nLabels);
            //Second scan
            cv::parallel_for_(range, SecondScan(imgLabels, P, sop, sopArray, nLabels), nThreads);
            StatsOp::mergeStats(imgLabels, sopArray, sop, nLabels);
            sop.finish();

            delete[] sopArray;
            cv::fastFree(chunksSizeAndLabels);
            cv::fastFree(P);
            return nLabels;
        }
    };//End struct LabelingWuParallel


    //Based on "Two Strategies to Speed up Connected Components Algorithms", the SAUF (Scan array union find) variant
    //using decision trees
    //Kesheng Wu, et al
    template<typename LabelT, typename PixelT, typename StatsOp = NoOp >
    struct LabelingWu{
        LabelT operator()(const cv::Mat& img, cv::Mat& imgLabels, int connectivity, StatsOp& sop){
            CV_Assert(imgLabels.rows == img.rows);
            CV_Assert(imgLabels.cols == img.cols);
            CV_Assert(connectivity == 8 || connectivity == 4);

            const int h = img.rows;
            const int w = img.cols;

            //A quick and dirty upper bound for the maximimum number of labels.
            //Following formula comes from the fact that a 2x2 block in 4-way connectivity
            //labeling can never have more than 2 new labels and 1 label for background.
            //Worst case image example pattern:
            //1 0 1 0 1...
            //0 1 0 1 0...
            //1 0 1 0 1...
            //............
            //Obviously, 4-way connectivity upper bound is also good for 8-way connectivity labeling
            const size_t Plength = (size_t(h) * size_t(w) + 1) / 2 + 1;
            //array P for equivalences resolution
            LabelT *P = (LabelT *)fastMalloc(sizeof(LabelT) *Plength);
            //first label is for background pixels
            P[0] = 0;
            LabelT lunique = 1;

            if (connectivity == 8){
                for (int r = 0; r < h; ++r){
                    // Get row pointers
                    PixelT const * const img_row = img.ptr<PixelT>(r);
                    PixelT const * const img_row_prev = (PixelT *)(((char *)img_row) - img.step.p[0]);
                    LabelT * const imgLabels_row = imgLabels.ptr<LabelT>(r);
                    LabelT * const imgLabels_row_prev = (LabelT *)(((char *)imgLabels_row) - imgLabels.step.p[0]);

                    for (int c = 0; c < w; ++c){

#define condition_p c>0 && r>0 && img_row_prev[c - 1]>0
#define condition_q r>0 && img_row_prev[c]>0
#define condition_r c < w - 1 && r > 0 && img_row_prev[c + 1] > 0
#define condition_s c > 0 && img_row[c - 1] > 0
#define condition_x img_row[c] > 0

                        if (condition_x){
                            if (condition_q){
                                //x <- q
                                imgLabels_row[c] = imgLabels_row_prev[c];
                            }
                            else{
                                // q = 0
                                if (condition_r){
                                    if (condition_p){
                                        // x <- merge(p,r)
                                        imgLabels_row[c] = set_union(P, imgLabels_row_prev[c - 1], imgLabels_row_prev[c + 1]);
                                    }
                                    else{
                                        // p = q = 0
                                        if (condition_s){
                                            // x <- merge(s,r)
                                            imgLabels_row[c] = set_union(P, imgLabels_row[c - 1], imgLabels_row_prev[c + 1]);
                                        }
                                        else{
                                            // p = q = s = 0
                                            // x <- r
                                            imgLabels_row[c] = imgLabels_row_prev[c + 1];
                                        }
                                    }
                                }
                                else{
                                    // r = q = 0
                                    if (condition_p){
                                        // x <- p
                                        imgLabels_row[c] = imgLabels_row_prev[c - 1];
                                    }
                                    else{
                                        // r = q = p = 0
                                        if (condition_s){
                                            imgLabels_row[c] = imgLabels_row[c - 1];
                                        }
                                        else{
                                            //new label
                                            imgLabels_row[c] = lunique;
                                            P[lunique] = lunique;
                                            lunique = lunique + 1;
                                        }
                                    }
                                }
                            }
                        }
                        else{
                            //x is a background pixel
                            imgLabels_row[c] = 0;
                        }
                    }
                }
#undef condition_p
#undef condition_q
#undef condition_r
#undef condition_s
#undef condition_x
            }
            else{
                for (int r = 0; r < h; ++r){
                    PixelT const * const img_row = img.ptr<PixelT>(r);
                    PixelT const * const img_row_prev = (PixelT *)(((char *)img_row) - img.step.p[0]);
                    LabelT * const imgLabels_row = imgLabels.ptr<LabelT>(r);
                    LabelT * const imgLabels_row_prev = (LabelT *)(((char *)imgLabels_row) - imgLabels.step.p[0]);
                    for (int c = 0; c < w; ++c) {

#define condition_q r > 0 && img_row_prev[c] > 0
#define condition_s c > 0 && img_row[c - 1] > 0
#define condition_x img_row[c] > 0

                        if (condition_x){
                            if (condition_q){
                                if (condition_s){
                                    //Merge s->x->q
                                    imgLabels_row[c] = set_union(P, imgLabels_row[c - 1], imgLabels_row_prev[c]);
                                }
                                else{
                                    //copy q
                                    imgLabels_row[c] = imgLabels_row_prev[c];
                                }
                            }
                            else{
                                if (condition_s){
                                    // copy s
                                    imgLabels_row[c] = imgLabels_row[c - 1];
                                }
                                else{
                                    //new label
                                    imgLabels_row[c] = lunique;
                                    P[lunique] = lunique;
                                    lunique = lunique + 1;
                                }
                            }
                        }
                        else{
                            //x is a background pixel
                            imgLabels_row[c] = 0;
                        }
                    }
                }
#undef condition_q
#undef condition_s
#undef condition_x
            }

            //analysis
            LabelT nLabels = flattenL(P, lunique);
            sop.init(nLabels);

            for (int r = 0; r < h; ++r) {
                LabelT * img_row_start = imgLabels.ptr<LabelT>(r);
                LabelT * const img_row_end = img_row_start + w;
                for (int c = 0; img_row_start != img_row_end; ++img_row_start, ++c){
                    *img_row_start = P[*img_row_start];
                    sop(r, c, *img_row_start);
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
    struct LabelingGranaParallel{

        class FirstScan : public cv::ParallelLoopBody{
        private:
            const cv::Mat& img_;
            cv::Mat& imgLabels_;
            LabelT *P_;
            int *chunksSizeAndLabels_;

        public:
            FirstScan(const cv::Mat& img, cv::Mat& imgLabels, LabelT *P, int *chunksSizeAndLabels)
                : img_(img), imgLabels_(imgLabels), P_(P), chunksSizeAndLabels_(chunksSizeAndLabels){}

            FirstScan&  operator=(const FirstScan&) { return *this; }

            void operator()(const cv::Range& range) const{

                int r = range.start;
                r += (r % 2);

                chunksSizeAndLabels_[r] = range.end + (range.end % 2);

                LabelT label = LabelT((r + 1) / 2)  * LabelT((imgLabels_.cols + 1) / 2) + 1;

                const LabelT firstLabel = label;
                const int h = img_.rows, w = img_.cols;
                const int limitLine = r + 1, startR = r;

                for (; r < range.end; r += 2){
                    // Get rows pointer
                    const PixelT * const img_row = img_.ptr<uchar>(r);
                    const PixelT * const img_row_prev = (PixelT *)(((char *)img_row) - img_.step.p[0]);
                    const PixelT * const img_row_prev_prev = (PixelT *)(((char *)img_row_prev) - img_.step.p[0]);
                    const PixelT * const img_row_fol = (PixelT *)(((char *)img_row) + img_.step.p[0]);
                    LabelT * const imgLabels_row = imgLabels_.ptr<LabelT>(r);
                    LabelT * const imgLabels_row_prev_prev = (LabelT *)(((char *)imgLabels_row) - imgLabels_.step.p[0] - imgLabels_.step.p[0]);
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

#define condition_b c-1>=0 && r > limitLine && img_row_prev_prev[c-1]>0
#define condition_c r > limitLine && img_row_prev_prev[c]>0
#define condition_d c+1<w && r > limitLine && img_row_prev_prev[c+1]>0
#define condition_e c+2<w && r > limitLine && img_row_prev_prev[c+2]>0

#define condition_g c-2>=0 && r > limitLine - 1 && img_row_prev[c-2]>0
#define condition_h c-1>=0 && r > limitLine - 1 && img_row_prev[c-1]>0
#define condition_i r > limitLine - 1 && img_row_prev[c]>0
#define condition_j c+1<w && r > limitLine - 1 && img_row_prev[c+1]>0
#define condition_k c+2<w && r > limitLine - 1 && img_row_prev[c+2]>0

#define condition_m c-2>=0 && img_row[c-2]>0
#define condition_n c-1>=0 && img_row[c-1]>0
#define condition_o img_row[c]>0
#define condition_p c+1<w && img_row[c+1]>0

#define condition_r c-1>=0 && r+1<h && img_row_fol[c-1]>0
#define condition_s r+1<h && img_row_fol[c]>0
#define condition_t c+1<w && r+1<h && img_row_fol[c+1]>0

                        // This is a decision tree which allows to choose which action to
                        // perform, checking as few conditions as possible.
                        // Actions are available after the tree.

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
                                                        imgLabels_row[c] = set_union(P_, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
                                                        continue;
                                                    }
                                                }
                                                else {
                                                    //Action_11: Merge labels of block Q and S
                                                    imgLabels_row[c] = set_union(P_, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
                                                    continue;
                                                }
                                            }
                                        }
                                        else {
                                            //Action_11: Merge labels of block Q and S
                                            imgLabels_row[c] = set_union(P_, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
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
                                                                    imgLabels_row[c] = set_union(P_, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);
                                                                    continue;
                                                                }
                                                            }
                                                            else {
                                                                //Action_12: Merge labels of block R and S
                                                                imgLabels_row[c] = set_union(P_, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);
                                                                continue;
                                                            }
                                                        }
                                                    }
                                                    else {
                                                        //Action_12: Merge labels of block R and S
                                                        imgLabels_row[c] = set_union(P_, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);
                                                        continue;
                                                    }
                                                }
                                            }
                                            else {
                                                //Action_12: Merge labels of block R and S
                                                imgLabels_row[c] = set_union(P_, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);
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
                                                        imgLabels_row[c] = set_union(P_, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
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
                                                                imgLabels_row[c] = set_union(P_, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
                                                                continue;
                                                            }
                                                        }
                                                    }
                                                    else {
                                                        //Action_11: Merge labels of block Q and S
                                                        imgLabels_row[c] = set_union(P_, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
                                                        continue;
                                                    }
                                                }
                                                else {
                                                    //Action_11: Merge labels of block Q and S
                                                    imgLabels_row[c] = set_union(P_, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
                                                    continue;
                                                }
                                            }
                                        }
                                        else {
                                            if (condition_i) {
                                                //Action_11: Merge labels of block Q and S
                                                imgLabels_row[c] = set_union(P_, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
                                                continue;
                                            }
                                            else {
                                                if (condition_h) {
                                                    if (condition_c) {
                                                        //Action_11: Merge labels of block Q and S
                                                        imgLabels_row[c] = set_union(P_, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
                                                        continue;
                                                    }
                                                    else {
                                                        //Action_14: Merge labels of block P_, Q and S
                                                        imgLabels_row[c] = set_union(P_, set_union(P_, imgLabels_row_prev_prev[c - 2], imgLabels_row_prev_prev[c]), imgLabels_row[c - 2]);
                                                        continue;
                                                    }
                                                }
                                                else {
                                                    //Action_11: Merge labels of block Q and S
                                                    imgLabels_row[c] = set_union(P_, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
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
                                                                    imgLabels_row[c] = set_union(P_, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);
                                                                    continue;
                                                                }
                                                            }
                                                        }
                                                        else {
                                                            //Action_12: Merge labels of block R and S
                                                            imgLabels_row[c] = set_union(P_, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);
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
                                                                            imgLabels_row[c] = set_union(P_, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);
                                                                            continue;
                                                                        }
                                                                    }
                                                                }
                                                                else {
                                                                    //Action_12: Merge labels of block R and S
                                                                    imgLabels_row[c] = set_union(P_, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);
                                                                    continue;
                                                                }
                                                            }
                                                            else {
                                                                //Action_12: Merge labels of block R and S
                                                                imgLabels_row[c] = set_union(P_, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);
                                                                continue;
                                                            }
                                                        }
                                                        else {
                                                            if (condition_i) {
                                                                if (condition_g) {
                                                                    if (condition_b) {
                                                                        //Action_12: Merge labels of block R and S
                                                                        imgLabels_row[c] = set_union(P_, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);
                                                                        continue;
                                                                    }
                                                                    else {
                                                                        //Action_16: labels of block Q, R and S
                                                                        imgLabels_row[c] = set_union(P_, set_union(P_, imgLabels_row_prev_prev[c], imgLabels_row_prev_prev[c + 2]), imgLabels_row[c - 2]);
                                                                        continue;
                                                                    }
                                                                }
                                                                else {
                                                                    //Action_16: labels of block Q, R and S
                                                                    imgLabels_row[c] = set_union(P_, set_union(P_, imgLabels_row_prev_prev[c], imgLabels_row_prev_prev[c + 2]), imgLabels_row[c - 2]);
                                                                    continue;
                                                                }
                                                            }
                                                            else {
                                                                //Action_12: Merge labels of block R and S
                                                                imgLabels_row[c] = set_union(P_, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);
                                                                continue;
                                                            }
                                                        }
                                                    }
                                                }
                                                else {
                                                    if (condition_i) {
                                                        if (condition_d) {
                                                            //Action_12: Merge labels of block R and S
                                                            imgLabels_row[c] = set_union(P_, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);
                                                            continue;
                                                        }
                                                        else {
                                                            //Action_16: labels of block Q, R and S
                                                            imgLabels_row[c] = set_union(P_, set_union(P_, imgLabels_row_prev_prev[c], imgLabels_row_prev_prev[c + 2]), imgLabels_row[c - 2]);
                                                            continue;
                                                        }
                                                    }
                                                    else {
                                                        if (condition_h) {
                                                            if (condition_d) {
                                                                if (condition_c) {
                                                                    //Action_12: Merge labels of block R and S
                                                                    imgLabels_row[c] = set_union(P_, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);
                                                                    continue;
                                                                }
                                                                else {
                                                                    //Action_15: Merge labels of block P_, R and S
                                                                    imgLabels_row[c] = set_union(P_, set_union(P_, imgLabels_row_prev_prev[c - 2], imgLabels_row_prev_prev[c + 2]), imgLabels_row[c - 2]);
                                                                    continue;
                                                                }
                                                            }
                                                            else {
                                                                //Action_15: Merge labels of block P_, R and S
                                                                imgLabels_row[c] = set_union(P_, set_union(P_, imgLabels_row_prev_prev[c - 2], imgLabels_row_prev_prev[c + 2]), imgLabels_row[c - 2]);
                                                                continue;
                                                            }
                                                        }
                                                        else {
                                                            //Action_12: Merge labels of block R and S
                                                            imgLabels_row[c] = set_union(P_, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);
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
                                                        // ACTION_9 Merge labels of block P_ and S
                                                        imgLabels_row[c] = set_union(P_, imgLabels_row_prev_prev[c - 2], imgLabels_row[c - 2]);
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
                                                                    imgLabels_row[c] = set_union(P_, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
                                                                    continue;
                                                                }
                                                            }
                                                            else {
                                                                //Action_11: Merge labels of block Q and S
                                                                imgLabels_row[c] = set_union(P_, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
                                                                continue;
                                                            }
                                                        }
                                                        else {
                                                            //Action_11: Merge labels of block Q and S
                                                            imgLabels_row[c] = set_union(P_, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
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
                                                    // ACTION_9 Merge labels of block P_ and S
                                                    imgLabels_row[c] = set_union(P_, imgLabels_row_prev_prev[c - 2], imgLabels_row[c - 2]);
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
                                                                imgLabels_row[c] = set_union(P_, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
                                                                continue;
                                                            }
                                                        }
                                                        else {
                                                            //Action_11: Merge labels of block Q and S
                                                            imgLabels_row[c] = set_union(P_, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
                                                            continue;
                                                        }
                                                    }
                                                    else {
                                                        //Action_11: Merge labels of block Q and S
                                                        imgLabels_row[c] = set_union(P_, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
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
                                                    //Action_7: Merge labels of block P_ and Q
                                                    imgLabels_row[c] = set_union(P_, imgLabels_row_prev_prev[c - 2], imgLabels_row_prev_prev[c]);
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
                                                        imgLabels_row[c] = set_union(P_, imgLabels_row_prev_prev[c], imgLabels_row_prev_prev[c + 2]);
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
                                                                //Action_8: Merge labels of block P_ and R
                                                                imgLabels_row[c] = set_union(P_, imgLabels_row_prev_prev[c - 2], imgLabels_row_prev_prev[c + 2]);
                                                                continue;
                                                            }
                                                        }
                                                        else {
                                                            //Action_8: Merge labels of block P_ and R
                                                            imgLabels_row[c] = set_union(P_, imgLabels_row_prev_prev[c - 2], imgLabels_row_prev_prev[c + 2]);
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
                                                        //Action_3: Assign label of block P_
                                                        imgLabels_row[c] = imgLabels_row_prev_prev[c - 2];
                                                        continue;
                                                    }
                                                    else {
                                                        //Action_2: New label (the block has foreground pixels and is not connected to anything else)
                                                        imgLabels_row[c] = label;
                                                        P_[label] = label;
                                                        label = label + 1;
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
                                                    //Action_3: Assign label of block P_
                                                    imgLabels_row[c] = imgLabels_row_prev_prev[c - 2];
                                                    continue;
                                                }
                                                else {
                                                    //Action_2: New label (the block has foreground pixels and is not connected to anything else)
                                                    imgLabels_row[c] = label;
                                                    P_[label] = label;
                                                    label = label + 1;
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
                                                                imgLabels_row[c] = set_union(P_, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
                                                                continue;
                                                            }
                                                        }
                                                        else {
                                                            //Action_11: Merge labels of block Q and S
                                                            imgLabels_row[c] = set_union(P_, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
                                                            continue;
                                                        }
                                                    }
                                                }
                                                else {
                                                    //Action_11: Merge labels of block Q and S
                                                    imgLabels_row[c] = set_union(P_, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
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
                                                                        imgLabels_row[c] = set_union(P_, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);
                                                                        continue;
                                                                    }
                                                                }
                                                                else {
                                                                    //Action_12: Merge labels of block R and S
                                                                    imgLabels_row[c] = set_union(P_, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);
                                                                    continue;
                                                                }
                                                            }
                                                        }
                                                        else {
                                                            //Action_12: Merge labels of block R and S
                                                            imgLabels_row[c] = set_union(P_, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);
                                                            continue;
                                                        }
                                                    }
                                                }
                                                else {
                                                    //Action_12: Merge labels of block R and S
                                                    imgLabels_row[c] = set_union(P_, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);
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
                                                                imgLabels_row[c] = set_union(P_, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
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
                                                                        imgLabels_row[c] = set_union(P_, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
                                                                        continue;
                                                                    }
                                                                }
                                                            }
                                                            else {
                                                                //Action_11: Merge labels of block Q and S
                                                                imgLabels_row[c] = set_union(P_, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
                                                                continue;
                                                            }
                                                        }
                                                        else {
                                                            //Action_11: Merge labels of block Q and S
                                                            imgLabels_row[c] = set_union(P_, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
                                                            continue;
                                                        }
                                                    }
                                                }
                                                else {
                                                    //Action_11: Merge labels of block Q and S
                                                    imgLabels_row[c] = set_union(P_, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
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
                                                                        imgLabels_row[c] = set_union(P_, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);
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
                                                                                imgLabels_row[c] = set_union(P_, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);
                                                                                continue;
                                                                            }
                                                                        }
                                                                    }
                                                                    else {
                                                                        //Action_12: Merge labels of block R and S
                                                                        imgLabels_row[c] = set_union(P_, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);
                                                                        continue;
                                                                    }
                                                                }
                                                                else {
                                                                    //Action_12: Merge labels of block R and S
                                                                    imgLabels_row[c] = set_union(P_, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);
                                                                    continue;
                                                                }
                                                            }
                                                        }
                                                        else {
                                                            //Action_12: Merge labels of block R and S
                                                            imgLabels_row[c] = set_union(P_, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);
                                                            continue;
                                                        }
                                                    }
                                                    else {
                                                        if (condition_i) {
                                                            if (condition_m) {
                                                                if (condition_h) {
                                                                    //Action_12: Merge labels of block R and S
                                                                    imgLabels_row[c] = set_union(P_, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);
                                                                    continue;
                                                                }
                                                                else {
                                                                    if (condition_g) {
                                                                        if (condition_b) {
                                                                            //Action_12: Merge labels of block R and S
                                                                            imgLabels_row[c] = set_union(P_, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);
                                                                            continue;
                                                                        }
                                                                        else {
                                                                            //Action_16: labels of block Q, R and S
                                                                            imgLabels_row[c] = set_union(P_, set_union(P_, imgLabels_row_prev_prev[c], imgLabels_row_prev_prev[c + 2]), imgLabels_row[c - 2]);
                                                                            continue;
                                                                        }
                                                                    }
                                                                    else {
                                                                        //Action_16: labels of block Q, R and S
                                                                        imgLabels_row[c] = set_union(P_, set_union(P_, imgLabels_row_prev_prev[c], imgLabels_row_prev_prev[c + 2]), imgLabels_row[c - 2]);
                                                                        continue;
                                                                    }
                                                                }
                                                            }
                                                            else {
                                                                //Action_16: labels of block Q, R and S
                                                                imgLabels_row[c] = set_union(P_, set_union(P_, imgLabels_row_prev_prev[c], imgLabels_row_prev_prev[c + 2]), imgLabels_row[c - 2]);
                                                                continue;
                                                            }
                                                        }
                                                        else {
                                                            //Action_12: Merge labels of block R and S
                                                            imgLabels_row[c] = set_union(P_, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);
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
                                                                        imgLabels_row[c] = set_union(P_, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
                                                                        continue;
                                                                    }
                                                                }
                                                                else {
                                                                    //Action_11: Merge labels of block Q and S
                                                                    imgLabels_row[c] = set_union(P_, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
                                                                    continue;
                                                                }
                                                            }
                                                        }
                                                        else {
                                                            //Action_11: Merge labels of block Q and S
                                                            imgLabels_row[c] = set_union(P_, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
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
                                                            imgLabels_row[c] = set_union(P_, imgLabels_row_prev_prev[c], imgLabels_row_prev_prev[c + 2]);
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
                                                        imgLabels_row[c] = label;
                                                        P_[label] = label;
                                                        label = label + 1;
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
                                            imgLabels_row[c] = label;
                                            P_[label] = label;
                                            label = label + 1;
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
                                                    imgLabels_row[c] = set_union(P_, imgLabels_row_prev_prev[c], imgLabels_row_prev_prev[c + 2]);
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
                                                imgLabels_row[c] = label;
                                                P_[label] = label;
                                                label = label + 1;
                                                continue;
                                            }
                                        }
                                    }
                                }
                                else {
                                    if (condition_t) {
                                        //Action_2: New label (the block has foreground pixels and is not connected to anything else)
                                        imgLabels_row[c] = label;
                                        P_[label] = label;
                                        label = label + 1;
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
                //write in the follower memory location
                chunksSizeAndLabels_[startR + 1] = label - firstLabel;
            }
#undef condition_k
#undef condition_j
#undef condition_i
#undef condition_h
#undef condition_g
#undef condition_e
#undef condition_d
#undef condition_c
#undef condition_b
        };

        class SecondScan : public cv::ParallelLoopBody{
        private:
            const cv::Mat& img_;
            cv::Mat& imgLabels_;
            LabelT *P_;
            StatsOp& sop_;
            StatsOp *sopArray_;
            LabelT& nLabels_;

        public:
            SecondScan(const cv::Mat& img, cv::Mat& imgLabels, LabelT *P, StatsOp& sop, StatsOp *sopArray, LabelT& nLabels)
                : img_(img), imgLabels_(imgLabels), P_(P), sop_(sop), sopArray_(sopArray), nLabels_(nLabels){}

            SecondScan&  operator=(const SecondScan& ) { return *this; }

            void operator()(const cv::Range& range) const{

                int r = range.start;
                r += (r % 2);
                const int rowBegin = r;
                const int rowEnd = range.end + range.end % 2;

                if (rowBegin > 0){
                    sopArray_[rowBegin].initElement(nLabels_);
                    sopArray_[rowBegin].setNextLoc(rowEnd); //_nextLoc = rowEnd;

                    if (imgLabels_.rows&  1){
                        if (imgLabels_.cols&  1){
                            //Case 1: both rows and cols odd
                            for (; r < rowEnd; r += 2){
                                // Get rows pointer
                                const PixelT * const img_row = img_.ptr<PixelT>(r);
                                const PixelT * const img_row_fol = (PixelT *)(((char *)img_row) + img_.step.p[0]);

                                LabelT * const imgLabels_row = imgLabels_.ptr<LabelT>(r);
                                LabelT * const imgLabels_row_fol = (LabelT *)(((char *)imgLabels_row) + imgLabels_.step.p[0]);
                                // Get rows pointer
                                for (int c = 0; c < imgLabels_.cols; c += 2) {
                                    LabelT iLabel = imgLabels_row[c];
                                    if (iLabel > 0) {
                                        iLabel = P_[iLabel];
                                        if (img_row[c] > 0){
                                            imgLabels_row[c] = iLabel;
                                            sopArray_[rowBegin](r, c, iLabel);
                                        }
                                        else{
                                            imgLabels_row[c] = 0;
                                            sopArray_[rowBegin](r, c, 0);
                                        }
                                        if (c + 1 < imgLabels_.cols) {
                                            if (img_row[c + 1] > 0){
                                                imgLabels_row[c + 1] = iLabel;
                                                sopArray_[rowBegin](r, c + 1, iLabel);
                                            }
                                            else{
                                                imgLabels_row[c + 1] = 0;
                                                sopArray_[rowBegin](r, c + 1, 0);
                                            }
                                            if (r + 1 < imgLabels_.rows) {
                                                if (img_row_fol[c] > 0){
                                                    imgLabels_row_fol[c] = iLabel;
                                                    sopArray_[rowBegin](r + 1, c, iLabel);
                                                }
                                                else{
                                                    imgLabels_row_fol[c] = 0;
                                                    sopArray_[rowBegin](r + 1, c, 0);
                                                }
                                                if (img_row_fol[c + 1] > 0){
                                                    imgLabels_row_fol[c + 1] = iLabel;
                                                    sopArray_[rowBegin](r + 1, c + 1, iLabel);
                                                }
                                                else{
                                                    imgLabels_row_fol[c + 1] = 0;
                                                    sopArray_[rowBegin](r + 1, c + 1, 0);
                                                }
                                            }
                                        }
                                        else if (r + 1 < imgLabels_.rows) {
                                            if (img_row_fol[c] > 0){
                                                imgLabels_row_fol[c] = iLabel;
                                                sopArray_[rowBegin](r + 1, c, iLabel);
                                            }
                                            else{
                                                imgLabels_row_fol[c] = 0;
                                                sopArray_[rowBegin](r + 1, c, 0);
                                            }
                                        }
                                    }
                                    else {
                                        imgLabels_row[c] = 0;
                                        sopArray_[rowBegin](r, c, 0);
                                        if (c + 1 < imgLabels_.cols) {
                                            imgLabels_row[c + 1] = 0;
                                            sopArray_[rowBegin](r, c + 1, 0);
                                            if (r + 1 < imgLabels_.rows) {
                                                imgLabels_row_fol[c] = 0;
                                                imgLabels_row_fol[c + 1] = 0;
                                                sopArray_[rowBegin](r + 1, c, 0);
                                                sopArray_[rowBegin](r + 1, c + 1, 0);
                                            }
                                        }
                                        else if (r + 1 < imgLabels_.rows) {
                                            imgLabels_row_fol[c] = 0;
                                            sopArray_[rowBegin](r + 1, c, 0);
                                        }
                                    }
                                }
                            }
                        }//END Case 1
                        else{
                            //Case 2: only rows odd
                            for (; r < rowEnd; r += 2){
                                // Get rows pointer
                                const PixelT * const img_row = img_.ptr<PixelT>(r);
                                const PixelT * const img_row_fol = (PixelT *)(((char *)img_row) + img_.step.p[0]);
                                LabelT * const imgLabels_row = imgLabels_.ptr<LabelT>(r);
                                LabelT * const imgLabels_row_fol = (LabelT *)(((char *)imgLabels_row) + imgLabels_.step.p[0]);
                                // Get rows pointer
                                for (int c = 0; c < imgLabels_.cols; c += 2) {
                                    LabelT iLabel = imgLabels_row[c];
                                    if (iLabel > 0) {
                                        iLabel = P_[iLabel];
                                        if (img_row[c] > 0){
                                            imgLabels_row[c] = iLabel;
                                            sopArray_[rowBegin](r, c, iLabel);
                                        }
                                        else{
                                            imgLabels_row[c] = 0;
                                            sopArray_[rowBegin](r, c, 0);
                                        }
                                        if (img_row[c + 1] > 0){
                                            imgLabels_row[c + 1] = iLabel;
                                            sopArray_[rowBegin](r, c + 1, iLabel);
                                        }
                                        else{
                                            imgLabels_row[c + 1] = 0;
                                            sopArray_[rowBegin](r, c + 1, 0);
                                        }
                                        if (r + 1 < imgLabels_.rows) {
                                            if (img_row_fol[c] > 0){
                                                imgLabels_row_fol[c] = iLabel;
                                                sopArray_[rowBegin](r + 1, c, iLabel);
                                            }
                                            else{
                                                imgLabels_row_fol[c] = 0;
                                                sopArray_[rowBegin](r + 1, c, 0);
                                            }
                                            if (img_row_fol[c + 1] > 0){
                                                imgLabels_row_fol[c + 1] = iLabel;
                                                sopArray_[rowBegin](r + 1, c + 1, iLabel);
                                            }
                                            else{
                                                imgLabels_row_fol[c + 1] = 0;
                                                sopArray_[rowBegin](r + 1, c + 1, 0);
                                            }
                                        }
                                    }
                                    else {
                                        imgLabels_row[c] = 0;
                                        imgLabels_row[c + 1] = 0;
                                        sopArray_[rowBegin](r, c, 0);
                                        sopArray_[rowBegin](r, c + 1, 0);
                                        if (r + 1 < imgLabels_.rows) {
                                            imgLabels_row_fol[c] = 0;
                                            imgLabels_row_fol[c + 1] = 0;
                                            sopArray_[rowBegin](r + 1, c, 0);
                                            sopArray_[rowBegin](r + 1, c + 1, 0);
                                        }
                                    }
                                }
                            }
                        }// END Case 2
                    }
                    else{
                        if (imgLabels_.cols&  1){
                            //Case 3: only cols odd
                            for (; r < rowEnd; r += 2){
                                // Get rows pointer
                                const PixelT * const img_row = img_.ptr<PixelT>(r);
                                const PixelT * const img_row_fol = (PixelT *)(((char *)img_row) + img_.step.p[0]);
                                LabelT * const imgLabels_row = imgLabels_.ptr<LabelT>(r);
                                LabelT * const imgLabels_row_fol = (LabelT *)(((char *)imgLabels_row) + imgLabels_.step.p[0]);
                                // Get rows pointer
                                for (int c = 0; c < imgLabels_.cols; c += 2) {
                                    LabelT iLabel = imgLabels_row[c];
                                    if (iLabel > 0) {
                                        iLabel = P_[iLabel];
                                        if (img_row[c] > 0){
                                            imgLabels_row[c] = iLabel;
                                            sopArray_[rowBegin](r, c, iLabel);
                                        }
                                        else{
                                            imgLabels_row[c] = 0;
                                            sopArray_[rowBegin](r, c, 0);
                                        }
                                        if (img_row_fol[c] > 0){
                                            imgLabels_row_fol[c] = iLabel;
                                            sopArray_[rowBegin](r + 1, c, iLabel);
                                        }
                                        else{
                                            imgLabels_row_fol[c] = 0;
                                            sopArray_[rowBegin](r + 1, c, 0);
                                        }
                                        if (c + 1 < imgLabels_.cols) {
                                            if (img_row[c + 1] > 0){
                                                imgLabels_row[c + 1] = iLabel;
                                                sopArray_[rowBegin](r, c + 1, iLabel);
                                            }
                                            else{
                                                imgLabels_row[c + 1] = 0;
                                                sopArray_[rowBegin](r, c + 1, 0);
                                            }
                                            if (img_row_fol[c + 1] > 0){
                                                imgLabels_row_fol[c + 1] = iLabel;
                                                sopArray_[rowBegin](r + 1, c + 1, iLabel);
                                            }
                                            else{
                                                imgLabels_row_fol[c + 1] = 0;
                                                sopArray_[rowBegin](r + 1, c + 1, 0);
                                            }
                                        }
                                    }
                                    else{
                                        imgLabels_row[c] = 0;
                                        imgLabels_row_fol[c] = 0;
                                        sopArray_[rowBegin](r, c, 0);
                                        sopArray_[rowBegin](r + 1, c, 0);
                                        if (c + 1 < imgLabels_.cols) {
                                            imgLabels_row[c + 1] = 0;
                                            imgLabels_row_fol[c + 1] = 0;
                                            sopArray_[rowBegin](r, c + 1, 0);
                                            sopArray_[rowBegin](r + 1, c + 1, 0);
                                        }
                                    }
                                }
                            }
                        }// END case 3
                        else{
                            //Case 4: nothing odd
                            for (; r < rowEnd; r += 2){
                                // Get rows pointer
                                const PixelT * const img_row = img_.ptr<PixelT>(r);
                                const PixelT * const img_row_fol = (PixelT *)(((char *)img_row) + img_.step.p[0]);
                                LabelT * const imgLabels_row = imgLabels_.ptr<LabelT>(r);
                                LabelT * const imgLabels_row_fol = (LabelT *)(((char *)imgLabels_row) + imgLabels_.step.p[0]);
                                // Get rows pointer
                                for (int c = 0; c < imgLabels_.cols; c += 2) {
                                    LabelT iLabel = imgLabels_row[c];
                                    if (iLabel > 0) {
                                        iLabel = P_[iLabel];
                                        if (img_row[c] > 0){
                                            imgLabels_row[c] = iLabel;
                                            sopArray_[rowBegin](r, c, iLabel);
                                        }
                                        else{
                                            imgLabels_row[c] = 0;
                                            sopArray_[rowBegin](r, c, 0);
                                        }
                                        if (img_row[c + 1] > 0){
                                            imgLabels_row[c + 1] = iLabel;
                                            sopArray_[rowBegin](r, c + 1, iLabel);
                                        }
                                        else{
                                            imgLabels_row[c + 1] = 0;
                                            sopArray_[rowBegin](r, c + 1, 0);
                                        }
                                        if (img_row_fol[c] > 0){
                                            imgLabels_row_fol[c] = iLabel;
                                            sopArray_[rowBegin](r + 1, c, iLabel);
                                        }
                                        else{
                                            imgLabels_row_fol[c] = 0;
                                            sopArray_[rowBegin](r + 1, c, 0);
                                        }
                                        if (img_row_fol[c + 1] > 0){
                                            imgLabels_row_fol[c + 1] = iLabel;
                                            sopArray_[rowBegin](r + 1, c + 1, iLabel);
                                        }
                                        else{
                                            imgLabels_row_fol[c + 1] = 0;
                                            sopArray_[rowBegin](r + 1, c + 1, 0);
                                        }
                                    }
                                    else {
                                        imgLabels_row[c] = 0;
                                        imgLabels_row[c + 1] = 0;
                                        imgLabels_row_fol[c] = 0;
                                        imgLabels_row_fol[c + 1] = 0;
                                        sopArray_[rowBegin](r, c, 0);
                                        sopArray_[rowBegin](r, c + 1, 0);
                                        sopArray_[rowBegin](r + 1, c, 0);
                                        sopArray_[rowBegin](r + 1, c + 1, 0);
                                    }
                                }
                            }//END case 4
                        }
                    }
                }
                else{
                    //the first thread uses sop in order to make less merges
                    sop_.setNextLoc(rowEnd);
                    if (imgLabels_.rows&  1){
                        if (imgLabels_.cols&  1){
                            //Case 1: both rows and cols odd
                            for (; r < rowEnd; r += 2){
                                // Get rows pointer
                                const PixelT * const img_row = img_.ptr<PixelT>(r);
                                const PixelT * const img_row_fol = (PixelT *)(((char *)img_row) + img_.step.p[0]);

                                LabelT * const imgLabels_row = imgLabels_.ptr<LabelT>(r);
                                LabelT * const imgLabels_row_fol = (LabelT *)(((char *)imgLabels_row) + imgLabels_.step.p[0]);
                                // Get rows pointer
                                for (int c = 0; c < imgLabels_.cols; c += 2) {
                                    LabelT iLabel = imgLabels_row[c];
                                    if (iLabel > 0) {
                                        iLabel = P_[iLabel];
                                        if (img_row[c] > 0){
                                            imgLabels_row[c] = iLabel;
                                            sop_(r, c, iLabel);
                                        }
                                        else{
                                            imgLabels_row[c] = 0;
                                            sop_(r, c, 0);
                                        }
                                        if (c + 1 < imgLabels_.cols) {
                                            if (img_row[c + 1] > 0){
                                                imgLabels_row[c + 1] = iLabel;
                                                sop_(r, c + 1, iLabel);
                                            }
                                            else{
                                                imgLabels_row[c + 1] = 0;
                                                sop_(r, c + 1, 0);
                                            }
                                            if (r + 1 < imgLabels_.rows) {
                                                if (img_row_fol[c] > 0){
                                                    imgLabels_row_fol[c] = iLabel;
                                                    sop_(r + 1, c, iLabel);
                                                }
                                                else{
                                                    imgLabels_row_fol[c] = 0;
                                                    sop_(r + 1, c, 0);
                                                }
                                                if (img_row_fol[c + 1] > 0){
                                                    imgLabels_row_fol[c + 1] = iLabel;
                                                    sop_(r + 1, c + 1, iLabel);
                                                }
                                                else{
                                                    imgLabels_row_fol[c + 1] = 0;
                                                    sop_(r + 1, c + 1, 0);
                                                }
                                            }
                                        }
                                        else if (r + 1 < imgLabels_.rows) {
                                            if (img_row_fol[c] > 0){
                                                imgLabels_row_fol[c] = iLabel;
                                                sop_(r + 1, c, iLabel);
                                            }
                                            else{
                                                imgLabels_row_fol[c] = 0;
                                                sop_(r + 1, c, 0);
                                            }
                                        }
                                    }
                                    else {
                                        imgLabels_row[c] = 0;
                                        sop_(r, c, 0);
                                        if (c + 1 < imgLabels_.cols) {
                                            imgLabels_row[c + 1] = 0;
                                            sop_(r, c + 1, 0);
                                            if (r + 1 < imgLabels_.rows) {
                                                imgLabels_row_fol[c] = 0;
                                                imgLabels_row_fol[c + 1] = 0;
                                                sop_(r + 1, c, 0);
                                                sop_(r + 1, c + 1, 0);
                                            }
                                        }
                                        else if (r + 1 < imgLabels_.rows) {
                                            imgLabels_row_fol[c] = 0;
                                            sop_(r + 1, c, 0);
                                        }
                                    }
                                }
                            }
                        }//END Case 1
                        else{
                            //Case 2: only rows odd
                            for (; r < rowEnd; r += 2){
                                // Get rows pointer
                                const PixelT * const img_row = img_.ptr<PixelT>(r);
                                const PixelT * const img_row_fol = (PixelT *)(((char *)img_row) + img_.step.p[0]);
                                LabelT * const imgLabels_row = imgLabels_.ptr<LabelT>(r);
                                LabelT * const imgLabels_row_fol = (LabelT *)(((char *)imgLabels_row) + imgLabels_.step.p[0]);
                                // Get rows pointer
                                for (int c = 0; c < imgLabels_.cols; c += 2) {
                                    LabelT iLabel = imgLabels_row[c];
                                    if (iLabel > 0) {
                                        iLabel = P_[iLabel];
                                        if (img_row[c] > 0){
                                            imgLabels_row[c] = iLabel;
                                            sop_(r, c, iLabel);
                                        }
                                        else{
                                            imgLabels_row[c] = 0;
                                            sop_(r, c, 0);
                                        }
                                        if (img_row[c + 1] > 0){
                                            imgLabels_row[c + 1] = iLabel;
                                            sop_(r, c + 1, iLabel);
                                        }
                                        else{
                                            imgLabels_row[c + 1] = 0;
                                            sop_(r, c + 1, 0);
                                        }
                                        if (r + 1 < imgLabels_.rows) {
                                            if (img_row_fol[c] > 0){
                                                imgLabels_row_fol[c] = iLabel;
                                                sop_(r + 1, c, iLabel);
                                            }
                                            else{
                                                imgLabels_row_fol[c] = 0;
                                                sop_(r + 1, c, 0);
                                            }
                                            if (img_row_fol[c + 1] > 0){
                                                imgLabels_row_fol[c + 1] = iLabel;
                                                sop_(r + 1, c + 1, iLabel);
                                            }
                                            else{
                                                imgLabels_row_fol[c + 1] = 0;
                                                sop_(r + 1, c + 1, 0);
                                            }
                                        }
                                    }
                                    else {
                                        imgLabels_row[c] = 0;
                                        imgLabels_row[c + 1] = 0;
                                        sop_(r, c, 0);
                                        sop_(r, c + 1, 0);
                                        if (r + 1 < imgLabels_.rows) {
                                            imgLabels_row_fol[c] = 0;
                                            imgLabels_row_fol[c + 1] = 0;
                                            sop_(r + 1, c, 0);
                                            sop_(r + 1, c + 1, 0);
                                        }
                                    }
                                }
                            }
                        }// END Case 2
                    }
                    else{
                        if (imgLabels_.cols&  1){
                            //Case 3: only cols odd
                            for (; r < rowEnd; r += 2){
                                // Get rows pointer
                                const PixelT * const img_row = img_.ptr<PixelT>(r);
                                const PixelT * const img_row_fol = (PixelT *)(((char *)img_row) + img_.step.p[0]);
                                LabelT * const imgLabels_row = imgLabels_.ptr<LabelT>(r);
                                LabelT * const imgLabels_row_fol = (LabelT *)(((char *)imgLabels_row) + imgLabels_.step.p[0]);
                                // Get rows pointer
                                for (int c = 0; c < imgLabels_.cols; c += 2) {
                                    LabelT iLabel = imgLabels_row[c];
                                    if (iLabel > 0) {
                                        iLabel = P_[iLabel];
                                        if (img_row[c] > 0){
                                            imgLabels_row[c] = iLabel;
                                            sop_(r, c, iLabel);
                                        }
                                        else{
                                            imgLabels_row[c] = 0;
                                            sop_(r, c, 0);
                                        }
                                        if (img_row_fol[c] > 0){
                                            imgLabels_row_fol[c] = iLabel;
                                            sop_(r + 1, c, iLabel);
                                        }
                                        else{
                                            imgLabels_row_fol[c] = 0;
                                            sop_(r + 1, c, 0);
                                        }
                                        if (c + 1 < imgLabels_.cols) {
                                            if (img_row[c + 1] > 0){
                                                imgLabels_row[c + 1] = iLabel;
                                                sop_(r, c + 1, iLabel);
                                            }
                                            else{
                                                imgLabels_row[c + 1] = 0;
                                                sop_(r, c + 1, 0);
                                            }
                                            if (img_row_fol[c + 1] > 0){
                                                imgLabels_row_fol[c + 1] = iLabel;
                                                sop_(r + 1, c + 1, iLabel);
                                            }
                                            else{
                                                imgLabels_row_fol[c + 1] = 0;
                                                sop_(r + 1, c + 1, 0);
                                            }
                                        }
                                    }
                                    else{
                                        imgLabels_row[c] = 0;
                                        imgLabels_row_fol[c] = 0;
                                        sop_(r, c, 0);
                                        sop_(r + 1, c, 0);
                                        if (c + 1 < imgLabels_.cols) {
                                            imgLabels_row[c + 1] = 0;
                                            imgLabels_row_fol[c + 1] = 0;
                                            sop_(r, c + 1, 0);
                                            sop_(r + 1, c + 1, 0);
                                        }
                                    }
                                }
                            }
                        }// END case 3
                        else{
                            //Case 4: nothing odd
                            for (; r < rowEnd; r += 2){
                                // Get rows pointer
                                const PixelT * const img_row = img_.ptr<PixelT>(r);
                                const PixelT * const img_row_fol = (PixelT *)(((char *)img_row) + img_.step.p[0]);
                                LabelT * const imgLabels_row = imgLabels_.ptr<LabelT>(r);
                                LabelT * const imgLabels_row_fol = (LabelT *)(((char *)imgLabels_row) + imgLabels_.step.p[0]);
                                // Get rows pointer
                                for (int c = 0; c < imgLabels_.cols; c += 2) {
                                    LabelT iLabel = imgLabels_row[c];
                                    if (iLabel > 0) {
                                        iLabel = P_[iLabel];
                                        if (img_row[c] > 0){
                                            imgLabels_row[c] = iLabel;
                                            sop_(r, c, iLabel);
                                        }
                                        else{
                                            imgLabels_row[c] = 0;
                                            sop_(r, c, 0);
                                        }
                                        if (img_row[c + 1] > 0){
                                            imgLabels_row[c + 1] = iLabel;
                                            sop_(r, c + 1, iLabel);
                                        }
                                        else{
                                            imgLabels_row[c + 1] = 0;
                                            sop_(r, c + 1, 0);
                                        }
                                        if (img_row_fol[c] > 0){
                                            imgLabels_row_fol[c] = iLabel;
                                            sop_(r + 1, c, iLabel);
                                        }
                                        else{
                                            imgLabels_row_fol[c] = 0;
                                            sop_(r + 1, c, 0);
                                        }
                                        if (img_row_fol[c + 1] > 0){
                                            imgLabels_row_fol[c + 1] = iLabel;
                                            sop_(r + 1, c + 1, iLabel);
                                        }
                                        else{
                                            imgLabels_row_fol[c + 1] = 0;
                                            sop_(r + 1, c + 1, 0);
                                        }
                                    }
                                    else {
                                        imgLabels_row[c] = 0;
                                        imgLabels_row[c + 1] = 0;
                                        imgLabels_row_fol[c] = 0;
                                        imgLabels_row_fol[c + 1] = 0;
                                        sop_(r, c, 0);
                                        sop_(r, c + 1, 0);
                                        sop_(r + 1, c, 0);
                                        sop_(r + 1, c + 1, 0);
                                    }
                                }
                            }//END case 4
                        }
                    }
                }
            }
        };

        inline static
        void mergeLabels(const cv::Mat& img, cv::Mat& imgLabels, LabelT *P, int *chunksSizeAndLabels){

                // Merge Mask
                // +---+---+---+
                // |P -|Q -|R -|
                // |- -|- -|- -|
                // +---+---+---+
                //	   |X -|
                //	   |- -|
                //	   +---+
                const int w = imgLabels.cols, h = imgLabels.rows;

                for (int r = chunksSizeAndLabels[0]; r < h; r = chunksSizeAndLabels[r]){

                    LabelT * const imgLabels_row = imgLabels.ptr<LabelT>(r);
                    LabelT * const  imgLabels_row_prev_prev = (LabelT *)(((char *)imgLabels_row) - imgLabels.step.p[0] - imgLabels.step.p[0]);
                    const PixelT * const img_row = img.ptr<PixelT>(r);
                    const PixelT * const img_row_prev = (PixelT *)(((char *)img_row) - img.step.p[0]);

                    for (int c = 0; c < w; c += 2){

#define condition_x imgLabels_row[c] > 0
#define condition_pppr c > 1 && imgLabels_row_prev_prev[c - 2] > 0
#define condition_qppr imgLabels_row_prev_prev[c] > 0
#define condition_qppr1 c < w - 1
#define condition_qppr2 c < w
#define condition_rppr c < w - 2 && imgLabels_row_prev_prev[c + 2] > 0

                        if (condition_x){
                            if (condition_pppr){
                                //check in img
                                if (img_row[c] > 0 && img_row_prev[c - 1] > 0)
                                    //assign the same label
                                    imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c - 2], imgLabels_row[c]);
                            }
                            if (condition_qppr){
                                if (condition_qppr1){
                                    if ((img_row[c] > 0 && img_row_prev[c] > 0) || (img_row[c + 1] > 0 && img_row_prev[c] > 0) ||
                                        (img_row[c] > 0 && img_row_prev[c + 1] > 0) || (img_row[c + 1] > 0 && img_row_prev[c + 1] > 0)){
                                        imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row[c]);
                                    }
                                }
                                else /*if (condition_qppr2)*/{
                                    if (img_row[c] > 0 && img_row_prev[c] > 0)
                                        imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row[c]);
                                }
                            }
                            if (condition_rppr){
                                if (img_row[c + 1] > 0 && img_row_prev[c + 2] > 0)
                                    imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c + 2], imgLabels_row[c]);
                            }
                        }
                    }
                }
            }

        LabelT operator()(const cv::Mat& img, cv::Mat& imgLabels, int connectivity, StatsOp& sop){
            CV_Assert(img.rows == imgLabels.rows);
            CV_Assert(img.cols == imgLabels.cols);
            CV_Assert(connectivity == 8);

            const int nThreads = cv::getNumberOfCPUs();
            cv::setNumThreads(nThreads);

            const int h = img.rows;
            const int w = img.cols;

            //A quick and dirty upper bound for the maximimum number of labels.
            //Following formula comes from the fact that a 2x2 block in 8-connectivity case
            //can never have more than 1 new label and 1 label for background.
            //Worst case image example pattern:
            //1 0 1 0 1...
            //0 0 0 0 0...
            //1 0 1 0 1...
            //............
            const size_t Plength = size_t(((h + 1) / 2) * size_t((w + 1) / 2)) + 1;

            //Array used to store info and labeled pixel by each thread.
            //Different threads affect different memory location of chunksSizeAndLabels
            int *chunksSizeAndLabels = (int *)cv::fastMalloc(h * sizeof(int));

            //Tree of labels
            LabelT *P = (LabelT *)cv::fastMalloc(Plength * sizeof(LabelT));
            //First label is for background
            P[0] = 0;

            cv::Range range(0, h);

            //First scan, each thread works with chunk of img.rows/nThreads rows
            //e.g. 300 rows, 4 threads -> each chunks is composed of 75 rows
            cv::parallel_for_(range, FirstScan(img, imgLabels, P, chunksSizeAndLabels), nThreads);

            //merge labels of different chunks
            mergeLabels(img, imgLabels, P, chunksSizeAndLabels);

            LabelT nLabels = 1;
            for (int i = 0; i < h; i = chunksSizeAndLabels[i]){
                flattenL(P, LabelT((i + 1) / 2) * LabelT((w + 1) / 2) + 1, chunksSizeAndLabels[i + 1], nLabels);
            }

            //Array for statistics data
            StatsOp *sopArray = new StatsOp[h];
            sop.init(nLabels);

            //Second scan
            cv::parallel_for_(range, SecondScan(img, imgLabels, P, sop, sopArray, nLabels), nThreads);

            StatsOp::mergeStats(imgLabels, sopArray, sop, nLabels);
            sop.finish();

            delete[] sopArray;
            cv::fastFree(chunksSizeAndLabels);
            cv::fastFree(P);
            return nLabels;
        }
    };//End struct LabelingGranaParallel

    // Based on "Optimized  Block-based Connected Components Labeling with Decision Trees", Costantino Grana et al
    // Only for 8-connectivity
    template<typename LabelT, typename PixelT, typename StatsOp = NoOp >
    struct LabelingGrana{
        LabelT operator()(const cv::Mat& img, cv::Mat& imgLabels, int connectivity, StatsOp& sop){
            CV_Assert(img.rows == imgLabels.rows);
            CV_Assert(img.cols == imgLabels.cols);
            CV_Assert(connectivity == 8);

            const int h = img.rows;
            const int w = img.cols;

            //A quick and dirty upper bound for the maximimum number of labels.
            //Following formula comes from the fact that a 2x2 block in 8-connectivity case
            //can never have more than 1 new label and 1 label for background.
            //Worst case image example pattern:
            //1 0 1 0 1...
            //0 0 0 0 0...
            //1 0 1 0 1...
            //............
            const size_t Plength = size_t(((h + 1) / 2) * size_t((w + 1) / 2)) + 1;

            LabelT *P = (LabelT *)fastMalloc(sizeof(LabelT) *Plength);
            P[0] = 0;
            LabelT lunique = 1;

            // First scan
            for (int r = 0; r < h; r += 2) {
                // Get rows pointer
                const PixelT * const img_row = img.ptr<PixelT>(r);
                const PixelT * const img_row_prev = (PixelT *)(((char *)img_row) - img.step.p[0]);
                const PixelT * const img_row_prev_prev = (PixelT *)(((char *)img_row_prev) - img.step.p[0]);
                const PixelT * const img_row_fol = (PixelT *)(((char *)img_row) + img.step.p[0]);
                LabelT * const imgLabels_row = imgLabels.ptr<LabelT>(r);
                LabelT * const imgLabels_row_prev_prev = (LabelT *)(((char *)imgLabels_row) - imgLabels.step.p[0] - imgLabels.step.p[0]);
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
#define condition_d c+1<w&& r-2>=0 && img_row_prev_prev[c+1]>0
#define condition_e c+2<w  && r-1>=0 && img_row_prev[c-1]>0

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
                    for (int r = 0; r < imgLabels.rows; r += 2) {
                        // Get rows pointer
                        const PixelT * const img_row = img.ptr<PixelT>(r);
                        const PixelT * const img_row_fol = (PixelT *)(((char *)img_row) + img.step.p[0]);
                        LabelT * const imgLabels_row = imgLabels.ptr<LabelT>(r);
                        LabelT * const imgLabels_row_fol = (LabelT *)(((char *)imgLabels_row) + imgLabels.step.p[0]);

                        for (int c = 0; c < imgLabels.cols; c += 2) {
                            LabelT iLabel = imgLabels_row[c];
                            if (iLabel > 0) {
                                iLabel = P[iLabel];
                                if (img_row[c] > 0){
                                    imgLabels_row[c] = iLabel;
                                    sop(r, c, iLabel);
                                }
                                else{
                                    imgLabels_row[c] = 0;
                                    sop(r, c, 0);
                                }
                                if (c + 1 < imgLabels.cols) {
                                    if (img_row[c + 1] > 0){
                                        imgLabels_row[c + 1] = iLabel;
                                        sop(r, c + 1, iLabel);
                                    }
                                    else{
                                        imgLabels_row[c + 1] = 0;
                                        sop(r, c + 1, 0);
                                    }
                                    if (r + 1 < imgLabels.rows) {
                                        if (img_row_fol[c] > 0){
                                            imgLabels_row_fol[c] = iLabel;
                                            sop(r + 1, c, iLabel);
                                        }
                                        else{
                                            imgLabels_row_fol[c] = 0;
                                            sop(r + 1, c, 0);
                                        }
                                        if (img_row_fol[c + 1] > 0){
                                            imgLabels_row_fol[c + 1] = iLabel;
                                            sop(r + 1, c + 1, iLabel);
                                        }
                                        else{
                                            imgLabels_row_fol[c + 1] = 0;
                                            sop(r + 1, c + 1, 0);
                                        }
                                    }
                                }
                                else if (r + 1 < imgLabels.rows) {
                                    if (img_row_fol[c] > 0){
                                        imgLabels_row_fol[c] = iLabel;
                                        sop(r + 1, c, iLabel);
                                    }
                                    else{
                                        imgLabels_row_fol[c] = 0;
                                        sop(r + 1, c, 0);
                                    }
                                }
                            }
                            else {
                                imgLabels_row[c] = 0;
                                sop(r, c, 0);
                                if (c + 1 < imgLabels.cols) {
                                    imgLabels_row[c + 1] = 0;
                                    sop(r, c + 1, 0);
                                    if (r + 1 < imgLabels.rows) {
                                        imgLabels_row_fol[c] = 0;
                                        imgLabels_row_fol[c + 1] = 0;
                                        sop(r + 1, c, 0);
                                        sop(r + 1, c + 1, 0);
                                    }
                                }
                                else if (r + 1 < imgLabels.rows) {
                                    imgLabels_row_fol[c] = 0;
                                    sop(r + 1, c, 0);
                                }
                            }
                        }
                    }
                }//END Case 1
                else{
                    //Case 2: only rows odd
                    for (int r = 0; r < imgLabels.rows; r += 2) {
                        // Get rows pointer
                        const PixelT * const img_row = img.ptr<PixelT>(r);
                        const PixelT * const img_row_fol = (PixelT *)(((char *)img_row) + img.step.p[0]);
                        LabelT * const imgLabels_row = imgLabels.ptr<LabelT>(r);
                        LabelT * const imgLabels_row_fol = (LabelT *)(((char *)imgLabels_row) + imgLabels.step.p[0]);

                        for (int c = 0; c < imgLabels.cols; c += 2) {
                            LabelT iLabel = imgLabels_row[c];
                            if (iLabel > 0) {
                                iLabel = P[iLabel];
                                if (img_row[c] > 0){
                                    imgLabels_row[c] = iLabel;
                                    sop(r, c, iLabel);
                                }
                                else{
                                    imgLabels_row[c] = 0;
                                    sop(r, c, 0);
                                }
                                if (img_row[c + 1] > 0){
                                    imgLabels_row[c + 1] = iLabel;
                                    sop(r, c + 1, iLabel);
                                }
                                else{
                                    imgLabels_row[c + 1] = 0;
                                    sop(r, c + 1, 0);
                                }
                                if (r + 1 < imgLabels.rows) {
                                    if (img_row_fol[c] > 0){
                                        imgLabels_row_fol[c] = iLabel;
                                        sop(r + 1, c, iLabel);
                                    }
                                    else{
                                        imgLabels_row_fol[c] = 0;
                                        sop(r + 1, c, 0);
                                    }
                                    if (img_row_fol[c + 1] > 0){
                                        imgLabels_row_fol[c + 1] = iLabel;
                                        sop(r + 1, c + 1, iLabel);
                                    }
                                    else{
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
                                if (r + 1 < imgLabels.rows) {
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
                    for (int r = 0; r < imgLabels.rows; r += 2) {
                        // Get rows pointer
                        const PixelT * const img_row = img.ptr<PixelT>(r);
                        const PixelT * const img_row_fol = (PixelT *)(((char *)img_row) + img.step.p[0]);
                        LabelT * const imgLabels_row = imgLabels.ptr<LabelT>(r);
                        LabelT * const imgLabels_row_fol = (LabelT *)(((char *)imgLabels_row) + imgLabels.step.p[0]);

                        for (int c = 0; c < imgLabels.cols; c += 2) {
                            LabelT iLabel = imgLabels_row[c];
                            if (iLabel > 0) {
                                iLabel = P[iLabel];
                                if (img_row[c] > 0){
                                    imgLabels_row[c] = iLabel;
                                    sop(r, c, iLabel);
                                }
                                else{
                                    imgLabels_row[c] = 0;
                                    sop(r, c, 0);
                                }
                                if (img_row_fol[c] > 0){
                                    imgLabels_row_fol[c] = iLabel;
                                    sop(r + 1, c, iLabel);
                                }
                                else{
                                    imgLabels_row_fol[c] = 0;
                                    sop(r + 1, c, 0);
                                }
                                if (c + 1 < imgLabels.cols) {
                                    if (img_row[c + 1] > 0){
                                        imgLabels_row[c + 1] = iLabel;
                                        sop(r, c + 1, iLabel);
                                    }
                                    else{
                                        imgLabels_row[c + 1] = 0;
                                        sop(r, c + 1, 0);
                                    }
                                    if (img_row_fol[c + 1] > 0){
                                        imgLabels_row_fol[c + 1] = iLabel;
                                        sop(r + 1, c + 1, iLabel);
                                    }
                                    else{
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
                                if (c + 1 < imgLabels.cols) {
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
                        const PixelT * const img_row = img.ptr<PixelT>(r);
                        const PixelT * const img_row_fol = (PixelT *)(((char *)img_row) + img.step.p[0]);
                        LabelT * const imgLabels_row = imgLabels.ptr<LabelT>(r);
                        LabelT * const imgLabels_row_fol = (LabelT *)(((char *)imgLabels_row) + imgLabels.step.p[0]);

                        for (int c = 0; c < imgLabels.cols; c += 2) {
                            LabelT iLabel = imgLabels_row[c];
                            if (iLabel > 0) {
                                iLabel = P[iLabel];
                                if (img_row[c] > 0){
                                    imgLabels_row[c] = iLabel;
                                    sop(r, c, iLabel);
                                }
                                else{
                                    imgLabels_row[c] = 0;
                                    sop(r, c, 0);
                                }
                                if (img_row[c + 1] > 0){
                                    imgLabels_row[c + 1] = iLabel;
                                    sop(r, c + 1, iLabel);
                                }
                                else{
                                    imgLabels_row[c + 1] = 0;
                                    sop(r, c + 1, 0);
                                }
                                if (img_row_fol[c] > 0){
                                    imgLabels_row_fol[c] = iLabel;
                                    sop(r + 1, c, iLabel);
                                }
                                else{
                                    imgLabels_row_fol[c] = 0;
                                    sop(r + 1, c, 0);
                                }
                                if (img_row_fol[c + 1] > 0){
                                    imgLabels_row_fol[c + 1] = iLabel;
                                    sop(r + 1, c + 1, iLabel);
                                }
                                else{
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
    };//End struct LabelingGrana
    }//end namespace connectedcomponents

    //L's type must have an appropriate depth for the number of pixels in I
    template<typename StatsOp>
    static
    int connectedComponents_sub1(const cv::Mat& I, cv::Mat& L, int connectivity, int ccltype, StatsOp& sop){
        CV_Assert(L.channels() == 1 && I.channels() == 1);
        CV_Assert(connectivity == 8 || connectivity == 4);
        CV_Assert(ccltype == CCL_GRANA || ccltype == CCL_WU || ccltype == CCL_DEFAULT);

        int lDepth = L.depth();
        int iDepth = I.depth();
        const char *currentParallelFramework = cv::currentParallelFramework();
        const int numberOfCPUs = cv::getNumberOfCPUs();

        CV_Assert(iDepth == CV_8U || iDepth == CV_8S);

        //Run parallel labeling only if the rows of the image are at least twice the number returned by getNumberOfCPUs
        const bool is_parallel = currentParallelFramework != NULL && numberOfCPUs > 1 && L.rows / numberOfCPUs >= 2;

        if (ccltype == CCL_WU || connectivity == 4){
            // Wu algorithm is used
            using connectedcomponents::LabelingWu;
            using connectedcomponents::LabelingWuParallel;
            //warn if L's depth is not sufficient?
            if (lDepth == CV_8U){
                //Not supported yet
            }
            else if (lDepth == CV_16U){
                return (int)LabelingWu<ushort, uchar, StatsOp>()(I, L, connectivity, sop);
            }
            else if (lDepth == CV_32S){
                //note that signed types don't really make sense here and not being able to use unsigned matters for scientific projects
                //OpenCV: how should we proceed?  .at<T> typechecks in debug mode
                if (!is_parallel)
                    return (int)LabelingWu<int, uchar, StatsOp>()(I, L, connectivity, sop);
                else
                    return (int)LabelingWuParallel<int, uchar, StatsOp>()(I, L, connectivity, sop);
            }
        }
        else if ((ccltype == CCL_GRANA || ccltype == CCL_DEFAULT) && connectivity == 8){
            // Grana algorithm is used
            using connectedcomponents::LabelingGrana;
            using connectedcomponents::LabelingGranaParallel;
            //warn if L's depth is not sufficient?
            if (lDepth == CV_8U){
                //Not supported yet
            }
            else if (lDepth == CV_16U){
                return (int)LabelingGrana<ushort, uchar, StatsOp>()(I, L, connectivity, sop);
            }
            else if (lDepth == CV_32S){
                //note that signed types don't really make sense here and not being able to use unsigned matters for scientific projects
                //OpenCV: how should we proceed?  .at<T> typechecks in debug mode
                if (!is_parallel)
                    return (int)LabelingGrana<int, uchar, StatsOp>()(I, L, connectivity, sop);
                else
                    return (int)LabelingGranaParallel<int, uchar, StatsOp>()(I, L, connectivity, sop);
            }
        }

        CV_Error(CV_StsUnsupportedFormat, "unsupported label/image type");
        return -1;
    }

}

// Simple wrapper to ensure binary and source compatibility (ABI)
int cv::connectedComponents(InputArray img_, OutputArray _labels, int connectivity, int ltype){
    return cv::connectedComponents(img_, _labels, connectivity, ltype, CCL_DEFAULT);
}

int cv::connectedComponents(InputArray img_, OutputArray _labels, int connectivity, int ltype, int ccltype){
    const cv::Mat img = img_.getMat();
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
int cv::connectedComponentsWithStats(InputArray img_, OutputArray _labels, OutputArray statsv,
    OutputArray centroids, int connectivity, int ltype)
{
    return cv::connectedComponentsWithStats(img_, _labels, statsv, centroids, connectivity, ltype, CCL_DEFAULT);
}

int cv::connectedComponentsWithStats(InputArray img_, OutputArray _labels, OutputArray statsv,
    OutputArray centroids, int connectivity, int ltype, int ccltype)
{
    const cv::Mat img = img_.getMat();
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
