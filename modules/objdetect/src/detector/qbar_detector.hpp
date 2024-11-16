#pragma once
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>
#include "../qbarstruct.hpp"
#include <memory>

namespace cv {
    typedef struct BoxInfo
    {
        float x1;
        float y1;
        float x2;
        float y2;
        float score;
        int label;
    } BoxInfo;

    struct DetectInfo
    {
        int class_id;
        float prob;
        int x, y, width, height;
        
        void PrintInfo()
        {
            printf("class %d, prob %.2f, x %d, y %d, width %d, height %d\n", class_id, prob, x, y, width, height);
        }
        
        bool operator<(const DetectInfo &other) const
        {
            return other.prob < prob;
        }
    };
    
    struct QBAR_POINT3
    {
        float a,b,c;
        QBAR_POINT3() {}
        QBAR_POINT3(float a_, float b_, float c_): a(a_), b(b_), c(c_) {}
    };
    struct QBAR_LINE
    {
        QBAR_POINT p1, p2;
        float length;
        QBAR_LINE() {}
        QBAR_LINE(QBAR_POINT p1_, QBAR_POINT p2_)
        {
            p1.x = p1_.x;
            p1.y = p1_.y;
            p2.x = p2_.x;
            p2.y = p2_.y;
            length = (p1.x-p2.x)*(p1.x-p2.x) + (p1.y-p2.y)*(p1.y-p2.y);
        }
        bool operator<(const QBAR_LINE &other) const
        {
            return other.length < length;
        }
    };
    
    struct QBarProcessPackage
    {
        int width, height;
        QBAR_READER readerId;
        int cutOffsetX, cutOffsetY;
        int detectX1, detectY1;
        int detectX2, detectY2;
        float prob;
        
        int width_1, height_1;
        int cutOffsetX_1, cutOffsetY_1;
        
        std::vector<float> scaleList;
    };

    class QBarDetector
    {
        public:
            QBarDetector(){};
            ~QBarDetector(){};
            int Init(const std::string &config_path);
            int Detect(const Mat &image,std::vector<DetectInfo> &bboxes);
            void setReferenceSize(int reference_size) {this->reference_size = reference_size;}
            void setScoreThres(float score_thres) {this->score_thres = score_thres;}
            void setIouThres(float iou_thres) {this->iou_thres = iou_thres;}
    
        private:
            int post_process_det(std::vector<Mat> outputs,float inputWidth,float inputHeight,std::vector<BoxInfo>& dets);
            int pre_process_det(const Mat &image,Mat &out_blob);
            void multiclass_nms(std::vector<BoxInfo> &input_boxes, std::vector<BoxInfo> &output_boxes, float thr, int inputWidth, int inputHeight);
            void decode_infer(float *clsPred, float *disPred, int stride, std::vector<std::vector<BoxInfo>> &results, const std::vector<int> &outShapeCls, const std::vector<int> &outShapeDis, float scoreThres,float inputHeight,float inputWidth);
            void nms(std::vector<BoxInfo>& input_boxes, float NMS_THRESH);
    
        private:
            std::shared_ptr<dnn::Net> qbar_detector;
            int reference_size;
            float score_thres, iou_thres;
    };
}  // namespace cv