#include "precomp.hpp"
#undef ALEX_DEBUG
#include "debug.hpp"
#include <vector>
#include <algorithm>

#define ABSCLIP(val,threshold) MIN(MAX((val),-(threshold)),(threshold))

namespace cv{namespace optim{

    class AddFloatToCharScaled{
        public:
            AddFloatToCharScaled(float scale):_scale(scale){}
            inline float operator()(float a,uchar b){
                return a+_scale*((float)b);
            }
        private:
            float _scale;
    };

    void denoise_TVL1(const std::vector<Mat>& observations,Mat& result, double lambda, int niters){

        CV_Assert(observations.size()>0 && niters>0 && lambda>0);

        const float L2 = 8.0f, tau = 0.02f, sigma = 1./(L2*tau), theta = 1.f;
        float clambda = (float)lambda;
        float s=0;
        const int workdepth = CV_32F;

        int i, x, y, rows=observations[0].rows, cols=observations[0].cols,count;
        for(i=1;i<(int)observations.size();i++){
            CV_Assert(observations[i].rows==rows && observations[i].cols==cols);
        }

        Mat X, P = Mat::zeros(rows, cols, CV_MAKETYPE(workdepth, 2));
        observations[0].convertTo(X, workdepth, 1./255);
        std::vector< Mat_<float> > Rs(observations.size());
        for(count=0;count<(int)Rs.size();count++){
            Rs[count]=Mat::zeros(rows,cols,workdepth);
        }

        for( i = 0; i < niters; i++ )
        {
            float currsigma = i == 0 ? 1 + sigma : sigma;

            // P_ = P + sigma*nabla(X)
            // P(x,y) = P_(x,y)/max(||P(x,y)||,1)
            for( y = 0; y < rows; y++ )
            {
                const float* x_curr = X.ptr<float>(y);
                const float* x_next = X.ptr<float>(std::min(y+1, rows-1));
                Point2f* p_curr = P.ptr<Point2f>(y);
                float dx, dy, m;
                for( x = 0; x < cols-1; x++ )
                {
                    dx = (x_curr[x+1] - x_curr[x])*currsigma + p_curr[x].x;
                    dy = (x_next[x] - x_curr[x])*currsigma + p_curr[x].y;
                    m = 1.f/std::max(std::sqrt(dx*dx + dy*dy), 1.f);
                    p_curr[x].x = dx*m;
                    p_curr[x].y = dy*m;
                }
                dy = (x_next[x] - x_curr[x])*currsigma + p_curr[x].y;
                m = 1.f/std::max(std::abs(dy), 1.f);
                p_curr[x].x = 0.f;
                p_curr[x].y = dy*m;
            }


            //Rs = clip(Rs + sigma*(X-imgs), -clambda, clambda)
            for(count=0;count<(int)Rs.size();count++){
                std::transform<MatIterator_<float>,MatConstIterator_<uchar>,MatIterator_<float>,AddFloatToCharScaled>(
                        Rs[count].begin(),Rs[count].end(),observations[count].begin<uchar>(),
                        Rs[count].begin(),AddFloatToCharScaled(-sigma/255.0));
                Rs[count]+=sigma*X;
                min(Rs[count],clambda,Rs[count]);
                max(Rs[count],-clambda,Rs[count]);
            }

            for( y = 0; y < rows; y++ )
            {
                float* x_curr = X.ptr<float>(y);
                const Point2f* p_curr = P.ptr<Point2f>(y);
                const Point2f* p_prev = P.ptr<Point2f>(std::max(y - 1, 0));

                // X1 = X + tau*(-nablaT(P))
                x = 0;
                s=0.0;
                for(count=0;count<(int)Rs.size();count++){
                    s=s+Rs[count](y,x);
                }
                float x_new = x_curr[x] + tau*(p_curr[x].y - p_prev[x].y)-tau*s;
                    // X = X2 + theta*(X2 - X)
                x_curr[x] = x_new + theta*(x_new - x_curr[x]);


                for(x = 1; x < cols; x++ )
                {
                    s=0.0;
                    for(count=0;count<(int)Rs.size();count++){
                        s+=Rs[count](y,x);
                    }
                        // X1 = X + tau*(-nablaT(P))
                    x_new = x_curr[x] + tau*(p_curr[x].x - p_curr[x-1].x + p_curr[x].y - p_prev[x].y)-tau*s;
                        // X = X2 + theta*(X2 - X)
                    x_curr[x] = x_new + theta*(x_new - x_curr[x]);
                }
            }
        }

        result.create(X.rows,X.cols,CV_8U);
        X.convertTo(result, CV_8U, 255);
    }
}}
