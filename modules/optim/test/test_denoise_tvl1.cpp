#include "test_precomp.hpp"
#include "opencv2/highgui.hpp"

void make_noisy(const cv::Mat& img, cv::Mat& noisy, double sigma, double pepper_salt_ratio,cv::RNG& rng){
    noisy.create(img.size(), img.type());
    cv::Mat noise(img.size(), img.type()), mask(img.size(), CV_8U);
    rng.fill(noise,cv::RNG::NORMAL,128.0,sigma);
    cv::addWeighted(img, 1, noise, 1, -128, noisy);
    cv::randn(noise, cv::Scalar::all(0), cv::Scalar::all(2));
    noise *= 255;
    cv::randu(mask, 0, cvRound(1./pepper_salt_ratio));
    cv::Mat half = mask.colRange(0, img.cols/2);
    half = cv::Scalar::all(1);
    noise.setTo(128, mask);
    cv::addWeighted(noisy, 1, noise, 1, -128, noisy);
}
void make_spotty(cv::Mat& img,cv::RNG& rng, int r=3,int n=1000){
    for(int i=0;i<n;i++){
        int x=rng(img.cols-r),y=rng(img.rows-r);
        if(rng(2)==0){
            img(cv::Range(y,y+r),cv::Range(x,x+r))=(uchar)0;
        }else{
            img(cv::Range(y,y+r),cv::Range(x,x+r))=(uchar)255;
        }
    }
}

bool validate_pixel(const cv::Mat& image,int x,int y,uchar val){
    printf("test: image(%d,%d)=%d vs %d - %s\n",x,y,(int)image.at<uchar>(x,y),val,(val==image.at<uchar>(x,y))?"true":"false");
    return (image.at<uchar>(x,y)==val);
}

TEST(Optim_denoise_tvl1, regression_basic){
    cv::RNG rng(42);
    cv::Mat img = cv::imread("lena.jpg", 0), noisy,res;
    if(img.rows!=512 || img.cols!=512){
        printf("\tplease, put lena.jpg from samples/c in the current folder\n");
        printf("\tnow, the test will fail...\n");
        ASSERT_TRUE(false);
    }

    const int obs_num=5;
    std::vector<cv::Mat> images(obs_num,cv::Mat());
    for(int i=0;i<(int)images.size();i++){
        make_noisy(img,images[i], 20, 0.02,rng);
        //make_spotty(images[i],rng);
    }

    //cv::imshow("test", images[0]);
    cv::optim::denoise_TVL1(images, res);
    //cv::imshow("denoised", res);
    //cv::waitKey();

#if 0
    ASSERT_TRUE(validate_pixel(res,248,334,179));
    ASSERT_TRUE(validate_pixel(res,489,333,172));
    ASSERT_TRUE(validate_pixel(res,425,507,104));
    ASSERT_TRUE(validate_pixel(res,489,486,105));
    ASSERT_TRUE(validate_pixel(res,223,208,64));
    ASSERT_TRUE(validate_pixel(res,418,3,78));
    ASSERT_TRUE(validate_pixel(res,63,76,97));
    ASSERT_TRUE(validate_pixel(res,29,134,126));
    ASSERT_TRUE(validate_pixel(res,219,291,174));
    ASSERT_TRUE(validate_pixel(res,384,124,76));
#endif

#if 1
    ASSERT_TRUE(validate_pixel(res,248,334,194));
    ASSERT_TRUE(validate_pixel(res,489,333,171));
    ASSERT_TRUE(validate_pixel(res,425,507,103));
    ASSERT_TRUE(validate_pixel(res,489,486,109));
    ASSERT_TRUE(validate_pixel(res,223,208,72));
    ASSERT_TRUE(validate_pixel(res,418,3,58));
    ASSERT_TRUE(validate_pixel(res,63,76,93));
    ASSERT_TRUE(validate_pixel(res,29,134,127));
    ASSERT_TRUE(validate_pixel(res,219,291,180));
    ASSERT_TRUE(validate_pixel(res,384,124,80));
#endif

}
