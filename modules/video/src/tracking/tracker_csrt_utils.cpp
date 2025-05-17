// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"

#include "tracker_csrt_utils.hpp"

namespace cv {

Mat circshift(Mat matrix, int dx, int dy)
{
    Mat matrix_out = matrix.clone();
    int idx_y = 0;
    int idx_x = 0;
    for(int i=0; i<matrix.rows; i++) {
        for(int j=0; j<matrix.cols; j++) {
            idx_y = modul(i+dy+1, matrix.rows);
            idx_x = modul(j+dx+1, matrix.cols);
            matrix_out.at<float>(idx_y, idx_x) = matrix.at<float>(i,j);
        }
    }
    return matrix_out;
}

Mat gaussian_shaped_labels(const float sigma, const int w, const int h)
{
    // create 2D Gaussian peak, convert to Fourier space and stores it into the yf
    Mat y = Mat::zeros(h, w, CV_32F);
    float w2 = static_cast<float>(cvFloor(w / 2));
    float h2 = static_cast<float>(cvFloor(h / 2));

    // calculate for each pixel separatelly
    for(int i=0; i<y.rows; i++) {
        for(int j=0; j<y.cols; j++) {
            y.at<float>(i,j) = (float)exp((-0.5 / pow(sigma, 2)) * (pow((i+1-h2), 2) + pow((j+1-w2), 2)));
        }
    }
    // wrap-around with the circulat shifting
    y = circshift(y, -cvFloor(y.cols / 2), -cvFloor(y.rows / 2));
    Mat yf;
    dft(y, yf, DFT_COMPLEX_OUTPUT);
    return yf;
}

std::vector<Mat> fourier_transform_features(const std::vector<Mat> &M)
{
    std::vector<Mat> out(M.size());
    Mat channel;
    // iterate over channels and convert them to Fourier domain
    for(size_t k = 0; k < M.size(); k++) {
        M[k].convertTo(channel, CV_32F);
        dft(channel, channel, DFT_COMPLEX_OUTPUT);
        out[k] = (channel);
    }
    return out;
}

Mat divide_complex_matrices(const Mat &A, const Mat &B)
{
    std::vector<Mat> va,vb;
    split(A, va);
    split(B, vb);

    Mat a = va.at(0);
    Mat b = va.at(1);
    Mat c = vb.at(0);
    Mat d = vb.at(1);

    Mat div = c.mul(c) + d.mul(d);
    Mat real_part = (a.mul(c) + b.mul(d));
    Mat im_part = (b.mul(c) - a.mul(d));
    divide(real_part, div, real_part);
    divide(im_part, div, im_part);

    std::vector<Mat> tmp(2);
    tmp[0] = real_part;
    tmp[1] = im_part;
    Mat res;
    merge(tmp, res);
    return res;
}

Mat get_subwindow(
        const Mat &image,
        const Point2f center,
        const int w,
        const int h,
        Rect *valid_pixels)
{
    int startx = cvFloor(center.x) + 1 - (cvFloor(w/2));
    int starty = cvFloor(center.y) + 1 - (cvFloor(h/2));
    Rect roi(startx, starty, w, h);
    int padding_left = 0, padding_right = 0, padding_top = 0, padding_bottom = 0;
    if(roi.x < 0) {
        padding_left = -roi.x;
        roi.x = 0;
    }
    if(roi.y < 0) {
        padding_top = -roi.y;
        roi.y = 0;
    }
    roi.width -= padding_left;
    roi.height-= padding_top;
    if(roi.x + roi.width >= image.cols) {
        padding_right = roi.x + roi.width - image.cols;
        roi.width = image.cols - roi.x;
    }
    if(roi.y + roi.height >= image.rows) {
        padding_bottom = roi.y + roi.height - image.rows;
        roi.height = image.rows - roi.y;
    }
    Mat subwin = image(roi).clone();
    copyMakeBorder(subwin, subwin, padding_top, padding_bottom, padding_left, padding_right, BORDER_REPLICATE);

    if(valid_pixels != NULL) {
        *valid_pixels = Rect(padding_left, padding_top, roi.width, roi.height);
    }
    return subwin;
}

float subpixel_peak(const Mat &response, const std::string &s, const Point2f &p)
{
    int i_p0, i_p_l, i_p_r;     // indexes in response
    float p0, p_l, p_r;         // values in response

    if(s.compare("vertical") == 0) {
        // neighbouring rows
        i_p0 = cvRound(p.y);
        i_p_l = modul(cvRound(p.y) - 1, response.rows);
        i_p_r = modul(cvRound(p.y) + 1, response.rows);
        int px = static_cast<int>(p.x);
        p0 = response.at<float>(i_p0, px);
        p_l = response.at<float>(i_p_l, px);
        p_r = response.at<float>(i_p_r, px);
    } else if(s.compare("horizontal") == 0) {
        // neighbouring cols
        i_p0 = cvRound(p.x);
        i_p_l = modul(cvRound(p.x) - 1, response.cols);
        i_p_r = modul(cvRound(p.x) + 1, response.cols);
        int py = static_cast<int>(p.y);
        p0 = response.at<float>(py, i_p0);
        p_l = response.at<float>(py, i_p_l);
        p_r = response.at<float>(py, i_p_r);
    } else {
        std::cout << "Warning: unknown subpixel peak direction!" << std::endl;
        return 0;
    }
    float delta = 0.5f * (p_r - p_l) / (2*p0 - p_r - p_l);
    if(!std::isfinite(delta)) {
        delta = 0;
    }

    return delta;
}

inline float chebpoly(const int n, const float x)
{
    float res;
    if (fabs(x) <= 1)
        res = cos(n*acos(x));
    else
        res = cosh(n*acosh(x));
    return res;
}

static Mat chebwin(int N, const float atten)
{
    Mat out(N , 1, CV_32FC1);
    int nn, i;
    float M, n, sum = 0, max=0;
    float tg = static_cast<float>(pow(10,atten/20.0f));  /* 1/r term [2], 10^gamma [2] */
    float x0 = cosh((1.0f/(N-1))*acosh(tg));
    M = (N-1)/2.0f;
    if(N%2==0)
        M = M + 0.5f; /* handle even length windows */
    for(nn=0; nn<(N/2+1); nn++) {
        n = nn-M;
        sum = 0;
        for(i=1; i<=M; i++){
            sum += chebpoly(N-1,x0*static_cast<float>(cos(CV_PI*i/N))) *
                static_cast<float>(cos(2.0f*n*CV_PI*i/N));
        }
        out.at<float>(nn,0) = tg + 2*sum;
        out.at<float>(N-nn-1,0) = out.at<float>(nn,0) ;
        if(out.at<float>(nn,0) > max)
            max = out.at<float>(nn,0);
    }
    for(nn=0; nn<N; nn++)
        out.at<float>(nn,0) /= max; /* normalize everything */

    return out;
}


static double modified_bessel(int order, double x)
{
    //  sum m=0:inf 1/(m! * Gamma(m + order + 1)) * (x/2)^(2m + order)
    const double eps = 1e-13;
    double result = 0;
    double m = 0;
    double gamma = 1.0;
    for(int i = 2; i <= order; ++i)
        gamma *= i;
    double term = pow(x,order) / (pow(2,order) * gamma);

    while(term  > eps * result) {
        result += term;
        //calculate new term in series
        ++m;
        term *= (x*x) / (4*m*(m+order));
    }
    return result;
}

Mat get_hann_win(Size sz)
{
    Mat hann_rows = Mat::ones(sz.height, 1, CV_32F);
    Mat hann_cols = Mat::ones(1, sz.width, CV_32F);
    int NN = sz.height - 1;
    if(NN != 0) {
        for (int i = 0; i < hann_rows.rows; ++i) {
            hann_rows.at<float>(i,0) = (float)(1.0/2.0 * (1.0 - cos(2*CV_PI*i/NN)));
        }
    }
    NN = sz.width - 1;
    if(NN != 0) {
        for (int i = 0; i < hann_cols.cols; ++i) {
            hann_cols.at<float>(0,i) = (float)(1.0/2.0 * (1.0 - cos(2*CV_PI*i/NN)));
        }
    }
    return hann_rows * hann_cols;
}

Mat get_kaiser_win(Size sz, float alpha)
{
    Mat kaiser_rows = Mat::ones(sz.height, 1, CV_32F);
    Mat kaiser_cols = Mat::ones(1, sz.width, CV_32F);

    int N = sz.height - 1;
    double shape = alpha;
    double den = 1.0 / modified_bessel(0, shape);

    for(int n = 0; n <= N; ++n) {
        double K = (2.0 * n * 1.0/N) - 1.0;
        double x = sqrt(1.0 - (K * K));
        kaiser_rows.at<float>(n,0) = static_cast<float>(modified_bessel(0, shape * x) * den);
    }

    N = sz.width - 1;
    for(int n = 0; n <= N; ++n) {
        double K = (2.0 * n * 1.0/N) - 1.0;
        double x = sqrt(1.0 - (K * K));
        kaiser_cols.at<float>(0,n) = static_cast<float>(modified_bessel(0, shape * x) * den);
    }

    return kaiser_rows * kaiser_cols;
}

Mat get_chebyshev_win(Size sz, float attenuation)
{
    Mat cheb_rows = chebwin(sz.height, attenuation);
    Mat cheb_cols = chebwin(sz.width, attenuation).t();
    return cheb_rows * cheb_cols;
}

static void computeHOG32D(const Mat &imageM, Mat &featM, const int sbin, const int pad_x, const int pad_y)
{
    const int dimHOG = 32;
    CV_Assert(pad_x >= 0);
    CV_Assert(pad_y >= 0);
    CV_Assert(imageM.channels() == 3);
    CV_Assert(imageM.depth() == CV_64F);

    // epsilon to avoid division by zero
    const double eps = 0.0001;
    // number of orientations
    const int numOrient = 18;
    // unit vectors to compute gradient orientation
    const double uu[9] = {1.000, 0.9397, 0.7660, 0.5000, 0.1736, -0.1736, -0.5000, -0.7660, -0.9397};
    const double vv[9] = {0.000, 0.3420, 0.6428, 0.8660, 0.9848,  0.9848,  0.8660,  0.6428,  0.3420};

    // image size
    const Size imageSize = imageM.size();
    // block size
    // int bW = cvRound((double)imageSize.width/(double)sbin);
    // int bH = cvRound((double)imageSize.height/(double)sbin);
    int bW = cvFloor((double)imageSize.width/(double)sbin);
    int bH = cvFloor((double)imageSize.height/(double)sbin);
    const Size blockSize(bW, bH);
    // size of HOG features
    int oW = max(blockSize.width-2, 0) + 2*pad_x;
    int oH = max(blockSize.height-2, 0) + 2*pad_y;
    Size outSize = Size(oW, oH);
    // size of visible
    const Size visible = blockSize*sbin;

    // initialize historgram, norm, output feature matrices
    Mat histM = Mat::zeros(Size(blockSize.width*numOrient, blockSize.height), CV_64F);
    Mat normM = Mat::zeros(Size(blockSize.width, blockSize.height), CV_64F);
    featM = Mat::zeros(Size(outSize.width*dimHOG, outSize.height), CV_64F);

    // get the stride of each matrix
    const size_t imStride = imageM.step1();
    const size_t histStride = histM.step1();
    const size_t normStride = normM.step1();
    const size_t featStride = featM.step1();

    // calculate the zero offset
    const double* im = imageM.ptr<double>(0);
    double* const hist = histM.ptr<double>(0);
    double* const norm = normM.ptr<double>(0);
    double* const feat = featM.ptr<double>(0);

    for (int y = 1; y < visible.height - 1; y++)
    {
        for (int x = 1; x < visible.width - 1; x++)
        {
            // OpenCV uses an interleaved format: BGR-BGR-BGR
            const double* s = im + 3*min(x, imageM.cols-2) + min(y, imageM.rows-2)*imStride;

            // blue image channel
            double dyb = *(s+imStride) - *(s-imStride);
            double dxb = *(s+3) - *(s-3);
            double vb = dxb*dxb + dyb*dyb;

            // green image channel
            s += 1;
            double dyg = *(s+imStride) - *(s-imStride);
            double dxg = *(s+3) - *(s-3);
            double vg = dxg*dxg + dyg*dyg;

            // red image channel
            s += 1;
            double dy = *(s+imStride) - *(s-imStride);
            double dx = *(s+3) - *(s-3);
            double v = dx*dx + dy*dy;

            // pick the channel with the strongest gradient
            if (vg > v) { v = vg; dx = dxg; dy = dyg; }
            if (vb > v) { v = vb; dx = dxb; dy = dyb; }

            // snap to one of the 18 orientations
            double best_dot = 0;
            int best_o = 0;
            for (int o = 0; o < (int)numOrient/2; o++)
            {
                double dot =  uu[o]*dx + vv[o]*dy;
                if (dot > best_dot)
                {
                    best_dot = dot;
                    best_o = o;
                }
                else if (-dot > best_dot)
                {
                    best_dot = -dot;
                    best_o = o + (int)(numOrient/2);
                }
            }

            // add to 4 historgrams around pixel using bilinear interpolation
            double yp =  ((double)y+0.5)/(double)sbin - 0.5;
            double xp =  ((double)x+0.5)/(double)sbin - 0.5;
            int iyp = (int)cvFloor(yp);
            int ixp = (int)cvFloor(xp);
            double vy0 = yp - iyp;
            double vx0 = xp - ixp;
            double vy1 = 1.0 - vy0;
            double vx1 = 1.0 - vx0;
            v = sqrt(v);

            // fill the value into the 4 neighborhood cells
            if (iyp >= 0 && ixp >= 0)
                *(hist + iyp*histStride + ixp*numOrient + best_o) += vy1*vx1*v;

            if (iyp >= 0 && ixp+1 < blockSize.width)
                *(hist + iyp*histStride + (ixp+1)*numOrient + best_o) += vx0*vy1*v;

            if (iyp+1 < blockSize.height && ixp >= 0)
                *(hist + (iyp+1)*histStride + ixp*numOrient + best_o) += vy0*vx1*v;

            if (iyp+1 < blockSize.height && ixp+1 < blockSize.width)
                *(hist + (iyp+1)*histStride + (ixp+1)*numOrient + best_o) += vy0*vx0*v;

        } // for y
    } // for x

    // compute the energy in each block by summing over orientation
    for (int y = 0; y < blockSize.height; y++)
    {
        const double* src = hist + y*histStride;
        double* dst = norm + y*normStride;
        double const* const dst_end = dst + blockSize.width;
        // for each cell
        while (dst < dst_end)
        {
            *dst = 0;
            for (int o = 0; o < (int)(numOrient/2); o++)
            {
                *dst += (*src + *(src + numOrient/2))*
                    (*src + *(src + numOrient/2));
                src++;
            }
            dst++;
            src += numOrient/2;
        }
    }

    // compute the features
    for (int y = pad_y; y < outSize.height - pad_y; y++)
    {
        for (int x = pad_x; x < outSize.width - pad_x; x++)
        {
            double* dst = feat + y*featStride + x*dimHOG;
            double* p, n1, n2, n3, n4;
            const double* src;

            p = norm + (y - pad_y + 1)*normStride + (x - pad_x + 1);
            n1 = 1.0f / sqrt(*p + *(p + 1) + *(p + normStride) + *(p + normStride + 1) + eps);
            p = norm + (y - pad_y)*normStride + (x - pad_x + 1);
            n2 = 1.0f / sqrt(*p + *(p + 1) + *(p + normStride) + *(p + normStride + 1) + eps);
            p = norm + (y- pad_y + 1)*normStride + x - pad_x;
            n3 = 1.0f / sqrt(*p + *(p + 1) + *(p + normStride) + *(p + normStride + 1) + eps);
            p = norm + (y - pad_y)*normStride + x - pad_x;
            n4 = 1.0f / sqrt(*p + *(p + 1) + *(p + normStride) + *(p + normStride + 1) + eps);

            double t1 = 0.0, t2 = 0.0, t3 = 0.0, t4 = 0.0;

            // contrast-sesitive features
            src = hist + (y - pad_y + 1)*histStride + (x - pad_x + 1)*numOrient;
            for (int o = 0; o < numOrient; o++)
            {
                double val = *src;
                double h1 = min(val*n1, 0.2);
                double h2 = min(val*n2, 0.2);
                double h3 = min(val*n3, 0.2);
                double h4 = min(val*n4, 0.2);
                *(dst++) = 0.5 * (h1 + h2 + h3 + h4);

                src++;
                t1 += h1;
                t2 += h2;
                t3 += h3;
                t4 += h4;
            }

            // contrast-insensitive features
            src =  hist + (y - pad_y + 1)*histStride + (x - pad_x + 1)*numOrient;
            for (int o = 0; o < numOrient/2; o++)
            {
                double sum = *src + *(src + numOrient/2);
                double h1 = min(sum * n1, 0.2);
                double h2 = min(sum * n2, 0.2);
                double h3 = min(sum * n3, 0.2);
                double h4 = min(sum * n4, 0.2);
                *(dst++) = 0.5 * (h1 + h2 + h3 + h4);
                src++;
            }

            // texture features
            *(dst++) = 0.2357 * t1;
            *(dst++) = 0.2357 * t2;
            *(dst++) = 0.2357 * t3;
            *(dst++) = 0.2357 * t4;
            // truncation feature
            *dst = 0;
        }// for x
    }// for y
    // Truncation features
    for (int m = 0; m < featM.rows; m++)
    {
        for (int n = 0; n < featM.cols; n += dimHOG)
        {
            if (m > pad_y - 1 && m < featM.rows - pad_y && n > pad_x*dimHOG - 1 && n < featM.cols - pad_x*dimHOG)
                continue;

            featM.at<double>(m, n + dimHOG - 1) = 1;
        } // for x
    }// for y
}

std::vector<Mat> get_features_hog(const Mat &im, const int bin_size)
{
    Mat hogmatrix;
    Mat im_;
    im.convertTo(im_, CV_64FC3, 1.0/255.0);
    computeHOG32D(im_,hogmatrix,bin_size,1,1);
    hogmatrix.convertTo(hogmatrix, CV_32F);
    Size hog_size = im.size();
    hog_size.width /= bin_size;
    hog_size.height /= bin_size;
    Mat hogc(hog_size, CV_32FC(32), hogmatrix.data);
    std::vector<Mat> features;
    split(hogc, features);
    return features;
}

// std::vector<Mat> get_features_cn(const Mat &ppatch_data, const Size &output_size) {
//     Mat patch_data = ppatch_data.clone();
//     Vec3b & pixel = patch_data.at<Vec3b>(0,0);
//     unsigned index;

//     Mat cnFeatures = Mat::zeros(patch_data.rows,patch_data.cols,CV_32FC(10));

//     for(int i=0;i<patch_data.rows;i++){
//         for(int j=0;j<patch_data.cols;j++){
//             pixel=patch_data.at<Vec3b>(i,j);
//             index=(unsigned)(cvFloor((float)pixel[2]/8)+32*cvFloor((float)pixel[1]/8)+32*32*cvFloor((float)pixel[0]/8));

//             //copy the values
//             for(int k=0;k<10;k++){
//                 cnFeatures.at<Vec<float,10> >(i,j)[k]=(float)ColorNames[index][k];
//             }
//         }
//     }
//     std::vector<Mat> result;
//     split(cnFeatures, result);
//     for (size_t i = 0; i < result.size(); i++) {
//         if (output_size.width > 0 && output_size.height > 0) {
//             resize(result.at(i), result.at(i), output_size, INTER_CUBIC);
//         }
//     }
//     return result;
// }

std::vector<Mat> get_features_rgb(const Mat &patch, const Size &output_size)
{
    std::vector<Mat> channels;
    split(patch, channels);
    for(size_t k=0; k<channels.size(); k++) {
        channels[k].convertTo(channels[k], CV_32F, 1.0/255.0, -0.5);
        channels[k] = channels[k] - mean(channels[k])[0];
        resize(channels[k], channels[k], output_size, INTER_CUBIC);
    }
    return channels;
}

double get_max(const Mat &m)
{
    double val;
    minMaxLoc(m, NULL, &val, NULL, NULL);
    return val;
}

double get_min(const Mat &m)
{
    double val;
    minMaxLoc(m, &val, NULL, NULL, NULL);
    return val;
}

Mat bgr2hsv(const Mat &img)
{
    Mat hsv_img;
    cvtColor(img, hsv_img, COLOR_BGR2HSV);
    std::vector<Mat> hsv_img_channels;
    split(hsv_img, hsv_img_channels);
    hsv_img_channels.at(0).convertTo(hsv_img_channels.at(0), CV_8UC1, 255.0 / 180.0);
    merge(hsv_img_channels, hsv_img);
    return hsv_img;
}

} //cv namespace
