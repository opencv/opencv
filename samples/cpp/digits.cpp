#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/ml.hpp"

#include <algorithm>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

const int SZ = 20;  // size of each digit is SZ x SZ
const int CLASS_N = 10;
const char* DIGITS_FN = "digits.png";

static void help()
{
    cout <<
    "\n"
    "SVM and KNearest digit recognition.\n"
    "\n"
    "Sample loads a dataset of handwritten digits from 'digits.png'.\n"
    "Then it trains a SVM and KNearest classifiers on it and evaluates\n"
    "their accuracy.\n"
    "\n"
    "Following preprocessing is applied to the dataset:\n"
    " - Moment-based image deskew (see deskew())\n"
    " - Digit images are split into 4 10x10 cells and 16-bin\n"
    "   histogram of oriented gradients is computed for each\n"
    "   cell\n"
    " - Transform histograms to space with Hellinger metric (see [1] (RootSIFT))\n"
    "\n"
    "\n"
    "[1] R. Arandjelovic, A. Zisserman\n"
    "    \"Three things everyone should know to improve object retrieval\"\n"
    "    http://www.robots.ox.ac.uk/~vgg/publications/2012/Arandjelovic12/arandjelovic12.pdf\n"
    "\n"
    "Usage:\n"
    "   ./digits\n" << endl;
}

static void split2d(const Mat& image, const Size cell_size, vector<Mat>& cells)
{
    int height = image.rows;
    int width = image.cols;

    int sx = cell_size.width;
    int sy = cell_size.height;

    cells.clear();

    for (int i = 0; i < height; i += sy)
    {
        for (int j = 0; j < width; j += sx)
        {
            cells.push_back(image(Rect(j, i, sx, sy)));
        }
    }
}

static void load_digits(const char* fn, vector<Mat>& digits, vector<int>& labels)
{
    digits.clear();
    labels.clear();

    String filename = samples::findFile(fn);

    cout << "Loading " << filename << " ..." << endl;

    Mat digits_img = imread(filename, IMREAD_GRAYSCALE);
    split2d(digits_img, Size(SZ, SZ), digits);

    for (int i = 0; i < CLASS_N; i++)
    {
        for (size_t j = 0; j < digits.size() / CLASS_N; j++)
        {
            labels.push_back(i);
        }
    }
}

static void deskew(const Mat& img, Mat& deskewed_img)
{
    Moments m = moments(img);

    if (abs(m.mu02) < 0.01)
    {
        deskewed_img = img.clone();
        return;
    }

    float skew = (float)(m.mu11 / m.mu02);
    float M_vals[2][3] = {{1, skew, -0.5f * SZ * skew}, {0, 1, 0}};
    Mat M(Size(3, 2), CV_32F);

    for (int i = 0; i < M.rows; i++)
    {
        for (int j = 0; j < M.cols; j++)
        {
            M.at<float>(i, j) = M_vals[i][j];
        }
    }

    warpAffine(img, deskewed_img, M, Size(SZ, SZ), WARP_INVERSE_MAP | INTER_LINEAR);
}

static void mosaic(const int width, const vector<Mat>& images, Mat& grid)
{
    int mat_width = SZ * width;
    int mat_height = SZ * (int)ceil((double)images.size() / width);

    if (!images.empty())
    {
        grid = Mat(Size(mat_width, mat_height), images[0].type());

        for (size_t i = 0; i < images.size(); i++)
        {
            Mat location_on_grid = grid(Rect(SZ * ((int)i % width), SZ * ((int)i / width), SZ, SZ));
            images[i].copyTo(location_on_grid);
        }
    }
}

static void evaluate_model(const vector<float>& predictions, const vector<Mat>& digits, const vector<int>& labels, Mat& mos)
{
    double err = 0;

    for (size_t i = 0; i < predictions.size(); i++)
    {
        if ((int)predictions[i] != labels[i])
        {
            err++;
        }
    }

    err /= predictions.size();

    cout << format("error: %.2f %%", err * 100) << endl;

    int confusion[10][10] = {};

    for (size_t i = 0; i < labels.size(); i++)
    {
        confusion[labels[i]][(int)predictions[i]]++;
    }

    cout << "confusion matrix:" << endl;
    for (int i = 0; i < 10; i++)
    {
        for (int j = 0; j < 10; j++)
        {
            cout << format("%2d ", confusion[i][j]);
        }
        cout << endl;
    }

    cout << endl;

    vector<Mat> vis;

    for (size_t i = 0; i < digits.size(); i++)
    {
        Mat img;
        cvtColor(digits[i], img, COLOR_GRAY2BGR);

        if ((int)predictions[i] != labels[i])
        {
            for (int j = 0; j < img.rows; j++)
            {
                for (int k = 0; k < img.cols; k++)
                {
                    img.at<Vec3b>(j, k)[0] = 0;
                    img.at<Vec3b>(j, k)[1] = 0;
                }
            }
        }

        vis.push_back(img);
    }

    mosaic(25, vis, mos);
}

static void bincount(const Mat& x, const Mat& weights, const int min_length, vector<double>& bins)
{
    double max_x_val = 0;
    minMaxLoc(x, NULL, &max_x_val);

    bins = vector<double>(max((int)max_x_val, min_length));

    for (int i = 0; i < x.rows; i++)
    {
        for (int j = 0; j < x.cols; j++)
        {
            bins[x.at<int>(i, j)] += weights.at<float>(i, j);
        }
    }
}

static void preprocess_hog(const vector<Mat>& digits, Mat& hog)
{
    int bin_n = 16;
    int half_cell = SZ / 2;
    double eps = 1e-7;

    hog = Mat(Size(4 * bin_n, (int)digits.size()), CV_32F);

    for (size_t img_index = 0; img_index < digits.size(); img_index++)
    {
        Mat gx;
        Sobel(digits[img_index], gx, CV_32F, 1, 0);

        Mat gy;
        Sobel(digits[img_index], gy, CV_32F, 0, 1);

        Mat mag;
        Mat ang;
        cartToPolar(gx, gy, mag, ang);

        Mat bin(ang.size(), CV_32S);

        for (int i = 0; i < ang.rows; i++)
        {
            for (int j = 0; j < ang.cols; j++)
            {
                bin.at<int>(i, j) = (int)(bin_n * ang.at<float>(i, j) / (2 * CV_PI));
            }
        }

        Mat bin_cells[] = {
            bin(Rect(0, 0, half_cell, half_cell)),
            bin(Rect(half_cell, 0, half_cell, half_cell)),
            bin(Rect(0, half_cell, half_cell, half_cell)),
            bin(Rect(half_cell, half_cell, half_cell, half_cell))
        };
        Mat mag_cells[] = {
            mag(Rect(0, 0, half_cell, half_cell)),
            mag(Rect(half_cell, 0, half_cell, half_cell)),
            mag(Rect(0, half_cell, half_cell, half_cell)),
            mag(Rect(half_cell, half_cell, half_cell, half_cell))
        };

        vector<double> hist;
        hist.reserve(4 * bin_n);

        for (int i = 0; i < 4; i++)
        {
            vector<double> partial_hist;
            bincount(bin_cells[i], mag_cells[i], bin_n, partial_hist);
            hist.insert(hist.end(), partial_hist.begin(), partial_hist.end());
        }

        // transform to Hellinger kernel
        double sum = 0;

        for (size_t i = 0; i < hist.size(); i++)
        {
            sum += hist[i];
        }

        for (size_t i = 0; i < hist.size(); i++)
        {
            hist[i] /= sum + eps;
            hist[i] = sqrt(hist[i]);
        }

        double hist_norm = norm(hist);

        for (size_t i = 0; i < hist.size(); i++)
        {
            hog.at<float>((int)img_index, (int)i) = (float)(hist[i] / (hist_norm + eps));
        }
    }
}

static void shuffle(vector<Mat>& digits, vector<int>& labels)
{
    vector<int> shuffled_indexes(digits.size());

    for (size_t i = 0; i < digits.size(); i++)
    {
        shuffled_indexes[i] = (int)i;
    }

    randShuffle(shuffled_indexes);

    vector<Mat> shuffled_digits(digits.size());
    vector<int> shuffled_labels(labels.size());

    for (size_t i = 0; i < shuffled_indexes.size(); i++)
    {
        shuffled_digits[shuffled_indexes[i]] = digits[i];
        shuffled_labels[shuffled_indexes[i]] = labels[i];
    }

    digits = shuffled_digits;
    labels = shuffled_labels;
}

int main()
{
    help();

    vector<Mat> digits;
    vector<int> labels;

    load_digits(DIGITS_FN, digits, labels);

    cout << "preprocessing..." << endl;

    // shuffle digits
    shuffle(digits, labels);

    vector<Mat> digits2;

    for (size_t i = 0; i < digits.size(); i++)
    {
        Mat deskewed_digit;
        deskew(digits[i], deskewed_digit);
        digits2.push_back(deskewed_digit);
    }

    Mat samples;

    preprocess_hog(digits2, samples);

    int train_n = (int)(0.9 * samples.rows);
    Mat test_set;

    vector<Mat> digits_test(digits2.begin() + train_n, digits2.end());
    mosaic(25, digits_test, test_set);
    imshow("test set", test_set);

    Mat samples_train = samples(Rect(0, 0, samples.cols, train_n));
    Mat samples_test = samples(Rect(0, train_n, samples.cols, samples.rows - train_n));
    vector<int> labels_train(labels.begin(), labels.begin() + train_n);
    vector<int> labels_test(labels.begin() + train_n, labels.end());

    Ptr<ml::KNearest> k_nearest;
    Ptr<ml::SVM> svm;
    vector<float> predictions;
    Mat vis;

    cout << "training KNearest..." << endl;
    k_nearest = ml::KNearest::create();
    k_nearest->train(samples_train, ml::ROW_SAMPLE, labels_train);

    // predict digits with KNearest
    k_nearest->findNearest(samples_test, 4, predictions);
    evaluate_model(predictions, digits_test, labels_test, vis);
    imshow("KNearest test", vis);
    k_nearest.release();

    cout << "training SVM..." << endl;
    svm = ml::SVM::create();
    svm->setGamma(5.383);
    svm->setC(2.67);
    svm->setKernel(ml::SVM::RBF);
    svm->setType(ml::SVM::C_SVC);
    svm->train(samples_train, ml::ROW_SAMPLE, labels_train);

    // predict digits with SVM
    svm->predict(samples_test, predictions);
    evaluate_model(predictions, digits_test, labels_test, vis);
    imshow("SVM test", vis);
    cout << "Saving SVM as \"digits_svm.yml\"..." << endl;
    svm->save("digits_svm.yml");
    svm.release();

    waitKey();

    return 0;
}
