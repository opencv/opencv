#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>

using namespace cv;
using namespace std;

int main(int, char**)
{
    const int MAX_CLUSTERS = 5;
    Scalar colorTab[] =
    {
        Scalar(0, 0, 255),
        Scalar(0, 255, 0),
        Scalar(255, 100, 100),
        Scalar(255, 0, 255),
        Scalar(0, 255, 255)
    };

    Mat img(500, 500, CV_8UC3);
    RNG rng(12345);
    const int max_iterations = 10; // 设置最大迭代次数
    int iteration_count = 0;

    while(iteration_count < max_iterations)
    {
        int k, clusterCount = rng.uniform(2, MAX_CLUSTERS + 1);
        int i, sampleCount = rng.uniform(1, 1001);
        Mat points(sampleCount, 1, CV_32FC2), labels;

        clusterCount = MIN(clusterCount, sampleCount);
        std::vector<Point2f> centers;

        // 生成来自多高斯分布的随机样本
        for(k = 0; k < clusterCount; k++)
        {
            Point center;
            center.x = rng.uniform(0, img.cols);
            center.y = rng.uniform(0, img.rows);
            Mat pointChunk = points.rowRange(k * sampleCount / clusterCount,
                                             k == clusterCount - 1 ? sampleCount :
                                             (k + 1) * sampleCount / clusterCount);
            rng.fill(pointChunk, RNG::NORMAL, Scalar(center.x, center.y), Scalar(img.cols * 0.05, img.rows * 0.05));
        }

        randShuffle(points, 1, &rng);

        double compactness = kmeans(points, clusterCount, labels,
            TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 1.0),
            3, KMEANS_PP_CENTERS, centers);

        img = Scalar::all(0);

        for(i = 0; i < sampleCount; i++)
        {
            int clusterIdx = labels.at<int>(i);
            Point ipt = points.at<Point2f>(i);
            circle(img, ipt, 2, colorTab[clusterIdx], FILLED, LINE_AA);
        }
        for(i = 0; i < (int)centers.size(); ++i)
        {
            Point2f c = centers[i];
            circle(img, c, 40, colorTab[i], 1, LINE_AA);
        }
        cout << "Iteration: " << iteration_count + 1 << ", Compactness: " << compactness << endl;

        imwrite("clusters_" + to_string(iteration_count + 1) + ".png", img); // 保存图像

        iteration_count++;
    }

    return 0;
}

