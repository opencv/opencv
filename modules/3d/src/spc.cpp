// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Author: Zhangjie Chen <zhangjiec01@gmail.com>

#include "opencv2/core.hpp"
#include "opencv2/3d.hpp"
#include "queue"
// TODO: test only
#include "iostream"
#include <chrono>
#define ENABLE_MAT_OUTPUT 0

#define START_TIMER start = std::chrono::high_resolution_clock::now();
#define STOP_TIMER(msg) \
    stop = std::chrono::high_resolution_clock::now(); \
    std::cout << (msg) << ": " << (stop - start).count() / 1000000.f << " ms" << std::endl;

namespace cv{

// TODO: test only, remember to remove
template<typename T>
void showMat(const String &name, const Mat_<T>& mat) {
    if(ENABLE_MAT_OUTPUT) {
        std::cout << name << ": " << std::endl;
        for (int i = 0; i < mat.rows; ++i) {
            for (int j = 0; j < mat.cols; ++j) {
                std::cout << (float)mat(i, j) << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}

SpectralCluster::SpectralCluster(float delta_val, float eta_val) : delta(delta_val), eta(eta_val) {}


void SpectralCluster::cluster(std::vector<int>& result, std::vector<Point3f> vertices,
                              std::vector<std::vector<int32_t>> indices, int k) {
    auto start = std::chrono::high_resolution_clock::now(), stop = std::chrono::high_resolution_clock::now();
    // check parameters
    CV_Assert(k > 1);

    int num_faces = static_cast<int>(indices.size());

    // test
    Mat distance_mat;
    START_TIMER
    getDistanceMat(distance_mat, vertices, indices);
    STOP_TIMER("Distance Matrix Calculation")

    // for test:
    showMat<float>("distance matrix", distance_mat);

    Mat affinity_mat;
    START_TIMER
    getAffinityMat(distance_mat, affinity_mat, indices);
    STOP_TIMER("Affinity Matrix Calculation")

    // for test:
    showMat<float>("affinity matrix", affinity_mat);

    Mat laplacian_mat;
    START_TIMER
    getLaplacianMat(affinity_mat, laplacian_mat);
    STOP_TIMER("Laplacian Matrix Calculation")

    // for test:
    showMat<float>("Laplacian matrix", laplacian_mat);
    Mat eigen_values;
    Mat eigen_vectors;
    // eigenvalues are arranged from largest to smallest
    // each row is a corresponding eigenvector
    START_TIMER
    eigen(laplacian_mat, eigen_values, eigen_vectors);
    STOP_TIMER("Eigen Calculation")

    // get first k eigenvectors with largest eigenvalues
    eigen_vectors = eigen_vectors.rowRange(0, k);
    // transpose
    Mat eigen_vectors_t;
    eigen_vectors_t = eigen_vectors.t();
    // for test:
    showMat<float>("eigen vectors", eigen_vectors_t);

    // normalize each row
    for (int i = 0; i < eigen_vectors_t.rows; ++i)
        eigen_vectors_t.row(i) /= static_cast<float>(norm(eigen_vectors_t.row(i)));
    showMat<float>("normalized eigen vectors", eigen_vectors_t);

    result.clear();

    // clustering on the eigenvectors:
    START_TIMER
    cv::kmeans(eigen_vectors_t, k, result,
               cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 10, 1.0),
               5, cv::KMEANS_PP_CENTERS);
    STOP_TIMER("K-means")

    // cluster complete
    String s;
    for (int w = 0; w < num_faces; ++w) {
        s += std::to_string(result[w]);
        s += ',';
    }
    std::cout << s << std::endl;
}

void SpectralCluster::getLaplacianMat(Mat &in, Mat &out) {
    int num_faces = in.rows;

    Mat degree_mat;
    reduce(in, degree_mat, 1, REDUCE_SUM);

    degree_mat = 1.0f / degree_mat;

    sqrt(degree_mat, degree_mat);

    Mat degree_mat_sqrt;
    repeat(degree_mat, 1, num_faces, degree_mat_sqrt);

    out = ((in.mul(degree_mat_sqrt).t()).mul(degree_mat_sqrt)).t();
}

void SpectralCluster::getAffinityMat(Mat &in, Mat &out, std::vector<std::vector<int32_t>> &indices) {
    // in: distance matrix from previous step
    int num_faces = int(indices.size());

    // using dijkstra algorithm to get the affinity matrix
    out = Mat::ones(num_faces, num_faces, CV_32F) * (double)std::numeric_limits<float>::infinity();
    // for dijkstra
    struct temp_node {
        int index;
        float distance;
        temp_node(int i, float d) : index(i), distance(d) {}
        bool operator<(const temp_node& other) const {
            return distance > other.distance;
        }
    };

    auto time1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_faces; ++i) {
        std::vector<bool> visited(num_faces, false);
        std::priority_queue<temp_node> pq;

        pq.emplace(i, 0);

        while (!pq.empty()) {
            temp_node current = pq.top();
            pq.pop();

            if (visited[current.index])
                continue;

            visited[current.index] = true;

            for (int neighbor = 0; neighbor < num_faces; ++neighbor) {
                float weight = in.at<float>(current.index, neighbor);
                if (weight > 0 && !visited[neighbor]) {
                    float distance = current.distance + weight;
                    if (distance < out.at<float>(i, neighbor)) {
                        out.at<float>(i, neighbor) = distance;
                        pq.emplace(neighbor, distance);
                    }
                }
            }
        }
    }

    auto time2 = std::chrono::high_resolution_clock::now();
    std::cout << "Dijkstra time cost" << ": " << (time2 - time1).count() / 1000000.f << " ms" << std::endl;
    showMat<float>("dijkstra mat", out);

    // replace inf with 0s for following steps
    out.setTo(0, out == std::numeric_limits<float>::infinity());

    Scalar sum = cv::sum(out);
    auto sum_val = static_cast<float>(sum[0]);
    float sigma = sum_val / static_cast<float>(num_faces * num_faces);
    float square_sigma = 2.0f * sigma * sigma;
    // Gaussian kernel
    for (int i = 0; i < out.rows; ++i) {
        for (int j = 0; j < out.cols; ++j) {
            auto &cur_elem = out.at<float>(i, j);
            cur_elem = (cur_elem != 0.f) ? exp(-cur_elem / square_sigma) : 0.f;
        }
    }

    // Set diagonal elements to 1
    for (int i = 0; i < num_faces; ++i)
        out.at<float>(i, i) = 1;
    time1 = std::chrono::high_resolution_clock::now();
    std::cout << "Other time cost" << ": " << (time1 - time2).count() / 1000000.f << " ms" << std::endl;
}

void SpectralCluster::getDistanceMat(Mat &out, std::vector<Point3f> &vertices, std::vector<std::vector<int32_t>> &indices) {
    int num_faces = int(indices.size());
    int num_adjacent = 0;
    // create two matrices for storing distance
    Mat g_dist_mat = Mat::zeros(num_faces, num_faces, CV_32F);
    Mat a_dist_mat = Mat::zeros(num_faces, num_faces, CV_32F);
    for (int i = 0; i < num_faces; ++i) {
        const std::vector<int32_t> &face1 = indices[i];
        for (int j = i + 1; j < num_faces; ++j) {
            const std::vector<int32_t> &face2 = indices[j];
            if (!isAdjacent(face1, face2))
                continue;
            ++num_adjacent;
            // adjacent faces: get distances
            std::vector<Point3f> f1 = {vertices[face1[0]], vertices[face1[1]], vertices[face1[2]]};
            std::vector<Point3f> f2 = {vertices[face2[0]], vertices[face2[1]], vertices[face2[2]]};

            float g_dist = getGeodesicDistance(f1, f2);
            float a_dist = getAngleDistance(f1, f2);

            g_dist_mat.at<float>(i, j) = g_dist;
            g_dist_mat.at<float>(j, i) = g_dist;

            a_dist_mat.at<float>(i, j) = a_dist;
            a_dist_mat.at<float>(j, i) = a_dist;
        }
    }
    num_adjacent <<= 1;
    g_dist_mat /= (sum(g_dist_mat)[0] / num_adjacent);
    a_dist_mat /= (sum(a_dist_mat)[0] / num_adjacent);

    out = this->delta * g_dist_mat + (1.f - this->delta) * a_dist_mat;
}



int SpectralCluster::isAdjacent(const std::vector<int32_t> &face1, const std::vector<int32_t> &face2) {
    unsigned char cnt = 0;
    for (auto v1 : face1)
        for (auto v2 : face2) {
            if (v1 == v2)
                ++cnt;
            if (cnt == 2)
                return 1;
        }
    return 0;
}

float SpectralCluster::getGeodesicDistance(const std::vector<Point3f>& face1, const std::vector<Point3f> &face2) {
    cv::Point3f centroid1 = calculateFaceCentroid(face1);
    cv::Point3f centroid2 = calculateFaceCentroid(face2);
    return float(norm(centroid1 - centroid2));
}

Point3f SpectralCluster::calculateFaceCentroid(const std::vector<cv::Point3f>& face) {
    cv::Point3f centroid(0.0f, 0.0f, 0.0f);
    for(const cv::Point3f& p : face)
        centroid += p;
    centroid /= 3.f;
    return centroid;
}

float SpectralCluster::getAngleDistance(const std::vector<Point3f>& face1, const std::vector<Point3f> &face2) const {
    cv::Vec3f normal1 = calculateFaceNormal(face1);
    cv::Vec3f normal2 = calculateFaceNormal(face2);

    float cos_angle = normal1.dot(normal2);

    float angular_distance = 1.0f - cos_angle;

    if (cos_angle < 0)
        // scaling down angle distance
        angular_distance *= this->eta;

    return angular_distance;
}

Point3f SpectralCluster::calculateFaceNormal(const std::vector<Point3f>& face) {
    Vec3f normal = Vec3f(face[1] - face[0]).cross(Vec3f(face[2] - face[0]));
    normal /= cv::norm(normal);
    return normal;
}
};
