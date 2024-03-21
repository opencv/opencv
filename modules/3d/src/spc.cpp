// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Author: Zhangjie Chen <zhangjiec01@gmail.com>

#include "opencv2/core.hpp"
#include "opencv2/3d.hpp"
// TODO: test only
#include "iostream"
#define ENABLE_MAT_OUTPUT 1

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

void SpectralCluster::cluster(std::vector<int>& result, std::vector<Point3f> vertices, std::vector<std::vector<int32_t>> indices, int k) {
    int num_faces = static_cast<int>(indices.size());
    // get adjacency matrix
    Mat adjacency_mat;
    getAdjacencyMat(adjacency_mat, indices);

    // for test:
    showMat<uchar>("adjacency matrix", adjacency_mat);

    // get distance matrix
    Mat distance_mat;
    getDistanceMat(adjacency_mat, distance_mat, vertices, indices);

    // for test:
    showMat<float>("distance matrix", distance_mat);

    Mat affinity_mat;
    getAffinityMat(distance_mat, affinity_mat, indices);

    // for test:
    showMat<float>("affinity matrix", affinity_mat);

    Mat laplacian_mat;
    getLaplacianMat(affinity_mat, laplacian_mat);

    // for test:
    showMat<float>("Laplacian matrix", laplacian_mat);
    Mat eigen_values;
    Mat eigen_vectors;
    // eigenvalues are arranged from largest to smallest
    // each row is a corresponding eigenvector
    eigen(laplacian_mat, eigen_values, eigen_vectors);
    Mat eigen_vectors_ttt;
    eigen_vectors_ttt = eigen_vectors.rowRange(0, k);
    Mat eigen_vectors_t;
    eigen_vectors_t = eigen_vectors_ttt.t();
    // for test:
    showMat<float>("eigen vectors", eigen_vectors_t);

    normalize(eigen_vectors_t, eigen_vectors_t);
    // normalize each row
    for (int i = 0; i < eigen_vectors_t.rows; ++i)
        eigen_vectors_t.row(i) /= static_cast<float>(norm(eigen_vectors_t.row(i)));

    showMat<float>("normalized eigen vectors", eigen_vectors_t);

    result.clear();
    cv::kmeans(eigen_vectors_t, k, result,
               cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 10, 1.0),
               3, cv::KMEANS_PP_CENTERS);

    // cluster complete
    std::cout << "Cluster complete! Labels:" << std::endl;
    String s;
    for (int w = 0; w < num_faces; ++w) {
        s += std::to_string(result[w]);
        s += ", ";
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

void SpectralCluster::getAdjacencyMat(Mat &out, std::vector<std::vector<int32_t>> &indices) {
    int num_faces = (int)indices.size();
    out = Mat(num_faces, num_faces, CV_8U, Scalar(0));

    for (size_t i = 0; i < indices.size(); ++i) {
        const std::vector<int32_t> &face1 = indices[i];
        for (size_t j = i + 1; j < indices.size(); ++j) {
            const std::vector<int32_t> &face2 = indices[j];
            if (isAdjacent(face1, face2)) {
                out.at<uchar>((int)i, (int)j) = 1;
                out.at<uchar>((int)j, (int)i) = 1;
            }
        }
    }
}


void SpectralCluster::getDistanceMat(cv::Mat &in, cv::Mat &out, std::vector<Point3f> &vertices, std::vector<std::vector<int32_t>> &indices) {
    // in: adjacent matrix from previous step
    int num_faces = int(indices.size());
    out = Mat(num_faces, num_faces, CV_32F, Scalar(0));
    std::vector<int> i_array(num_faces * num_faces), j_array(num_faces * num_faces);
    std::vector<float> g_dist_array(num_faces * num_faces), a_dist_array(num_faces * num_faces);

    float g_dist_sum = .0f, a_dist_sum = .0f;

    int cnt = 0;

    for (int i = 0; i < num_faces; ++i) {
        for (int j = i+1; j < num_faces; ++j) {
            // for two adjacent faces, calculate their geodesic and angle distances
            if (in.at<uchar>(i, j)) {
                std::vector<Point3f> face1 = {vertices[indices[i][0]], vertices[indices[i][1]], vertices[indices[i][2]]};
                std::vector<Point3f> face2 = {vertices[indices[j][0]], vertices[indices[j][1]], vertices[indices[j][2]]};
                float g_dist = getGeodesicDistance(face1, face2);
                float a_dist = getAngleDistance(face1, face2);

                i_array[cnt] = i;
                j_array[cnt] = j;

                g_dist_array[cnt] = g_dist;
                a_dist_array[cnt++] = a_dist;

                g_dist_sum += g_dist;
                a_dist_sum += a_dist;
            }
        }
    }

    float g_mean_val = g_dist_sum / (float)cnt;
    float a_mean_val = a_dist_sum / (float)cnt;

    for (int k = 0; k < cnt; ++k) {
        out.at<float>(i_array[k], j_array[k]) = this->delta * g_dist_array[k] / g_mean_val + (1.0f - delta) * a_dist_array[k] / a_mean_val;
        out.at<float>(j_array[k], i_array[k]) = this->delta * g_dist_array[k] / g_mean_val + (1.0f - delta) * a_dist_array[k] / a_mean_val;
    }
}


void SpectralCluster::getAffinityMat(Mat &in, Mat &out, std::vector<std::vector<int32_t>> &indices) {
    // in: distance matrix from previous step
    int num_faces = int(indices.size());

    // using dijkstra algorithm to get the affinity matrix
    out = Mat::ones(num_faces, num_faces, CV_32F) * (double)std::numeric_limits<float>::infinity();

    for (int i = 0; i < num_faces; ++i) {
        out.at<float>(i, i) = 0;

        std::vector<bool> visited(num_faces, false);

        while (true) {
            int current = -1;
            for (int j = 0; j < num_faces; ++j) {
                if (!visited[j] && (current == -1 || out.at<float>(i, j) < out.at<float>(i, current))) {
                    current = j;
                }
            }

            if (current == -1) {
                break;
            }

            visited[current] = true;

            for (int neighbor = 0; neighbor < num_faces; ++neighbor) {
                float weight = in.at<float>(current, neighbor);
                if (!visited[neighbor] && weight > 0) {
                    float distance = out.at<float>(i, current) + weight;
                    if (distance < out.at<float>(i, neighbor)) {
                        out.at<float>(i, neighbor) = distance;
                    }
                }
            }
        }
    }
    showMat<float>("dijkstra mat", out);
    // replace inf with 0s for following steps
    out.setTo(0, out == std::numeric_limits<float>::infinity());

    Scalar sum = cv::sum(out);
    auto sum_val = static_cast<float>(sum[0]);
    float sigma = sum_val / static_cast<float>(num_faces * num_faces);
    // Gaussian kernel
    for (int i = 0; i < out.rows; ++i) {
        for (int j = 0; j < out.cols; ++j) {
            auto &cur_elem = out.at<float>(i, j);
            cur_elem = (cur_elem != 0.f) ? exp(-cur_elem / (2.0f * sigma * sigma)) : 0;
        }
    }

    // Set diagonal elements to 1
    for (int i = 0; i < num_faces; ++i)
        out.at<float>(i, i) = 1;
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
    centroid /= static_cast<float>(face.size());
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
    Vec3f edge1 = Vec3f(face[1] - face[0]);
    Vec3f edge2 = Vec3f(face[2] - face[0]);

    Vec3f normal = edge1.cross(edge2);
    normal /= cv::norm(normal);

    return normal;
}
};
