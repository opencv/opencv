#pragma  once

#include<vector>

namespace zxing {

typedef unsigned int uint;

struct Cluster
{
    std::vector<double> centroid;
    std::vector<uint> samples;
};


double cal_distance(std::vector<double> a, std::vector<double> b);
std::vector<Cluster> k_means(std::vector<std::vector<double> > trainX, uint k, uint maxepoches, uint minchanged);

}  // namespace zxing
