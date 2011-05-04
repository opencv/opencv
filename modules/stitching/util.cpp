#include "util.hpp"

using namespace std;
using namespace cv;

void DjSets::create(int n) {
    rank_.assign(n, 0);
    size.assign(n, 1);
    parent.resize(n);
    for (int i = 0; i < n; ++i)
        parent[i] = i;
}


int DjSets::find(int elem) {
    int set = elem;
    while (set != parent[set])
        set = parent[set];
    int next;
    while (elem != parent[elem]) {
        next = parent[elem];
        parent[elem] = set;
        elem = next;
    }
    return set;
}


int DjSets::merge(int set1, int set2) {
    if (rank_[set1] < rank_[set2]) {
        parent[set1] = set2;
        size[set2] += size[set1];
        return set2;
    }
    if (rank_[set2] < rank_[set1]) {
        parent[set2] = set1;
        size[set1] += size[set2];
        return set1;
    }
    parent[set1] = set2;
    rank_[set2]++;
    size[set2] += size[set1];
    return set2;
}


void Graph::addEdge(int from, int to, float weight)
{
    edges_[from].push_back(GraphEdge(from, to, weight));
}
