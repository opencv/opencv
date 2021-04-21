//
// Created by daniel on 4/18/21.
//

#include "general.h"
#include "kdtree_index.h"

cvflann::KDTreeIndex<cvflann::L2<float> >::heap_ = nullptr;
cvflann::KDTreeIndex<cvflann::L1<float> >::heap_ = nullptr;
