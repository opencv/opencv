#ifndef _BOOST_LBP_H_
#define _BOOST_LBP_H_

#include "common.h"

bool boostTrain(MBLBPStagef * pStrong,
                const Mat & samplesLBP,
                const Mat & labels,
                MBLBPWeakf * features,
                bool * featuresMask,
                int numFeatures,
                int numPos,
                int numNeg,
                int maxWeakCount,
                double min_hit_rate);

#endif
