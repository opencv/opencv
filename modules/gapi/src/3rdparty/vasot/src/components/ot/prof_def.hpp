/*******************************************************************************
 * Copyright (C) 2023 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef VAS_OT_PROF_DEF_HPP
#define VAS_OT_PROF_DEF_HPP

#include "../../common/prof.hpp"

// 0 ~ 999 : Reserved group id for UnitTests.
// 1000 ~  : User Defined id range under modules.

#define PROF_COMPONENTS_OT_RUN_TRACK PROF_TAG_GENERATE(OT, 1000, " ObjectTracker::Track")

#define PROF_COMPONENTS_OT_ZEROTERM_RUN_TRACKER PROF_TAG_GENERATE(OT, 1400, " ZeroTermTracker::TrackObjects")
#define PROF_COMPONENTS_OT_ZEROTERM_KALMAN_PREDICTION PROF_TAG_GENERATE(OT, 1411, " ZeroTermTracker::KalmanPrediction")
#define PROF_COMPONENTS_OT_ZEROTERM_RUN_ASSOCIATION PROF_TAG_GENERATE(OT, 1421, " ZeroTermTracker::Association")
#define PROF_COMPONENTS_OT_ZEROTERM_UPDATE_STATUS PROF_TAG_GENERATE(OT, 1441, " ZeroTermTracker::UpdateTrackedStatus")
#define PROF_COMPONENTS_OT_ZEROTERM_COMPUTE_OCCLUSION PROF_TAG_GENERATE(OT, 1461, " ZeroTermTracker::ComputeOcclusion")
#define PROF_COMPONENTS_OT_ZEROTERM_UPDATE_MODEL PROF_TAG_GENERATE(OT, 1481, " ZeroTermTracker::UpdateModel")
#define PROF_COMPONENTS_OT_ZEROTERM_REGISTER_OBJECT PROF_TAG_GENERATE(OT, 1491, " ZeroTermTracker::RegisterObject")

#define PROF_COMPONENTS_OT_ASSOCIATE_COMPUTE_DIST_TABLE                                                                \
    PROF_TAG_GENERATE(OT, 1600, " Association::ComputeDistanceTable")
#define PROF_COMPONENTS_OT_ASSOCIATE_COMPUTE_COST_TABLE PROF_TAG_GENERATE(OT, 1610, " Association::ComputeCostTable")
#define PROF_COMPONENTS_OT_ASSOCIATE_WITH_HUNGARIAN PROF_TAG_GENERATE(OT, 1620, " Association::AssociateWithHungarian")

#endif // __OT_PROF_DEF_H__
