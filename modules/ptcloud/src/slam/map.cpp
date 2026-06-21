// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "precomp.hpp"

#include <mutex>
#include <set>
#include <unordered_map>

namespace cv {
namespace slam {

struct Map::Impl
{
    std::set<KeyFrame*> keyframes;
    std::set<MapPoint*> mapPoints;
    std::unordered_map<int, KeyFrame*> kfIndex;
    std::unordered_map<int, MapPoint*> mpIndex;

    KeyFrame* refKf = nullptr;
    KeyFrame* currentKf = nullptr;

    std::vector<Matx44d> trajectory;
    std::mutex mutex;

    int nextKfId = 0;
    int nextMpId = 0;
};

Map::Map() : impl(makePtr<Impl>()) {}

Map::~Map()
{
    for (KeyFrame* kf : impl->keyframes) delete kf;
    for (MapPoint* mp : impl->mapPoints) delete mp;
}

// Keyframes

KeyFrame* Map::addKeyframe(KeyFrame* kf)
{
    CV_Assert(kf);
    if (kf->id < 0)
        kf->id = impl->nextKfId++;
    else if (kf->id >= impl->nextKfId)
        impl->nextKfId = kf->id + 1;
    impl->keyframes.insert(kf);
    impl->kfIndex[kf->id] = kf;
    return kf;
}

KeyFrame* Map::getKeyframe(int id) const
{
    auto it = impl->kfIndex.find(id);
    return (it != impl->kfIndex.end()) ? it->second : nullptr;
}

const std::set<KeyFrame*>& Map::keyframes() const { return impl->keyframes; }
int Map::numKeyframes() const { return (int)impl->keyframes.size(); }

// Map points

MapPoint* Map::addMapPoint(MapPoint* mp)
{
    CV_Assert(mp);
    if (mp->id < 0)
        mp->id = impl->nextMpId++;
    else if (mp->id >= impl->nextMpId)
        impl->nextMpId = mp->id + 1;
    impl->mapPoints.insert(mp);
    impl->mpIndex[mp->id] = mp;
    return mp;
}

MapPoint* Map::getMapPoint(int id) const
{
    auto it = impl->mpIndex.find(id);
    return (it != impl->mpIndex.end()) ? it->second : nullptr;
}

const std::set<MapPoint*>& Map::mapPoints() const { return impl->mapPoints; }
int Map::numMapPoints() const { return (int)impl->mapPoints.size(); }

// Observations

void Map::addObservation(KeyFrame* kf, size_t kpIdx, MapPoint* mp)
{
    CV_Assert(kf && mp);
    CV_Assert(kpIdx < kf->mapPoints.size());
    if (kf->mapPoints[kpIdx] != nullptr) return;
    kf->mapPoints[kpIdx] = mp;
    mp->observations[kf] = kpIdx;
}

void Map::removeObservation(KeyFrame* kf, MapPoint* mp)
{
    if (!kf || !mp) return;
    auto it = mp->observations.find(kf);
    if (it == mp->observations.end()) return;
    size_t kpIdx = it->second;
    if (kpIdx < kf->mapPoints.size())
        kf->mapPoints[kpIdx] = nullptr;
    mp->observations.erase(it);
}

void Map::removeMapPoint(MapPoint* mp)
{
    if (!mp) return;
    for (auto& [kf, kpIdx] : mp->observations)
        if (kpIdx < kf->mapPoints.size())
            kf->mapPoints[kpIdx] = nullptr;
    impl->mapPoints.erase(mp);
    impl->mpIndex.erase(mp->id);
    delete mp;
}

// Reference / current keyframes

void Map::setRefKeyframe(KeyFrame* kf) { impl->refKf = kf; }
KeyFrame* Map::getRefKeyframe() const { return impl->refKf; }

void Map::setCurrentKeyframe(KeyFrame* kf) { impl->currentKf = kf; }
KeyFrame* Map::getCurrentKeyframe() const { return impl->currentKf; }

// Trajectory

void Map::appendPose(const Matx44d& T_cw) { impl->trajectory.push_back(T_cw); }
const std::vector<Matx44d>& Map::trajectory() const { return impl->trajectory; }

// Lifecycle

void Map::clear()
{
    for (KeyFrame* kf : impl->keyframes) delete kf;
    for (MapPoint* mp : impl->mapPoints) delete mp;
    impl->keyframes.clear();
    impl->mapPoints.clear();
    impl->kfIndex.clear();
    impl->mpIndex.clear();
    impl->refKf = nullptr;
    impl->currentKf = nullptr;
    impl->trajectory.clear();
    impl->nextKfId = 0;
    impl->nextMpId = 0;
}

}} // namespace cv::slam
