// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.


#include "precomp.hpp"

namespace cv { namespace slam {

struct Map::Impl
{
    std::vector<KeyFrame*>             kf_vec;
    std::vector<MapPoint*>             mp_vec;
    std::unordered_map<int, KeyFrame*> kf_map;
    std::unordered_map<int, MapPoint*> mp_map;
    std::vector<Matx44d>               trajectory;
    int next_kf_id = 0;
    int next_mp_id = 0;
};

Map::Map()  : impl_(makePtr<Impl>()) {}

Map::~Map()
{
    clear();
}

int Map::addKeyframe(KeyFrame& kf)
{
    if (kf.id < 0)
        kf.id = impl_->next_kf_id++;
    KeyFrame* ptr = new KeyFrame(kf);
    impl_->kf_vec.push_back(ptr);
    impl_->kf_map[ptr->id] = ptr;
    kf.id = ptr->id;
    return ptr->id;
}

KeyFrame* Map::getKeyframe(int id)
{
    auto it = impl_->kf_map.find(id);
    return it != impl_->kf_map.end() ? it->second : nullptr;
}

const std::vector<KeyFrame*>& Map::keyframes() const { return impl_->kf_vec; }

int Map::numKeyframes() const { return (int)impl_->kf_vec.size(); }

int Map::addMapPoint(MapPoint& mp)
{
    if (mp.id < 0)
        mp.id = impl_->next_mp_id++;
    MapPoint* ptr = new MapPoint(mp);
    impl_->mp_vec.push_back(ptr);
    impl_->mp_map[ptr->id] = ptr;
    mp.id = ptr->id;
    return ptr->id;
}

MapPoint* Map::getMapPoint(int id)
{
    auto it = impl_->mp_map.find(id);
    return it != impl_->mp_map.end() ? it->second : nullptr;
}

std::vector<MapPoint*> Map::mapPoints() const
{
    std::vector<MapPoint*> live;
    live.reserve(impl_->mp_vec.size());
    for (MapPoint* mp : impl_->mp_vec)
        if (mp && !mp->bad)
            live.push_back(mp);
    return live;
}

int Map::numMapPoints() const
{
    int n = 0;
    for (const MapPoint* mp : impl_->mp_vec)
        if (mp && !mp->bad) ++n;
    return n;
}

void Map::addObservation(KeyFrame* kf, int kp_idx, MapPoint* mp)
{
    CV_Assert(kf && mp);
    mp->observations[kf] = kp_idx;
    if (kp_idx < (int)kf->mappoints.size())
        kf->mappoints[kp_idx] = mp;
    if (kp_idx < (int)kf->kpt_to_mp.size())
        kf->kpt_to_mp[kp_idx] = mp->id;
}

void Map::removeMapPoint(int mp_id)
{
    MapPoint* mp = getMapPoint(mp_id);
    if (!mp || mp->bad) return;
    for (auto& [obs_kf, kp_idx] : mp->observations)
    {
        if (!obs_kf) continue;
        if (kp_idx < (int)obs_kf->mappoints.size()) obs_kf->mappoints[kp_idx] = nullptr;
        if (kp_idx < (int)obs_kf->kpt_to_mp.size()) obs_kf->kpt_to_mp[kp_idx] = -1;
    }
    mp->observations.clear();
    mp->bad = true;
}

void Map::appendPose(const Matx44d& T_cw) { impl_->trajectory.push_back(T_cw); }

const std::vector<Matx44d>& Map::trajectory() const { return impl_->trajectory; }

void Map::clear()
{
    for (KeyFrame* kf : impl_->kf_vec) delete kf;
    for (MapPoint* mp : impl_->mp_vec) delete mp;
    impl_->kf_vec.clear();
    impl_->mp_vec.clear();
    impl_->kf_map.clear();
    impl_->mp_map.clear();
    impl_->trajectory.clear();
    impl_->next_kf_id = 0;
    impl_->next_mp_id = 0;
}

}} // namespace cv::slam
