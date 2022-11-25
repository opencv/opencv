// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// Partially rewritten from https://github.com/Nerei/kinfu_remake
// Copyright(c) 2012, Anatoly Baksheev. All rights reserved.

#ifndef OPENCV_3D_HASH_TSDF_FUNCTIONS_HPP
#define OPENCV_3D_HASH_TSDF_FUNCTIONS_HPP

#include <unordered_set>

#include "utils.hpp"
#include "tsdf_functions.hpp"

#define USE_INTERPOLATION_IN_GETNORMAL 1
#define VOLUMES_SIZE 8192

namespace cv
{

//! Spatial hashing
struct tsdf_hash
{
    size_t operator()(const Vec3i& x) const noexcept
    {
        size_t seed = 0;
        constexpr uint32_t GOLDEN_RATIO = 0x9e3779b9;
        for (uint16_t i = 0; i < 3; i++)
        {
            seed ^= std::hash<int>()(x[i]) + GOLDEN_RATIO + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

struct VolumeUnit
{
    cv::Vec3i coord;
    int index;
    cv::Matx44f pose;
    int lastVisibleIndex = 0;
    bool isActive;
};

class CustomHashSet
{
public:
    static const int hashDivisor = 32768;
    static const int startCapacity = 2048;

    std::vector<int> hashes;
    // 0-3 for key, 4th for internal use
    // don't keep keep value
    std::vector<Vec4i> data;
    int capacity;
    int last;

    CustomHashSet()
    {
        hashes.resize(hashDivisor);
        for (int i = 0; i < hashDivisor; i++)
            hashes[i] = -1;
        capacity = startCapacity;

        data.resize(capacity);
        for (int i = 0; i < capacity; i++)
            data[i] = { 0, 0, 0, -1 };

        last = 0;
    }

    ~CustomHashSet() { }

    inline size_t calc_hash(Vec3i x) const
    {
        uint32_t seed = 0;
        constexpr uint32_t GOLDEN_RATIO = 0x9e3779b9;
        for (int i = 0; i < 3; i++)
        {
            seed ^= x[i] + GOLDEN_RATIO + (seed << 6) + (seed >> 2);
        }
        return seed;
    }

    // should work on existing elements too
    // 0 - need resize
    // 1 - idx is inserted
    // 2 - idx already exists
    int insert(Vec3i idx)
    {
        if (last < capacity)
        {
            int hash = int(calc_hash(idx) % hashDivisor);
            int place = hashes[hash];
            if (place >= 0)
            {
                int oldPlace = place;
                while (place >= 0)
                {
                    if (data[place][0] == idx[0] &&
                        data[place][1] == idx[1] &&
                        data[place][2] == idx[2])
                        return 2;
                    else
                    {
                        oldPlace = place;
                        place = data[place][3];
                        //std::cout << "place=" << place << std::endl;
                    }
                }

                // found, create here
                data[oldPlace][3] = last;
            }
            else
            {
                // insert at last
                hashes[hash] = last;
            }

            data[last][0] = idx[0];
            data[last][1] = idx[1];
            data[last][2] = idx[2];
            data[last][3] = -1;
            last++;

            return 1;
        }
        else
            return 0;
    }

    int find(Vec3i idx) const
    {
        int hash = int(calc_hash(idx) % hashDivisor);
        int place = hashes[hash];
        // search a place
        while (place >= 0)
        {
            if (data[place][0] == idx[0] &&
                data[place][1] == idx[1] &&
                data[place][2] == idx[2])
                break;
            else
            {
                place = data[place][3];
            }
        }

        return place;
    }
};

// TODO: remove this structure as soon as HashTSDFGPU data is completely on GPU;
// until then CustomHashTable can be replaced by this one if needed

const int NAN_ELEMENT = -2147483647;

struct Volume_NODE
{
    Vec4i idx = Vec4i(NAN_ELEMENT);
    int32_t row = -1;
    int32_t nextVolumeRow = -1;
    int32_t dummy = 0;
    int32_t dummy2 = 0;
};

const int _hash_divisor = 32768;
const int _list_size = 4;

class VolumesTable
{
public:
    const int hash_divisor = _hash_divisor;
    const int list_size = _list_size;
    const int32_t free_row = -1;
    const int32_t free_isActive = 0;

    const cv::Vec4i nan4 = cv::Vec4i(NAN_ELEMENT);

    int bufferNums;
    cv::Mat volumes;

    VolumesTable() : bufferNums(1)
    {
        this->volumes = cv::Mat(hash_divisor * list_size, 1, rawType<Volume_NODE>());
        for (int i = 0; i < volumes.size().height; i++)
        {
            Volume_NODE* v = volumes.ptr<Volume_NODE>(i);
            v->idx = nan4;
            v->row = -1;
            v->nextVolumeRow = -1;
        }
    }
    const VolumesTable& operator=(const VolumesTable& vt)
    {
        this->volumes = vt.volumes;
        this->bufferNums = vt.bufferNums;
        return *this;
    }
    ~VolumesTable() {};

    bool insert(Vec3i idx, int row)
    {
        CV_Assert(row >= 0);

        int bufferNum = 0;
        int hash = int(calc_hash(idx) % hash_divisor);
        int start = getPos(idx, bufferNum);
        int i = start;

        while (i >= 0)
        {
            Volume_NODE* v = volumes.ptr<Volume_NODE>(i);

            if (v->idx[0] == NAN_ELEMENT)
            {
                Vec4i idx4(idx[0], idx[1], idx[2], 0);

                bool extend = false;
                if (i != start && i % list_size == 0)
                {
                    if (bufferNum >= bufferNums - 1)
                    {
                        extend = true;
                        volumes.resize(hash_divisor * bufferNums);
                        bufferNums++;
                    }
                    bufferNum++;
                    v->nextVolumeRow = (bufferNum * hash_divisor + hash) * list_size;
                }
                else
                {
                    v->nextVolumeRow = i + 1;
                }

                v->idx = idx4;
                v->row = row;

                return extend;
            }

            i = v->nextVolumeRow;
        }
        return false;
    }
    int findRow(Vec3i idx) const
    {
        int bufferNum = 0;
        int i = getPos(idx, bufferNum);

        while (i >= 0)
        {
            const Volume_NODE* v = volumes.ptr<Volume_NODE>(i);

            if (v->idx == Vec4i(idx[0], idx[1], idx[2], 0))
                return v->row;
            else
                i = v->nextVolumeRow;
        }

        return -1;
    }

    inline int getPos(Vec3i idx, int bufferNum) const
    {
        int hash = int(calc_hash(idx) % hash_divisor);
        return (bufferNum * hash_divisor + hash) * list_size;
    }

    inline size_t calc_hash(Vec3i x) const
    {
        uint32_t seed = 0;
        constexpr uint32_t GOLDEN_RATIO = 0x9e3779b9;
        for (int i = 0; i < 3; i++)
        {
            seed ^= x[i] + GOLDEN_RATIO + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

int calcVolumeUnitDegree(Point3i volumeResolution);

typedef std::unordered_map<cv::Vec3i, VolumeUnit, tsdf_hash> VolumeUnitIndexes;

void integrateHashTsdfVolumeUnit(
    const VolumeSettings& settings, const Matx44f& cameraPose, int& lastVolIndex, const int frameId, const int volumeUnitDegree, bool enableGrowth,
    InputArray _depth, InputArray _pixNorms, InputArray _volUnitsData, VolumeUnitIndexes& volumeUnits);

void raycastHashTsdfVolumeUnit(
    const VolumeSettings& settings, const Matx44f& cameraPose, int height, int width, InputArray intr, const int volumeUnitDegree,
    InputArray _volUnitsData, const VolumeUnitIndexes& volumeUnits, OutputArray _points, OutputArray _normals);

void fetchNormalsFromHashTsdfVolumeUnit(
    const VolumeSettings& settings, InputArray _volUnitsData, const VolumeUnitIndexes& volumeUnits,
    const int volumeUnitDegree, InputArray _points, OutputArray _normals);

void fetchPointsNormalsFromHashTsdfVolumeUnit(
    const VolumeSettings& settings, InputArray _volUnitsData, const VolumeUnitIndexes& volumeUnits,
    const int volumeUnitDegree, OutputArray _points, OutputArray _normals);

#ifdef HAVE_OPENCL
void ocl_integrateHashTsdfVolumeUnit(
    const VolumeSettings& settings, const Matx44f& cameraPose, int& lastVolIndex, const int frameId, int& bufferSizeDegree, const int volumeUnitDegree, bool enableGrowth,
    InputArray _depth, InputArray _pixNorms, InputArray _lastVisibleIndices, InputArray _volUnitsDataCopy, InputArray _volUnitsData, CustomHashSet& hashTable, InputArray _isActiveFlags);

void ocl_raycastHashTsdfVolumeUnit(
    const VolumeSettings& settings, const Matx44f& cameraPose, int height, int width, InputArray intr, const int volumeUnitDegree,
    const CustomHashSet& hashTable, InputArray _volUnitsData, OutputArray _points, OutputArray _normals);

void ocl_fetchNormalsFromHashTsdfVolumeUnit(
    const VolumeSettings& settings, const int volumeUnitDegree, InputArray _volUnitsData, InputArray _volUnitsDataCopy,
    const CustomHashSet& hashTable, InputArray _points, OutputArray _normals);

void ocl_fetchPointsNormalsFromHashTsdfVolumeUnit(
    const VolumeSettings& settings, const int volumeUnitDegree, InputArray _volUnitsData, InputArray _volUnitsDataCopy,
    const CustomHashSet& hashTable, OutputArray _points, OutputArray _normals);
#endif

} // namespace cv

#endif
