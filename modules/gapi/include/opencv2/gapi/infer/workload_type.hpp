// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2025 Intel Corporation

#ifndef OPENCV_WORKLOADTYPE_HPP
#define OPENCV_WORKLOADTYPE_HPP

#include <string>
#include <functional>
#include <vector>
#include <algorithm>

using Callback = std::function<void(const std::string &type)>;

class WorkloadListener {
    Callback callback;
public:
    uint64_t id;
    WorkloadListener(const Callback &cb, uint64_t listener_id) : callback(cb), id(listener_id) {}

    void operator()(const std::string &type) const {
        if (callback) {
            callback(type);
        }
    }

    bool operator==(const WorkloadListener& other) const {
        return id == other.id;
    }
};

class WorkloadType {
    std::vector<WorkloadListener> listeners;
    uint64_t nextId = 1;
public:
    uint64_t addListener(const Callback &cb) {
        uint64_t id = nextId++;
        listeners.emplace_back(cb, id);
        return id;
    }

    void removeListener(uint64_t id) {
        auto it = std::remove_if(listeners.begin(), listeners.end(),
            [id](const WorkloadListener& entry) { return entry.id == id; });
        if (it != listeners.end()) {
            listeners.erase(it, listeners.end());
        }
    }

    void set(const std::string &type) {
        for (const auto &listener : listeners) {
            listener(type);
        }
    }
};

#endif // OPENCV_WORKLOADTYPE_HPP
