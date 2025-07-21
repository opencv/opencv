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
    Callback cb;

public:
    WorkloadListener(const Callback &cb) : cb(cb) {}
    
    void operator()(const std::string &type) const {
        if (cb) {
            cb(type);
        }
    }
    
    bool operator==(const WorkloadListener& other) const {
        // Compare function pointers if both are function pointers
        auto thisPtr = cb.target<void(*)(const std::string&)>();
        auto otherPtr = other.cb.target<void(*)(const std::string&)>();
        
        if (thisPtr && otherPtr) {
            return *thisPtr == *otherPtr;
        }
        
        // For lambdas and other callables, compare target type and address
        return cb.target_type() == other.cb.target_type() &&
               cb.target<void(*)(const std::string&)>() == other.cb.target<void(*)(const std::string&)>();
    }
};
#include <iostream>
class WorkloadType {
    std::vector<WorkloadListener> listeners;
    
public:
    void addListener(const Callback &cb) {
        listeners.push_back(cb);
    }
    
    void removeListener(const Callback& cb) {
        std::cout << "removeListener: " << listeners.size()  << std::endl;
        WorkloadListener toRemove(cb);
        listeners.erase(
            std::remove(listeners.begin(), listeners.end(), toRemove),
            listeners.end()
        );
        std::cout << "after removeListener: " << listeners.size()  << std::endl;

    }
    
    void notify(const std::string &type) {
        for (const auto &listener : listeners) {
            listener(type);
        }
    }
};

#endif // OPENCV_WORKLOADTYPE_HPP
