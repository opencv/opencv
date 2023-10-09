/***********************************************************************
 * Software License Agreement (BSD License)
 *
 * Copyright 2008-2011  Marius Muja (mariusm@cs.ubc.ca). All rights reserved.
 * Copyright 2008-2011  David G. Lowe (lowe@cs.ubc.ca). All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *************************************************************************/


#ifndef OPENCV_FLANN_PARAMS_H_
#define OPENCV_FLANN_PARAMS_H_

//! @cond IGNORED

#include "any.h"
#include "general.h"
#include <iostream>
#include <map>


namespace cvflann
{

typedef std::map<cv::String, any> IndexParams;

struct SearchParams : public IndexParams
{
    SearchParams(int checks = 32, float eps = 0, bool sorted = true )
    {
        init(checks, eps, sorted, false);
    }

    SearchParams(int checks, float eps, bool sorted, bool explore_all_trees )
    {
        init(checks, eps, sorted, explore_all_trees);
    }

    void init(int checks = 32, float eps = 0, bool sorted = true, bool explore_all_trees = false )
    {
        // how many leafs to visit when searching for neighbours (-1 for unlimited)
        (*this)["checks"] = checks;
        // search for eps-approximate neighbours (default: 0)
        (*this)["eps"] = eps;
        // only for radius search, require neighbours sorted by distance (default: true)
        (*this)["sorted"] = sorted;
        // if false, search stops at the tree reaching the number of  max checks (original behavior).
        // When true, we do a descent in each tree and. Like before the alternative paths
        // stored in the heap are not be processed further when max checks is reached.
        (*this)["explore_all_trees"] = explore_all_trees;
    }
};


template<typename T>
T get_param(const IndexParams& params, const cv::String& name, const T& default_value)
{
    IndexParams::const_iterator it = params.find(name);
    if (it != params.end()) {
        try {
            return it->second.cast<T>();
        } catch (const std::exception& e) {
            CV_Error_(cv::Error::StsBadArg,
                      ("FLANN '%s' param type mismatch: %s", name.c_str(), e.what()));
        }
    }
    else {
        return default_value;
    }
}

template<typename T>
T get_param(const IndexParams& params, const cv::String& name)
{
    IndexParams::const_iterator it = params.find(name);
    if (it != params.end()) {
        try {
            return it->second.cast<T>();
        } catch (const std::exception& e) {
            CV_Error_(cv::Error::StsBadArg,
                      ("FLANN '%s' param type mismatch: %s", name.c_str(), e.what()));
        }
    }
    else {
        FLANN_THROW(cv::Error::StsBadArg, cv::String("Missing parameter '")+name+cv::String("' in the parameters given"));
    }
}

inline void print_params(const IndexParams& params, std::ostream& stream)
{
    IndexParams::const_iterator it;

    for(it=params.begin(); it!=params.end(); ++it) {
        stream << it->first << " : " << it->second << std::endl;
    }
}

inline void print_params(const IndexParams& params)
{
    print_params(params, std::cout);
}

}

//! @endcond

#endif /* OPENCV_FLANN_PARAMS_H_ */
