// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_HAL_ACCEL_HIP_IMGPROC_RULES_HPP
#define OPENCV_HAL_ACCEL_HIP_IMGPROC_RULES_HPP

#include "opencv2/core.hpp"

#include <algorithm>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

// The kernel rules of the HIP imgproc ops, read by the HIP backend (imgproc/ops.cpp).
namespace cv { namespace accel { namespace hip_impl { namespace rules {

const int SCHEMA = 3;

//! what a rule can match: ksize is 0 for an op without a window, or a window that is not square
struct Workload
{
    int cn, ksize;
};

struct Choice
{
    std::vector<int> cn, ksize;         //!< empty matches any
    std::vector<std::string> kernels;   //!< best first
    Size block;                         //!< empty when the rules give none

    bool matches(const Workload& w) const
    {
        return (cn.empty() || std::find(cn.begin(), cn.end(), w.cn) != cn.end()) &&
               (ksize.empty() || std::find(ksize.begin(), ksize.end(), w.ksize) != ksize.end());
    }
};

struct OpRules
{
    Choice general;
    std::vector<Choice> workloads;      //!< the first that matches is used instead of general
    std::string evidence;

    const Choice& choose(const Workload& w) const
    {
        for (size_t i = 0; i < workloads.size(); i++)
            if (workloads[i].matches(w))
                return workloads[i];
        return general;
    }
};

struct ArchRules
{
    std::string section;
    std::map<std::string, OpRules> ops;
    std::string device, plugin, date;   //!< measured_on, in sections of tuned rules
    int runtime = 0;
};

typedef std::map<std::string, ArchRules> RuleFile;

inline bool implemented(const std::string& op, const std::string& kernel)
{
    return (op == "boxFilter" && (kernel == "lds_tile" || kernel == "sliding_column")) ||
           (op == "nv12ToBgr" && kernel == "chroma_share");
}

inline bool takesBlock(const std::string& op) { return op == "nv12ToBgr"; }
inline bool takesKsize(const std::string& op) { return op == "boxFilter"; }

inline bool readInts(const FileNode& n, std::vector<int>& v)
{
    if (!n.isSeq() || n.size() == 0)
        return false;
    for (FileNodeIterator it = n.begin(); it != n.end(); ++it)
    {
        if (!(*it).isInt() || (int)*it <= 0)
            return false;
        v.push_back((int)*it);
    }
    return true;
}

//! what is wrong with @p n as a choice of @p op, or empty
inline std::string readChoice(const FileNode& n, const std::string& op, bool workload, Choice& c)
{
    for (FileNodeIterator it = n.begin(); it != n.end(); ++it)
    {
        const FileNode v = *it;
        const std::string key = v.name();
        if (key == "kernels")
        {
            if (!v.isSeq() || v.size() == 0)
                return "kernels is not a list";
            for (FileNodeIterator k = v.begin(); k != v.end(); ++k)
            {
                const std::string kernel = (*k).isString() ? (std::string)*k : std::string("?");
                if (!implemented(op, kernel))
                    return "no kernel '" + kernel + "'";
                c.kernels.push_back(kernel);
            }
        }
        else if (key == "block" && takesBlock(op))
        {
            std::vector<int> b;
            if (!readInts(v, b) || b.size() != 2 || b[0] * b[1] > 256)
                return "a block is [x, y] with at most 256 threads";
            c.block = Size(b[0], b[1]);
        }
        else if (key == "cn" && workload)
        {
            if (!readInts(v, c.cn))
                return "cn is not a list of channel counts";
        }
        else if (key == "ksize" && workload && takesKsize(op))
        {
            if (!readInts(v, c.ksize))
                return "ksize is not a list of window sizes";
        }
        else if (workload || (key != "workloads" && key != "evidence"))
            return "unexpected '" + key + "'";
    }
    if (c.kernels.empty())
        return "no kernels";
    if (workload && c.cn.empty() && c.ksize.empty())
        return "a workload matches on cn or ksize";
    return std::string();
}

inline std::string readOp(const FileNode& n, const std::string& op, OpRules& r)
{
    if (!takesBlock(op) && !takesKsize(op))
        return "no such operation";
    if (!n.isMap())
        return "not a map";
    std::string why = readChoice(n, op, false, r.general);
    const FileNode ws = n["workloads"];
    if (why.empty() && !ws.empty() && !ws.isSeq())
        why = "workloads is not a list";
    for (FileNodeIterator it = ws.begin(); why.empty() && it != ws.end(); ++it)
    {
        Choice c;
        why = readChoice(*it, op, true, c);
        if (c.block.empty())
            c.block = r.general.block;
        r.workloads.push_back(c);
    }
    r.evidence = (std::string)n["evidence"];
    return why;
}

//! the rules in @p path; an invalid file, section or operation is left out and described in @p warnings
inline RuleFile read(const std::string& path, std::vector<std::string>& warnings)
{
    RuleFile out;
    try
    {
        FileStorage fs(path, FileStorage::READ);
        if (!fs.isOpened())
        {
            warnings.push_back("cannot open " + path);
            return out;
        }
        const FileNode schema = fs["schema"];
        if (!schema.isInt() || (int)schema != SCHEMA)
        {
            warnings.push_back(format("%s is not schema %d; ignored", path.c_str(), SCHEMA));
            return out;
        }
        const FileNode archs = fs["archs"];
        for (FileNodeIterator a = archs.begin(); a != archs.end(); ++a)
        {
            ArchRules r;
            r.section = (*a).name();
            for (FileNodeIterator o = (*a).begin(); o != (*a).end(); ++o)
            {
                const FileNode n = *o;
                const std::string name = n.name();
                if (name == "measured_on")
                {
                    r.device = (std::string)n["device"];
                    r.runtime = (int)n["hip_runtime"];
                    r.plugin = (std::string)n["plugin"];
                    r.date = (std::string)n["date"];
                    continue;
                }
                OpRules op;
                const std::string why = readOp(n, name, op);
                if (why.empty())
                    r.ops[name] = op;
                else
                    warnings.push_back(path + ", " + r.section + ", " + name + ": " + why + "; ignored");
            }
            out[r.section] = r;
        }
    }
    catch (const cv::Exception& e)
    {
        std::string what = e.what();
        std::replace(what.begin(), what.end(), '\n', ' ');
        warnings.push_back(path + ": " + what + "; ignored");
        out.clear();
    }
    return out;
}

//! FNV-1a of the bytes of @p path, or empty when it cannot be read
inline std::string fingerprint(const std::string& path)
{
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f)
        return std::string();
    uint64 h = 14695981039346656037ULL;
    std::vector<unsigned char> buf(1 << 16);
    size_t n;
    while ((n = std::fread(&buf[0], 1, buf.size(), f)) > 0)
        for (size_t i = 0; i < n; i++)
            h = (h ^ buf[i]) * 1099511628211ULL;
    std::fclose(f);
    return format("fnv1a-%016llx", (unsigned long long)h);
}

//! shipped rules of @p arch (else "default"), with its tuned ops on top when measured on this device, runtime and plugin
inline ArchRules effective(const RuleFile& shipped, const RuleFile& tuned, const std::string& arch,
                           const std::string& device, int runtime, const std::string& plugin,
                           std::vector<std::string>& warnings, bool& found)
{
    RuleFile::const_iterator base = shipped.find(arch);
    if (base == shipped.end())
        base = shipped.find("default");
    RuleFile::const_iterator t = tuned.find(arch);
    if (t != tuned.end() && (t->second.device != device || t->second.runtime != runtime || t->second.plugin != plugin))
    {
        const char* what = t->second.device != device ? "another GPU" : t->second.runtime != runtime ? "another HIP runtime"
                                                                                                       : "another build of the plugin";
        warnings.push_back("the tuned rules for " + arch + " were measured with " + what + " (" + t->second.device +
                           format(", HIP %d, ", t->second.runtime) + t->second.plugin + "); using the shipped rules. "
                           "Tune again.");
        t = tuned.end();
    }

    ArchRules r;
    if (base != shipped.end())
        r = base->second;
    if (t != tuned.end())
    {
        for (std::map<std::string, OpRules>::const_iterator op = t->second.ops.begin(); op != t->second.ops.end(); ++op)
            r.ops[op->first] = op->second;
        r.section = arch + " (tuned)" + (base != shipped.end() ? " over " + base->second.section : std::string());
    }
    found = base != shipped.end() || t != tuned.end();
    return r;
}

}}}} // namespace cv::accel::hip_impl::rules

#endif // OPENCV_HAL_ACCEL_HIP_IMGPROC_RULES_HPP
