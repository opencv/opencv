// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019 Intel Corporation

#include "compiler/gmodel.hpp"
#include <iostream>
#include <fstream>

namespace cv
{
namespace gimpl
{
// FIXME? Name is too long?
namespace serialization {

// FIXME? Split RcDesc,Kernel,Op,Data etc
// into serializable/non-serializable parts?
// All structures below are partial copies of default (non-serializable) ones
struct Kernel
{
    std::string name;
    std::string tag;
};

struct RcDesc
{
    GShape shape;
    int    id;
    bool operator==(const RcDesc& rc) const { return id == rc.id && shape == rc.shape; }
};

struct Op
{
    Kernel k;
    // FIXME: GArg needs to be serialized properly
    //GArg data
    std::vector<int>   kind;
    std::vector<int>   opaque_kind;
    std::vector<RcDesc> outs;
    std::vector<RcDesc> ins;
    //opaque args
    std::vector<int> opaque_ints;
    std::vector<double> opaque_doubles;
    std::vector<cv::Size> opaque_cvsizes;
    std::vector<bool> opaque_bools;
    std::vector<cv::Scalar> opaque_cvscalars;
    std::vector<cv::Point> opaque_cvpoints;
    std::vector<cv::Mat> opaque_cvmats;
    std::vector<cv::Rect> opaque_cvrects;
};

struct Data
{
    // GModel::Data consists of shape+(int)rc
    RcDesc rc;
    GMetaArg meta;
    // storage?

    bool operator==(const Data& d) const { return rc == d.rc; }
};

struct GSerialized
{
    // Need to monitor ins/outs of the graph?
    // Remove m_?
    std::vector<Op> m_ops;
    std::vector<Data> m_datas;
};

GSerialized serialize(const gimpl::GModel::ConstGraph& m_gm, const std::vector<ade::NodeHandle>& nodes);
void deserialize(const GSerialized& gs);
void mkDataNode(ade::Graph& g, const Data& data);
void mkOpNode(ade::Graph& g, const Op& op);
std::vector<ade::NodeHandle> linkNodes(ade::Graph& g);
void putData(GSerialized& s, const GModel::ConstGraph& cg, const ade::NodeHandle nh);
void putOp(GSerialized& s, const GModel::ConstGraph& cg, const ade::NodeHandle nh);
void printOp(const Op& op);
void printData(const Data& data);
void printGSerialized(const GSerialized s);
void dumpGSerializedOps(const GSerialized s, std::ofstream &ofs_ops);
void cleanGSerializedOps(GSerialized &s);
void readGSerializedOps(GSerialized &s, std::ifstream &ifs_ops);
void dumpGSerializedDatas(const GSerialized s, std::ofstream &ofs_data);
void cleanupGSerializedDatas(GSerialized &s);
void readGSerializedDatas(GSerialized &s, std::ifstream &ifs_data);

//Graph dump operators
std::ostream& operator << (std::ostream& os, Kernel k);
std::ostream& operator << (std::ostream& os, std::string str);
std::ostream& operator << (std::ostream& os, std::vector<int> ints);
std::ostream& operator << (std::ostream& os, std::vector<RcDesc> descs);
std::ostream& operator << (std::ostream& os, RcDesc desc);
std::ostream& operator << (std::ostream& os, std::vector<double> doubles);
std::ostream& operator << (std::ostream& os, std::vector<cv::Size> cvsizes);
std::ostream& operator << (std::ostream& os, std::vector<bool> bools);
std::ostream& operator << (std::ostream& os, std::vector<cv::Scalar> cvscalars);
std::ostream& operator << (std::ostream& os, std::vector<cv::Point> cvpoints);
std::ostream& operator << (std::ostream& os, std::vector<cv::Mat> cvmats);
std::ostream& operator << (std::ostream& os, std::vector<cv::Rect> cvrects);
std::ostream& operator << (std::ostream& os, Data data);
std::ostream& operator << (std::ostream& os, std::vector<Data> datas);
std::ostream& operator << (std::ostream& os, Op op);
std::ostream& operator << (std::ostream& os, std::vector<Op> ops);

//Graph restore operators
std::istream& operator >> (std::istream& is, Kernel& k);
std::istream& operator >> (std::istream& is, std::string& str);
std::istream& operator >> (std::istream& is, std::vector<int>& ints);
std::istream& operator >> (std::istream& is, std::vector<RcDesc>& descs);
std::istream& operator >> (std::istream& is, RcDesc& desc);
std::istream& operator >> (std::istream& is, std::vector<double>& doubles);
std::istream& operator >> (std::istream& is, std::vector<cv::Size>& cvsizes);
std::istream& operator >> (std::istream& is, std::vector<bool>& bools);
std::istream& operator >> (std::istream& is, std::vector<cv::Scalar>& cvscalars);
std::istream& operator >> (std::istream& is, std::vector<cv::Point>& cvpoints);
std::istream& operator >> (std::istream& is, std::vector<cv::Mat>& cvmats);
std::istream& operator >> (std::istream& is, std::vector<cv::Rect>& cvrects);
std::istream& operator >> (std::istream& is, Data& data);
std::istream& operator >> (std::istream& is, std::vector<Data>& datas);
std::istream& operator >> (std::istream& is, Op& op);
std::istream& operator >> (std::istream& is, std::vector<Op>& ops);

} // namespace serialization
} // namespace gimpl

namespace detail
{
    template<> struct GTypeTraits<cv::gimpl::serialization::RcDesc>
    {
        static constexpr const ArgKind kind = ArgKind::GOBJREF;
    };
} // namespace detail
} // namespace cv
