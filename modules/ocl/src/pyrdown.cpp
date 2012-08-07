
#include "precomp.hpp"

using namespace cv;
using namespace cv::ocl;
using namespace std;

using std::cout;
using std::endl;

namespace cv
{
    namespace ocl
    {
        ///////////////////////////OpenCL kernel strings///////////////////////////
        extern const char *pyr_down;

    }
}

//////////////////////////////////////////////////////////////////////////////
/////////////////////// add subtract multiply divide /////////////////////////
//////////////////////////////////////////////////////////////////////////////
template<typename T>
void pyrdown_run(const oclMat &src, const oclMat &dst)
{
    CV_Assert(src.cols / 2 == dst.cols && src.rows / 2 == dst.rows);

    CV_Assert(src.type() == dst.type());
    CV_Assert(src.depth() != CV_8S);

    Context  *clCxt = src.clCxt;
    //int channels = dst.channels();
    //int depth = dst.depth();

    string kernelName = "pyrDown";

    //int vector_lengths[4][7] = {{4, 0, 4, 4, 1, 1, 1},
    //    {4, 0, 4, 4, 1, 1, 1},
    //    {4, 0, 4, 4, 1, 1, 1},
    //    {4, 0, 4, 4, 1, 1, 1}
    //};

    //size_t vector_length = vector_lengths[channels-1][depth];
    //int offset_cols = (dst.offset / dst.elemSize1()) & (vector_length - 1);

    size_t localThreads[3]  = { 256, 1, 1 };
    size_t globalThreads[3] = { src.cols, dst.rows, 1};

    //int dst_step1 = dst.cols * dst.elemSize();
    vector<pair<size_t , const void *> > args;
    args.push_back( make_pair( sizeof(cl_mem), (void *)&src.data ));
    args.push_back( make_pair( sizeof(cl_int), (void *)&src.step ));
    args.push_back( make_pair( sizeof(cl_int), (void *)&src.offset ));
    args.push_back( make_pair( sizeof(cl_int), (void *)&src.rows));
    args.push_back( make_pair( sizeof(cl_int), (void *)&src.cols));
    args.push_back( make_pair( sizeof(cl_mem), (void *)&dst.data ));
    args.push_back( make_pair( sizeof(cl_int), (void *)&dst.step ));
    args.push_back( make_pair( sizeof(cl_int), (void *)&dst.offset ));
    args.push_back( make_pair( sizeof(cl_int), (void *)&dst.cols));

    openCLExecuteKernel(clCxt, &pyr_down, kernelName, globalThreads, localThreads, args, src.channels(), src.depth());
}
void pyrdown_run(const oclMat &src, const oclMat &dst)
{
	switch(src.depth())
	{
	case 0:
	    pyrdown_run<unsigned char>(src, dst);
		break;

	case 1:
	    pyrdown_run<char>(src, dst);
		break;

	case 2:
	    pyrdown_run<unsigned short>(src, dst);
		break;

	case 3:
	    pyrdown_run<short>(src, dst);
		break;

	case 4:
	    pyrdown_run<int>(src, dst);
		break;

	case 5:
	    pyrdown_run<float>(src, dst);
		break;

	case 6:
	    pyrdown_run<double>(src, dst);
		break;

	default:
		break;
	}
}
//////////////////////////////////////////////////////////////////////////////
// pyrDown

void cv::ocl::pyrDown(const oclMat& src, oclMat& dst)
{
    CV_Assert(src.depth() <= CV_32F && src.channels() <= 4);

	//src.step = src.rows;

    dst.create((src.rows + 1) / 2, (src.cols + 1) / 2, src.type());

	//dst.step = dst.rows;

    pyrdown_run(src, dst);
}

