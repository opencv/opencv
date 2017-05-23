#include "changed_pixels_widget.hpp"

#include <QLabel>
#include <QVBoxLayout>


#include "../types.hpp"


//forward
template<int Depth>
void changedPixelImage(const cv::Mat& mat0, const cv::Mat& mat1, cv::Mat& out);

template<int Depth, int Channels>
void changedPixelImage(const cv::Mat& mat0, const cv::Mat& mat1, cv::Mat& out);

void changedPixelImage(const cv::Mat& mat0, const cv::Mat& mat1, cv::Mat& out)
{
	// need same size
	if (mat0.size() != mat1.size())
	{
		return;
	}

	//need same # of channels
	if (mat0.channels()!=mat1.channels())
	{
		return;
	}
	//need same depth
	if (mat0.depth()!=mat1.depth())
	{
		return;
	}

	//split depth
	switch(mat0.depth())
	{
		case CV_8U :changedPixelImage<CV_8U >(mat0,mat1,out); break;
		case CV_8S :changedPixelImage<CV_8S >(mat0,mat1,out); break;
		case CV_16U:changedPixelImage<CV_16U>(mat0,mat1,out); break;
		case CV_16S:changedPixelImage<CV_16S>(mat0,mat1,out); break;
		case CV_32S:changedPixelImage<CV_32S>(mat0,mat1,out); break;
		case CV_32F:changedPixelImage<CV_32F>(mat0,mat1,out); break;
		case CV_64F:changedPixelImage<CV_64F>(mat0,mat1,out); break;
	}
}

template<int Depth>
void changedPixelImage(const cv::Mat& mat0, const cv::Mat& mat1, cv::Mat& out)
{
	switch(mat0.channels())
	{
		case 1  :changedPixelImage<Depth,1 >(mat0,mat1,out); break;
		case 2  :changedPixelImage<Depth,2 >(mat0,mat1,out); break;
		case 3  :changedPixelImage<Depth,3 >(mat0,mat1,out); break;
		case 4  :changedPixelImage<Depth,4 >(mat0,mat1,out); break;
		case 5  :changedPixelImage<Depth,5 >(mat0,mat1,out); break;
		case 6  :changedPixelImage<Depth,6 >(mat0,mat1,out); break;
		case 7  :changedPixelImage<Depth,7 >(mat0,mat1,out); break;
		case 8  :changedPixelImage<Depth,8 >(mat0,mat1,out); break;
		case 9  :changedPixelImage<Depth,9 >(mat0,mat1,out); break;
		case 10 :changedPixelImage<Depth,10>(mat0,mat1,out); break;
	}
}

template<int Depth, int Channels>
void changedPixelImage(const cv::Mat& mat0, const cv::Mat& mat1, cv::Mat& out)
{
	using PixelInType=const cvv::qtutil::PixelType<Depth,Channels>;
	using PixelOutType=cvv::qtutil::DepthType<CV_8U>;


	cv::Mat result=cv::Mat::zeros(mat0.rows, mat0.cols,CV_8U);
	bool same;
	PixelInType* in0;
	PixelInType* in1;
	for(int i=0;i<result.rows;i++)
	{
		for(int j=0;j<result.cols;j++)
		{
			same=true;
			in0=&(mat0.at<PixelInType>(i,j));
			in1=&(mat1.at<PixelInType>(i,j));

			for(int chan=0;chan<Channels;chan++)
			{
				same = same && (in0[chan]==in1[chan]);
			}
			//same color => set pixel color white
			if(same)
			{
				(result.at<PixelOutType>(i,j))=255;
			}
		}
	}
	out=result;
}

namespace cvv
{
namespace qtutil
{


ChangedPixelsWidget::ChangedPixelsWidget(QWidget* parent): FilterFunctionWidget<2, 1>{parent}
{
	auto lay=util::make_unique<QVBoxLayout>();
	lay->addWidget(util::make_unique<QLabel>("Changed pixels will be black<br>"
						 " unchanged white.").release());
	setLayout(lay.release());
}

void ChangedPixelsWidget::applyFilter(ChangedPixelsWidget::InputArray in,
				      ChangedPixelsWidget::OutputArray out) const
{
	changedPixelImage(in.at(0).get(),in.at(1).get(),out.at(0).get());
}

std::pair<bool, QString> ChangedPixelsWidget::checkInput(InputArray in) const
{
	if (in.at(0).get().size() != in.at(1).get().size())
	{
		return std::make_pair(false, "images need to have same size");
	}

	size_t inChannels = in.at(0).get().channels();

	if (inChannels != static_cast<size_t>(in.at(1).get().channels()))
	{
		return std::make_pair(
		    false, "images need to have same number of channels");
	}

	if (inChannels>10 || inChannels<1)
	{
		return std::make_pair(
		    false, "images need to have 1 up to 10 channels");
	}

	int i0depth=in.at(0).get().depth();

	if (i0depth!=in.at(1).get().depth())
	{
		return std::make_pair(
		    false, "images need to have the same depth");
	}

	if (!((i0depth==CV_8U)||(i0depth==CV_8S)||(i0depth==CV_16U)||(i0depth==CV_16S)||
		(i0depth==CV_32S)||(i0depth==CV_32F)||(i0depth==CV_64F)))
	{
		return std::make_pair(false, "images have unknown depth");
	}

	return std::make_pair(true, "");
}


} // qtutil
} // cvv
