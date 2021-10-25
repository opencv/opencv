// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc/types_c.h>

#include "../precomp.hpp"
#include "utils.hpp"

namespace cv
{

class OdometryFrame::Impl
{
public:
    Impl() {};
    ~Impl() {};
    virtual void setImage(InputArray  image) = 0;
    virtual void getImage(OutputArray image) = 0;
    virtual void getGrayImage(OutputArray image) = 0;
    virtual void setDepth(InputArray  depth) = 0;
    virtual void getDepth(OutputArray depth) = 0;
    virtual void setMask(InputArray  mask) = 0;
    virtual void getMask(OutputArray mask) = 0;
    virtual void setNormals(InputArray  normals) = 0;
    virtual void getNormals(OutputArray normals) = 0;
    virtual void   setPyramidLevel(size_t _nLevels, OdometryFramePyramidType oftype) = 0;
    virtual void   setPyramidLevels(size_t _nLevels) = 0;
    virtual size_t getPyramidLevels(OdometryFramePyramidType oftype) = 0;
    virtual void setPyramidAt(InputArray  img,
        OdometryFramePyramidType pyrType, size_t level) = 0;
    virtual void getPyramidAt(OutputArray img,
        OdometryFramePyramidType pyrType, size_t level) = 0;
};

template<typename TMat>
class OdometryFrameImplTMat : public OdometryFrame::Impl
{
public:
	OdometryFrameImplTMat();
	~OdometryFrameImplTMat() {};

	virtual void setImage(InputArray  image) override;
	virtual void getImage(OutputArray image) override;
	virtual void getGrayImage(OutputArray image) override;
	virtual void setDepth(InputArray  depth) override;
	virtual void getDepth(OutputArray depth) override;
	virtual void setMask(InputArray  mask) override;
	virtual void getMask(OutputArray mask) override;
	virtual void setNormals(InputArray  normals) override;
	virtual void getNormals(OutputArray normals) override;
	virtual void   setPyramidLevel(size_t _nLevels, OdometryFramePyramidType oftype) override;
	virtual void   setPyramidLevels(size_t _nLevels) override;
	virtual size_t getPyramidLevels(OdometryFramePyramidType oftype) override;
	virtual void setPyramidAt(InputArray  img,
		OdometryFramePyramidType pyrType, size_t level) override;
	virtual void getPyramidAt(OutputArray img,
		OdometryFramePyramidType pyrType, size_t level) override;

private:
	void findMask(InputArray image);

	TMat image;
	TMat imageGray;
	TMat depth;
	TMat mask;
	TMat normals;
	std::vector< std::vector<TMat> > pyramids;
};

OdometryFrame::OdometryFrame()
{
    this->impl = makePtr<OdometryFrameImplTMat<Mat>>();
};

OdometryFrame::OdometryFrame(OdometryFrameStoreType matType)
{
	if (matType == OdometryFrameStoreType::UMAT)
		this->impl = makePtr<OdometryFrameImplTMat<UMat>>();
	else
		this->impl = makePtr<OdometryFrameImplTMat<Mat>>();
};

void OdometryFrame::setImage(InputArray  image) { this->impl->setImage(image); }
void OdometryFrame::getImage(OutputArray image) const { this->impl->getImage(image); }
void OdometryFrame::getGrayImage(OutputArray image) const { this->impl->getGrayImage(image); }
void OdometryFrame::setDepth(InputArray  depth) { this->impl->setDepth(depth); }
void OdometryFrame::getDepth(OutputArray depth) const { this->impl->getDepth(depth); }
void OdometryFrame::setMask(InputArray  mask) { this->impl->setMask(mask); }
void OdometryFrame::getMask(OutputArray mask) const { this->impl->getMask(mask); }
void OdometryFrame::setNormals(InputArray  normals) { this->impl->setNormals(normals); }
void OdometryFrame::getNormals(OutputArray normals) const { this->impl->getNormals(normals); }
void OdometryFrame::setPyramidLevel(size_t _nLevels, OdometryFramePyramidType oftype)
{
    this->impl->setPyramidLevel(_nLevels, oftype);
}
void OdometryFrame::setPyramidLevels(size_t _nLevels) { this->impl->setPyramidLevels(_nLevels); }
size_t OdometryFrame::getPyramidLevels(OdometryFramePyramidType oftype) const { return this->impl->getPyramidLevels(oftype); }
void OdometryFrame::setPyramidAt(InputArray  img, OdometryFramePyramidType pyrType, size_t level)
{
    this->impl->setPyramidAt(img, pyrType, level);
}
void OdometryFrame::getPyramidAt(OutputArray img, OdometryFramePyramidType pyrType, size_t level) const
{
    this->impl->getPyramidAt(img, pyrType, level);
}

template<typename TMat>
OdometryFrameImplTMat<TMat>::OdometryFrameImplTMat()
	: pyramids(OdometryFramePyramidType::N_PYRAMIDS)
{
};

template<typename TMat>
void OdometryFrameImplTMat<TMat>::setImage(InputArray _image)
{
	this->image = getTMat<TMat>(_image);
	Mat gray;
    if (_image.channels() != 1)
        cvtColor(_image, gray, COLOR_BGR2GRAY, 1);
    else
        gray = getTMat<Mat>(_image);
    gray.convertTo(gray, CV_8UC1);
	this->imageGray = getTMat<TMat>(gray);
}

template<typename TMat>
void OdometryFrameImplTMat<TMat>::getImage(OutputArray _image)
{
	_image.assign(this->image);
}

template<typename TMat>
void OdometryFrameImplTMat<TMat>::getGrayImage(OutputArray _image)
{
	_image.assign(this->imageGray);
}

template<typename TMat>
void OdometryFrameImplTMat<TMat>::setDepth(InputArray _depth)
{

    Mat depth_tmp, depth_flt;
    depth_tmp = _depth.getMat();
    // Odometry works with depth values in range [0, 10)
    // if the max depth value more than 10, it needs to devide by 5000
    double max;
    cv::minMaxLoc(depth_tmp, NULL, &max);
    if (max > 10)
    {
        depth_tmp.convertTo(depth_flt, CV_32FC1, 1.f / 5000.f);
        depth_flt.setTo(std::numeric_limits<float>::quiet_NaN(), depth_flt < FLT_EPSILON);
        depth_tmp = depth_flt;
    }
    this->depth = getTMat<TMat>(depth_tmp);
	this->findMask(_depth);
}

template<typename TMat>
void OdometryFrameImplTMat<TMat>::getDepth(OutputArray _depth)
{
	_depth.assign(this->depth);
}

template<typename TMat>
void OdometryFrameImplTMat<TMat>::setMask(InputArray _mask)
{
	this->mask = getTMat<TMat>(_mask);
}

template<typename TMat>
void OdometryFrameImplTMat<TMat>::getMask(OutputArray _mask)
{
	_mask.assign(this->mask);
}

template<typename TMat>
void OdometryFrameImplTMat<TMat>::setNormals(InputArray _normals)
{
	this->normals = getTMat<TMat>(_normals);
}

template<typename TMat>
void OdometryFrameImplTMat<TMat>::getNormals(OutputArray _normals)
{
	_normals.assign(this->normals);
}

template<typename TMat>
void OdometryFrameImplTMat<TMat>::setPyramidLevel(size_t _nLevels, OdometryFramePyramidType oftype)
{
	if (oftype < OdometryFramePyramidType::N_PYRAMIDS)
		pyramids[oftype].resize(_nLevels, TMat());
	else
		std::cout << "Incorrect type." << std::endl;

}

template<typename TMat>
void OdometryFrameImplTMat<TMat>::setPyramidLevels(size_t _nLevels)
{
	for (auto& p : pyramids)
	{
		p.resize(_nLevels, TMat());
	}
}

template<typename TMat>
size_t OdometryFrameImplTMat<TMat>::getPyramidLevels(OdometryFramePyramidType oftype)
{
	if (oftype < OdometryFramePyramidType::N_PYRAMIDS)
		return pyramids[oftype].size();
	else
		return 0;
}

template<typename TMat>
void OdometryFrameImplTMat<TMat>::setPyramidAt(InputArray  _img, OdometryFramePyramidType pyrType, size_t level)
{
    //CV_Assert(_img.channels() != 4);
	TMat img = getTMat<TMat>(_img);
	pyramids[pyrType][level] = img;
}

template<typename TMat>
void OdometryFrameImplTMat<TMat>::getPyramidAt(OutputArray _img, OdometryFramePyramidType pyrType, size_t level)
{
	TMat img = pyramids[pyrType][level];
	_img.assign(img);
}

template<typename TMat>
void OdometryFrameImplTMat<TMat>::findMask(InputArray _depth)
{
	Mat depth = _depth.getMat();
	Mat mask(depth.size(), CV_8UC1, Scalar(255));
	for (int y = 0; y < depth.rows; y++)
        for (int x = 0; x < depth.cols; x++)
        {
            if (cvIsNaN(depth.at<float>(y, x)) || depth.at<float>(y, x) <= FLT_EPSILON)
                mask.at<uchar>(y, x) = 0;
        }
	this->setMask(mask);
}

}
