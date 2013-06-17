/* 
  License
*/

#include "precomp.hpp"

namespace cv
{

SCD::SCD()
{
	int nAngularBins = 5;
    int nRadialBins = 4;
    bool logscale = true;
}

SCD::SCD(int _nAngularBins, int _nRadialBins, bool _logscale)
{
    nAngularBins = _nAngularBins;
    nRadialBins = _nRadialBins;
    logscale = _logscale;
}

int SCD::descriptorSize() const { return nAngularBins*nRadialBins; }

void SCD::operator()(InputArray img, CV_OUT std::vector<KeyPoint>& keypoints,
                    OutputArray descriptors) const
{
    
}

}
