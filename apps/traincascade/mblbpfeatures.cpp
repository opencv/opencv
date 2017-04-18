#include "opencv2/core/core.hpp"
#include "opencv2/core/internal.hpp"

#include "mblbpfeatures.h"
#include "cascadeclassifier.h"

using namespace std;
using namespace cv;

CvMBLBPFeatureParams::CvMBLBPFeatureParams()
{
    name = MBLBPF_NAME;
}

void CvMBLBPEvaluator::init(const CvFeatureParams *_featureParams, int _maxSampleCount, Size _winSize){
    CV_Assert( _maxSampleCount > 0);
    sum.create((int)_maxSampleCount, (_winSize.width + 1) * (_winSize.height + 1), CV_32SC1);
    CvFeatureEvaluator::init( _featureParams, _maxSampleCount, _winSize );
}


CvMBLBPEvaluator::Feature::Feature()
{
    rect = cvRect(0, 0, 0, 0);
}

void CvMBLBPEvaluator::generateFeatures()
{
    cout<<"generating features"<<endl;
    int offset = winSize.width+1;
    int count = 0;
    this->numFeatures = 0;
    for(int round=0; round < 2; round++)
    {
        for( int x = 0; x < winSize.width; x++ )
            for( int y = 0; y < winSize.height; y++ )
                for( int w = 1; w <= winSize.width / 3; w++ )
                    for( int h = 1; h <= winSize.height / 3; h++ )
                    {
                        if((x+3*w <= winSize.width) && (y+3*h <= winSize.height) )
                        {
                            if ( round==0 )
                                this->numFeatures++; //count how many features
                            else //set features
                            {
                                features.push_back(Feature(x,y,w,h));
                            }
                        }
                    }
        

        if(round==0) //alloc memory for features in the first round
        {
            cout << "numFeatures = " << this->numFeatures << endl;
            int mem_size = sizeof(MBLBPWeakf)*numFeatures;
            //this->features.mblbpfeatures = (MBLBPWeakf*) cvAlloc(mem_size);
            this->featuresMask = (bool*) cvAlloc(sizeof(bool)*numFeatures);

            if( this->featuresMask == NULL)
            {
                cerr << "Can not alloc memory. size = " <<  mem_size << endl;
                
            }
            else
            {
                //memset(this->features, 0, mem_size);
                memset(this->featuresMask, 0, sizeof(bool)*this->numFeatures);
            }

        } //end if round
        else //check 
        {
            if( count != this->numFeatures)
            {
                cerr << "count("<< count <<") != numFeatures("<< this->numFeatures <<")" << endl;
                //cvFree( &(this->features) );
                //cvFree( &(this->featuresMask) );
                
            }
        }//end else round
        
    }

    //cal fast pointer offsets for each features
    for(int i = 0; i < this->numFeatures; i++)
    {
        int x = this->features[i].x;
        int y = this->features[i].y;
        int w = this->features[i].cellwidth;
        int h = this->features[i].cellheight;

        this->features[i].offsets[ 0] = y * offset + (x      );
        this->features[i].offsets[ 1] = y * offset + (x + w  );
        this->features[i].offsets[ 2] = y * offset + (x + w*2);
        this->features[i].offsets[ 3] = y * offset + (x + w*3);
        
        this->features[i].offsets[ 4] = (y+h) * offset + (x      );
        this->features[i].offsets[ 5] = (y+h) * offset + (x + w  );
        this->features[i].offsets[ 6] = (y+h) * offset + (x + w*2);
        this->features[i].offsets[ 7] = (y+h) * offset + (x + w*3);
        
        this->features[i].offsets[ 8] = (y+h*2) * offset + (x      );
        this->features[i].offsets[ 9] = (y+h*2) * offset + (x + w  );
        this->features[i].offsets[10] = (y+h*2) * offset + (x + w*2);
        this->features[i].offsets[11] = (y+h*2) * offset + (x + w*3);
        
        this->features[i].offsets[12] = (y+h*3) * offset + (x      );
        this->features[i].offsets[13] = (y+h*3) * offset + (x + w  );
        this->features[i].offsets[14] = (y+h*3) * offset + (x + w*2);
        this->features[i].offsets[15] = (y+h*3) * offset + (x + w*3);
    }
}

void CvMBLBPEvaluator::setImage(const Mat &img, int idx,bool isSum)
{
    Mat sum;

    if( idx >= samplesLBP.cols )
    {
        cerr << "The sample index is out of rangle: index=" << idx << "; Range [0, " << samplesLBP.cols << ")" << endl;
    }
    unsigned char * pLBP = samplesLBP.ptr(0)+idx;

    if(isSum)
        sum = img;
    else
        integral(img, sum);

    if(sum.cols != this->winSize.width + 1 || 
       sum.rows != this->winSize.height + 1)
    {
        cerr << "The sum image's size is incorrect." << endl;
    }

    for(int i = 0; i < numFeatures; i++)
    {
        pLBP[i*samplesLBP.step] = LBPcode(sum, features[i].offsets);
    }
}

CvMBLBPEvaluator::Feature::Feature( int offset, int x, int y, int _blockWidth, int _blockHeight )
{
    Rect tr = rect = cvRect(x, y, _blockWidth, _blockHeight);
    CV_SUM_OFFSETS( p[0], p[1], p[4], p[5], tr, offset )
    tr.x += 2*rect.width;
    CV_SUM_OFFSETS( p[2], p[3], p[6], p[7], tr, offset )
    tr.y +=2*rect.height;
    CV_SUM_OFFSETS( p[10], p[11], p[14], p[15], tr, offset )
    tr.x -= 2*rect.width;
    CV_SUM_OFFSETS( p[8], p[9], p[12], p[13], tr, offset )
}

CvMBLBPEvaluator::Feature::Feature( int x, int y, int cellwidth, int cellheight )
{
    this->x = x;
    this->y = y;
    this->cellwidth = cellwidth;
    this->cellheight = cellheight;
}

void CvMBLBPEvaluator::Feature::write(FileStorage &fs) const
{
    fs << CC_RECT << "[:" << rect.x << rect.y << rect.width << rect.height << "]";
}

void CvMBLBPEvaluator::writeFeatures(FileStorage &fs, const Mat& featureMap ) const
{
    _writeFeatures( features, fs, featureMap );
}