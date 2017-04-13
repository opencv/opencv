#include "opencv2/core/core.hpp"
#include "opencv2/core/internal.hpp"

#include "mblbpfeatures.h"
#include "cascadeclassifier.h"

using namespace std;
using namespace cv;


void CvMBLBPEvaluator::generateFeatures()
{
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
                                this->features[count].x = x;
                                this->features[count].y = y;
                                this->features[count].cellwidth = w;
                                this->features[count].cellheight = h;
                                count++;
                            }
                        }
                    }
        

        if(round==0) //alloc memory for features in the first round
        {
            cout << "numFeatures = " << this->numFeatures << endl;
            int mem_size = sizeof(MBLBPWeakf)*numFeatures;
            this->features = (MBLBPWeakf*) cvAlloc(mem_size);
            this->featuresMask = (bool*) cvAlloc(sizeof(bool)*numFeatures);

            if(this->features == NULL || this->featuresMask == NULL)
            {
                cerr << "Can not alloc memory. size = " <<  mem_size << endl;
                return false;
            }
            else
            {
                memset(this->features, 0, mem_size);
                memset(this->featuresMask, 0, sizeof(bool)*this->numFeatures);
            }

        } //end if round
        else //check 
        {
            if( count != this->numFeatures)
            {
                cerr << "count("<< count <<") != numFeatures("<< this->numFeatures <<")" << endl;
                cvFree( &(this->features) );
                cvFree( &(this->featuresMask) );
                return false;
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
    return true;
}

void CvMBLBPEvaluator::setImage(const Mat &img, int idx,bool isSum)
{
    Mat sum;

    if( idx >= samplesLBP.cols )
    {
        cerr << "The sample index is out of rangle: index=" << idx << "; Range [0, " << samplesLBP.cols << ")" << endl;
        return false;
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
        return false;
    }

    for(int i = 0; i < numFeatures; i++)
    {
        pLBP[i*samplesLBP.step] = LBPcode(sum, features[i].offsets);
    }

    return true;
}

bool CvMBLBPEvaluator::loadPosSamples(string _posFilename){
    //positive sampls
    this->numSamples = 0;
    this->numSamples += this->numPos;
    this->numSamples += this->numNeg;

    //alloc memory for samples
    this->labels = Mat::zeros(1, numSamples, CV_8UC1);
    this->samplesLBP.create(this->numFeatures, numSamples, CV_8UC1);
    if(labels.empty() || samplesLBP.empty())
    {
        cerr << "Cannot alloc memory for samples." << endl;
        return false;
    }


    //load postive samples and convert them to integral images
    {
        Mat img(this->winSize, CV_8UC1);
        posReader.create(_posFilename);

        int currPos = 0;
        while( posReader.get(img))
        {
            //imshow("pos", img);
            //waitKey(10);

            labels.at<unsigned char>(0, currPos) = 1;
            setImage(img, currPos, false);
            currPos++;

            if(currPos >= this->numPos)
                break;
        }

        if( currPos != this->numPos)
        {
            cerr << "Read vec file " << _posFilename << " error" << endl;
            cerr << "There are " << this->numPos << " samples, but only " << currPos << " loaded." << endl;
            return false;
        }

        cout << currPos << " positive samples loaded." << endl;
    }

    return true;

}