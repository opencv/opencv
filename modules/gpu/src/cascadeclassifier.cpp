/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other GpuMaterials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or bpied warranties, including, but not limited to, the bpied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "precomp.hpp"

using namespace cv;
using namespace cv::gpu;
using namespace std;


#if !defined (HAVE_CUDA) || (defined(_MSC_VER) && _MSC_VER != 1500) || !defined(_MSC_VER)

cv::gpu::CascadeClassifier_GPU::CascadeClassifier_GPU()  { throw_nogpu(); }
cv::gpu::CascadeClassifier_GPU::CascadeClassifier_GPU(const string&)  { throw_nogpu(); }
cv::gpu::CascadeClassifier_GPU::~CascadeClassifier_GPU()  { throw_nogpu(); }

bool cv::gpu::CascadeClassifier_GPU::empty() const { throw_nogpu(); return true; }
bool cv::gpu::CascadeClassifier_GPU::load(const string&)  { throw_nogpu(); return true; }
Size cv::gpu::CascadeClassifier_GPU::getClassifierSize() const { throw_nogpu(); return Size(); }

int cv::gpu::CascadeClassifier_GPU::detectMultiScale( const GpuMat& , GpuMat& , double , int , Size)  { throw_nogpu(); return 0; }

#if defined (HAVE_CUDA)
	NCVStatus loadFromXML(const string&, HaarClassifierCascadeDescriptor&, vector<HaarStage64>&, 
						  vector<HaarClassifierNode128>&, vector<HaarFeature64>&) { throw_nogpu(); return NCVStatus(); }

	void groupRectangles(vector<NcvRect32u>&, int, double, vector<Ncv32u>*) { throw_nogpu(); }
#endif

#else

struct cv::gpu::CascadeClassifier_GPU::CascadeClassifierImpl
{    
    CascadeClassifierImpl(const string& filename) : lastAllocatedFrameSize(-1, -1)
    {
        ncvSetDebugOutputHandler(NCVDebugOutputHandler);            
        if (ncvStat != load(filename))
            CV_Error(CV_GpuApiCallError, "Error in GPU cacade load");
    }    
    NCVStatus process(const GpuMat& src, GpuMat& objects, float scaleStep, int minNeighbors, bool findLargestObject, bool visualizeInPlace, NcvSize32u ncvMinSize, /*out*/unsigned int& numDetections)
    {   
        calculateMemReqsAndAllocate(src.size());        

        NCVMemPtr src_beg;
        src_beg.ptr = (void*)src.ptr<Ncv8u>();
        src_beg.memtype = NCVMemoryTypeDevice;

        NCVMemSegment src_seg;
        src_seg.begin = src_beg;
        src_seg.size  = src.step * src.rows;

        NCVMatrixReuse<Ncv8u> d_src(src_seg, devProp.textureAlignment, src.cols, src.rows, src.step, true);        
		ncvAssertReturn(d_src.isMemReused(), NCV_ALLOCATOR_BAD_REUSE);
        
        //NCVMatrixAlloc<Ncv8u> d_src(*gpuAllocator, src.cols, src.rows);
        //ncvAssertReturn(d_src.isMemAllocated(), NCV_ALLOCATOR_BAD_ALLOC);

        //NCVMatrixAlloc<Ncv8u> h_src(*cpuAllocator, src.cols, src.rows);
        //ncvAssertReturn(h_src.isMemAllocated(), NCV_ALLOCATOR_BAD_ALLOC);

        CV_Assert(objects.rows == 1);

        NCVMemPtr objects_beg;
        objects_beg.ptr = (void*)objects.ptr<NcvRect32u>();
        objects_beg.memtype = NCVMemoryTypeDevice;

        NCVMemSegment objects_seg;
        objects_seg.begin = objects_beg;
        objects_seg.size = objects.step * objects.rows;
        NCVVectorReuse<NcvRect32u> d_rects(objects_seg, objects.cols);
		ncvAssertReturn(d_rects.isMemReused(), NCV_ALLOCATOR_BAD_REUSE);
        //NCVVectorAlloc<NcvRect32u> d_rects(*gpuAllocator, 100);        
        //ncvAssertReturn(d_rects.isMemAllocated(), NCV_ALLOCATOR_BAD_ALLOC);        
            
        NcvSize32u roi;
        roi.width = d_src.width();
        roi.height = d_src.height();

        Ncv32u flags = 0;
        flags |= findLargestObject? NCVPipeObjDet_FindLargestObject : 0;
        flags |= visualizeInPlace ? NCVPipeObjDet_VisualizeInPlace  : 0;
        
        ncvStat = ncvDetectObjectsMultiScale_device(
            d_src, roi, d_rects, numDetections, haar, *h_haarStages,
            *d_haarStages, *d_haarNodes, *d_haarFeatures,
            ncvMinSize,
            minNeighbors,
            scaleStep, 1,
            flags,
            *gpuAllocator, *cpuAllocator, devProp.major, devProp.minor,  0);
        ncvAssertReturnNcvStat(ncvStat);
        ncvAssertCUDAReturn(cudaStreamSynchronize(0), NCV_CUDA_ERROR);
                       
        return NCV_SUCCESS;
    }
    ////
    
    NcvSize32u getClassifierSize() const  { return haar.ClassifierSize; }
    cv::Size getClassifierCvSize() const { return cv::Size(haar.ClassifierSize.width, haar.ClassifierSize.height); }
private:

    static void NCVDebugOutputHandler(const char* msg) { CV_Error(CV_GpuApiCallError, msg); }

    NCVStatus load(const string& classifierFile)
    {        
        int devId = cv::gpu::getDevice();           
        ncvAssertCUDAReturn(cudaGetDeviceProperties(&devProp, devId), NCV_CUDA_ERROR);

        // Load the classifier from file (assuming its size is about 1 mb) using a simple allocator
        gpuCascadeAllocator = new NCVMemNativeAllocator(NCVMemoryTypeDevice);        
        cpuCascadeAllocator = new NCVMemNativeAllocator(NCVMemoryTypeHostPinned);

        ncvAssertPrintReturn(gpuCascadeAllocator->isInitialized(), "Error creating cascade GPU allocator", NCV_CUDA_ERROR);
        ncvAssertPrintReturn(cpuCascadeAllocator->isInitialized(), "Error creating cascade CPU allocator", NCV_CUDA_ERROR);

        Ncv32u haarNumStages, haarNumNodes, haarNumFeatures;
        ncvStat = ncvHaarGetClassifierSize(classifierFile, haarNumStages, haarNumNodes, haarNumFeatures);
        ncvAssertPrintReturn(ncvStat == NCV_SUCCESS, "Error reading classifier size (check the file)", NCV_FILE_ERROR);

        h_haarStages   = new NCVVectorAlloc<HaarStage64>(*cpuCascadeAllocator, haarNumStages);        
        h_haarNodes    = new NCVVectorAlloc<HaarClassifierNode128>(*cpuCascadeAllocator, haarNumNodes);
        h_haarFeatures = new NCVVectorAlloc<HaarFeature64>(*cpuCascadeAllocator, haarNumFeatures);

        ncvAssertPrintReturn(h_haarStages->isMemAllocated(), "Error in cascade CPU allocator", NCV_CUDA_ERROR);
        ncvAssertPrintReturn(h_haarNodes->isMemAllocated(), "Error in cascade CPU allocator", NCV_CUDA_ERROR);        
        ncvAssertPrintReturn(h_haarFeatures->isMemAllocated(), "Error in cascade CPU allocator", NCV_CUDA_ERROR);

        ncvStat = ncvHaarLoadFromFile_host(classifierFile, haar, *h_haarStages, *h_haarNodes, *h_haarFeatures);
        ncvAssertPrintReturn(ncvStat == NCV_SUCCESS, "Error loading classifier", NCV_FILE_ERROR);

        d_haarStages   = new NCVVectorAlloc<HaarStage64>(*gpuCascadeAllocator, haarNumStages);
        d_haarNodes    = new NCVVectorAlloc<HaarClassifierNode128>(*gpuCascadeAllocator, haarNumNodes);
        d_haarFeatures = new NCVVectorAlloc<HaarFeature64>(*gpuCascadeAllocator, haarNumFeatures);

        ncvAssertPrintReturn(d_haarStages->isMemAllocated(), "Error in cascade GPU allocator", NCV_CUDA_ERROR);
        ncvAssertPrintReturn(d_haarNodes->isMemAllocated(), "Error in cascade GPU allocator", NCV_CUDA_ERROR);                        
        ncvAssertPrintReturn(d_haarFeatures->isMemAllocated(), "Error in cascade GPU allocator", NCV_CUDA_ERROR);

        ncvStat = h_haarStages->copySolid(*d_haarStages, 0);
        ncvAssertPrintReturn(ncvStat == NCV_SUCCESS, "Error copying cascade to GPU", NCV_CUDA_ERROR);
        ncvStat = h_haarNodes->copySolid(*d_haarNodes, 0);
        ncvAssertPrintReturn(ncvStat == NCV_SUCCESS, "Error copying cascade to GPU", NCV_CUDA_ERROR);
        ncvStat = h_haarFeatures->copySolid(*d_haarFeatures, 0);
        ncvAssertPrintReturn(ncvStat == NCV_SUCCESS, "Error copying cascade to GPU", NCV_CUDA_ERROR);    

        return NCV_SUCCESS;
    }
    ////

    NCVStatus calculateMemReqsAndAllocate(const Size& frameSize)
    {        
        if (lastAllocatedFrameSize == frameSize)
            return NCV_SUCCESS;

        // Calculate memory requirements and create real allocators
        NCVMemStackAllocator gpuCounter(devProp.textureAlignment);
        NCVMemStackAllocator cpuCounter(devProp.textureAlignment);

        ncvAssertPrintReturn(gpuCounter.isInitialized(), "Error creating GPU memory counter", NCV_CUDA_ERROR);        
        ncvAssertPrintReturn(cpuCounter.isInitialized(), "Error creating CPU memory counter", NCV_CUDA_ERROR);
        
        NCVMatrixAlloc<Ncv8u> d_src(gpuCounter, frameSize.width, frameSize.height);
        NCVMatrixAlloc<Ncv8u> h_src(cpuCounter, frameSize.width, frameSize.height);

        ncvAssertReturn(d_src.isMemAllocated(), NCV_ALLOCATOR_BAD_ALLOC);        
        ncvAssertReturn(h_src.isMemAllocated(), NCV_ALLOCATOR_BAD_ALLOC);

        NCVVectorAlloc<NcvRect32u> d_rects(gpuCounter, 100);        
        ncvAssertReturn(d_rects.isMemAllocated(), NCV_ALLOCATOR_BAD_ALLOC);

        NcvSize32u roi;
        roi.width = d_src.width();
        roi.height = d_src.height();
        Ncv32u numDetections;
        ncvStat = ncvDetectObjectsMultiScale_device(d_src, roi, d_rects, numDetections, haar, *h_haarStages,
            *d_haarStages, *d_haarNodes, *d_haarFeatures, haar.ClassifierSize, 4, 1.2f, 1, 0, gpuCounter, cpuCounter, devProp.major, devProp.minor, 0);

        ncvAssertReturnNcvStat(ncvStat);
        ncvAssertCUDAReturn(cudaStreamSynchronize(0), NCV_CUDA_ERROR);
                      
        gpuAllocator = new NCVMemStackAllocator(NCVMemoryTypeDevice, gpuCounter.maxSize(), devProp.textureAlignment);        
        cpuAllocator = new NCVMemStackAllocator(NCVMemoryTypeHostPinned, cpuCounter.maxSize(), devProp.textureAlignment);

        ncvAssertPrintReturn(gpuAllocator->isInitialized(), "Error creating GPU memory allocator", NCV_CUDA_ERROR);
        ncvAssertPrintReturn(cpuAllocator->isInitialized(), "Error creating CPU memory allocator", NCV_CUDA_ERROR);        
        return NCV_SUCCESS;
    }
    //// 

    cudaDeviceProp devProp;
    NCVStatus ncvStat;

    Ptr<NCVMemNativeAllocator> gpuCascadeAllocator;        
    Ptr<NCVMemNativeAllocator> cpuCascadeAllocator;

    Ptr<NCVVectorAlloc<HaarStage64> >           h_haarStages;        
    Ptr<NCVVectorAlloc<HaarClassifierNode128> > h_haarNodes;
    Ptr<NCVVectorAlloc<HaarFeature64> >         h_haarFeatures;

    HaarClassifierCascadeDescriptor haar;

    Ptr<NCVVectorAlloc<HaarStage64> >           d_haarStages;
    Ptr<NCVVectorAlloc<HaarClassifierNode128> > d_haarNodes;
    Ptr<NCVVectorAlloc<HaarFeature64> >         d_haarFeatures;

    Size lastAllocatedFrameSize;

    Ptr<NCVMemStackAllocator> gpuAllocator;        
    Ptr<NCVMemStackAllocator> cpuAllocator;
};



cv::gpu::CascadeClassifier_GPU::CascadeClassifier_GPU() : findLargestObject(false), visualizeInPlace(false), impl(0) {}
cv::gpu::CascadeClassifier_GPU::CascadeClassifier_GPU(const string& filename) : findLargestObject(false), visualizeInPlace(false), impl(0) { load(filename); }
cv::gpu::CascadeClassifier_GPU::~CascadeClassifier_GPU() { release(); }
bool cv::gpu::CascadeClassifier_GPU::empty() const { return impl == 0; }

void cv::gpu::CascadeClassifier_GPU::release() { if (impl) { delete impl; impl = 0; } }

bool cv::gpu::CascadeClassifier_GPU::load(const string& filename)
{         
    release();
    impl = new CascadeClassifierImpl(filename);
    return !this->empty();    
}

Size cv::gpu::CascadeClassifier_GPU::getClassifierSize() const
{
    return this->empty() ? Size() : impl->getClassifierCvSize();
}
                            
int cv::gpu::CascadeClassifier_GPU::detectMultiScale( const GpuMat& image, GpuMat& objectsBuf, double scaleFactor, int minNeighbors, Size minSize)
{     
    CV_Assert( scaleFactor > 1 && image.depth() == CV_8U);
    CV_Assert( !this->empty());
        
    const int defaultObjSearchNum = 100;
    if (objectsBuf.empty())
        objectsBuf.create(1, defaultObjSearchNum, DataType<Rect>::type);
    
    NcvSize32u ncvMinSize = impl->getClassifierSize();

    if (ncvMinSize.width < (unsigned)minSize.width && ncvMinSize.height < (unsigned)minSize.height)
    {
        ncvMinSize.width = minSize.width;
        ncvMinSize.height = minSize.height;
    }    
                
    unsigned int numDetections;
    NCVStatus ncvStat = impl->process(image, objectsBuf, (float)scaleFactor, minNeighbors, findLargestObject, visualizeInPlace, ncvMinSize, numDetections);                 
    if (ncvStat != NCV_SUCCESS)
        CV_Error(CV_GpuApiCallError, "Error in face detectioln");

    return numDetections;
}

	struct RectConvert
	{
		Rect operator()(const NcvRect32u& nr) const { return Rect(nr.x, nr.y, nr.width, nr.height); }
		NcvRect32u operator()(const Rect& nr) const 
		{ 
			NcvRect32u rect;
			rect.x = nr.x;
			rect.y = nr.y;
			rect.width = nr.width;
			rect.height = nr.height;
			return rect; 
		}
	};

	void groupRectangles(std::vector<NcvRect32u> &hypotheses, int groupThreshold, double eps, std::vector<Ncv32u> *weights)
	{
		vector<Rect> rects(hypotheses.size());    
		std::transform(hypotheses.begin(), hypotheses.end(), rects.begin(), RectConvert());
	    
		if (weights) 
		{
			vector<int> weights_int;
			weights_int.assign(weights->begin(), weights->end());        
			cv::groupRectangles(rects, weights_int, groupThreshold, eps);
		}
		else
		{   
			cv::groupRectangles(rects, groupThreshold, eps);
		}
		std::transform(rects.begin(), rects.end(), hypotheses.begin(), RectConvert());    
		hypotheses.resize(rects.size());
	}


#if 1 /* loadFromXML implementation switch */

NCVStatus loadFromXML(const std::string &filename, 
                      HaarClassifierCascadeDescriptor &haar, 
                      std::vector<HaarStage64> &haarStages, 
                      std::vector<HaarClassifierNode128> &haarClassifierNodes, 
                      std::vector<HaarFeature64> &haarFeatures)
{
    NCVStatus ncvStat;

    haar.NumStages = 0;
    haar.NumClassifierRootNodes = 0;
    haar.NumClassifierTotalNodes = 0;
    haar.NumFeatures = 0;
    haar.ClassifierSize.width = 0;
    haar.ClassifierSize.height = 0;        
    haar.bHasStumpsOnly = true;
    haar.bNeedsTiltedII = false;
    Ncv32u curMaxTreeDepth;

    std::vector<char> xmlFileCont;   

    std::vector<HaarClassifierNode128> h_TmpClassifierNotRootNodes;
    haarStages.resize(0);
    haarClassifierNodes.resize(0);
    haarFeatures.resize(0);    
    
    Ptr<CvHaarClassifierCascade> oldCascade = (CvHaarClassifierCascade*)cvLoad(filename.c_str(), 0, 0, 0);
    if (oldCascade.empty())
        return NCV_HAAR_XML_LOADING_EXCEPTION;
    
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                                          
    haar.ClassifierSize.width = oldCascade->orig_window_size.width;
    haar.ClassifierSize.height = oldCascade->orig_window_size.height;

    int stagesCound = oldCascade->count;
    for(int s = 0; s < stagesCound; ++s) // by stages
    {
        HaarStage64 curStage;
        curStage.setStartClassifierRootNodeOffset(haarClassifierNodes.size());

        curStage.setStageThreshold(oldCascade->stage_classifier[s].threshold);

        int treesCount = oldCascade->stage_classifier[s].count;
        for(int t = 0; t < treesCount; ++t) // bytrees
        {                                
            Ncv32u nodeId = 0;
            CvHaarClassifier* tree = &oldCascade->stage_classifier[s].classifier[t];

            int nodesCount = tree->count;
            for(int n = 0; n < nodesCount; ++n)  //by features             
            {   
                CvHaarFeature* feature = &tree->haar_feature[n];

                HaarClassifierNode128 curNode;                                        
                curNode.setThreshold(tree->threshold[n]);
                
                HaarClassifierNodeDescriptor32 nodeLeft;
                if ( tree->left[n] <= 0 )
                {   
                    Ncv32f leftVal = tree->alpha[-tree->left[n]];
                    ncvStat = nodeLeft.create(leftVal);
                    ncvAssertReturn(ncvStat == NCV_SUCCESS, ncvStat);                    
                }
                else
                {   
                    Ncv32u leftNodeOffset = tree->left[n];                        
                    nodeLeft.create((Ncv32u)(h_TmpClassifierNotRootNodes.size() + leftNodeOffset - 1));
                    haar.bHasStumpsOnly = false;
                }
                curNode.setLeftNodeDesc(nodeLeft);
                
                HaarClassifierNodeDescriptor32 nodeRight;
                if ( tree->right[n] <= 0 )
                {                                         
                    Ncv32f rightVal = tree->alpha[-tree->right[n]];                        
                    ncvStat = nodeRight.create(rightVal);
                    ncvAssertReturn(ncvStat == NCV_SUCCESS, ncvStat);
                }
                else
                {                                               
                    Ncv32u rightNodeOffset = tree->right[n];                        
                    nodeRight.create((Ncv32u)(h_TmpClassifierNotRootNodes.size() + rightNodeOffset - 1));
                    haar.bHasStumpsOnly = false;
                }
                curNode.setRightNodeDesc(nodeRight);                    

                Ncv32u tiltedVal = feature->tilted;
                haar.bNeedsTiltedII = (tiltedVal != 0);                

                Ncv32u featureId = 0;                    
                for(int l = 0; l < CV_HAAR_FEATURE_MAX; ++l) //by rects
                {                        
                    Ncv32u rectX = feature->rect[l].r.x; 
                    Ncv32u rectY = feature->rect[l].r.y;
                    Ncv32u rectWidth = feature->rect[l].r.width;
                    Ncv32u rectHeight = feature->rect[l].r.height;

                    Ncv32f rectWeight = feature->rect[l].weight;

                    if (rectWeight == 0/* && rectX == 0 &&rectY == 0 && rectWidth == 0 && rectHeight == 0*/)
                        break;

                    HaarFeature64 curFeature;
                    ncvStat = curFeature.setRect(rectX, rectY, rectWidth, rectHeight, haar.ClassifierSize.width, haar.ClassifierSize.height);
                    curFeature.setWeight(rectWeight);
                    ncvAssertReturn(NCV_SUCCESS == ncvStat, ncvStat);
                    haarFeatures.push_back(curFeature);

                    featureId++;
                }

                HaarFeatureDescriptor32 tmpFeatureDesc;
                ncvStat = tmpFeatureDesc.create(haar.bNeedsTiltedII, featureId, haarFeatures.size() - featureId);
                ncvAssertReturn(NCV_SUCCESS == ncvStat, ncvStat);
                curNode.setFeatureDesc(tmpFeatureDesc);

                if (!nodeId)
                {
                    //root node
                    haarClassifierNodes.push_back(curNode);
                    curMaxTreeDepth = 1;
                }
                else
                {
                    //other node
                    h_TmpClassifierNotRootNodes.push_back(curNode);
                    curMaxTreeDepth++;
                }

                nodeId++;
            }               
        }

        curStage.setNumClassifierRootNodes(treesCount);
        haarStages.push_back(curStage);            
    }
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


    //fill in cascade stats
    haar.NumStages = haarStages.size();
    haar.NumClassifierRootNodes = haarClassifierNodes.size();
    haar.NumClassifierTotalNodes = haar.NumClassifierRootNodes + h_TmpClassifierNotRootNodes.size();
    haar.NumFeatures = haarFeatures.size();

    //merge root and leaf nodes in one classifiers array
    Ncv32u offsetRoot = haarClassifierNodes.size();
    for (Ncv32u i=0; i<haarClassifierNodes.size(); i++)
    {
        HaarClassifierNodeDescriptor32 nodeLeft = haarClassifierNodes[i].getLeftNodeDesc();
        if (!nodeLeft.isLeaf())
        {
            Ncv32u newOffset = nodeLeft.getNextNodeOffset() + offsetRoot;
            nodeLeft.create(newOffset);
        }
        haarClassifierNodes[i].setLeftNodeDesc(nodeLeft);

        HaarClassifierNodeDescriptor32 nodeRight = haarClassifierNodes[i].getRightNodeDesc();
        if (!nodeRight.isLeaf())
        {
            Ncv32u newOffset = nodeRight.getNextNodeOffset() + offsetRoot;
            nodeRight.create(newOffset);
        }
        haarClassifierNodes[i].setRightNodeDesc(nodeRight);
    }
    for (Ncv32u i=0; i<h_TmpClassifierNotRootNodes.size(); i++)
    {
        HaarClassifierNodeDescriptor32 nodeLeft = h_TmpClassifierNotRootNodes[i].getLeftNodeDesc();
        if (!nodeLeft.isLeaf())
        {
            Ncv32u newOffset = nodeLeft.getNextNodeOffset() + offsetRoot;
            nodeLeft.create(newOffset);
        }
        h_TmpClassifierNotRootNodes[i].setLeftNodeDesc(nodeLeft);

        HaarClassifierNodeDescriptor32 nodeRight = h_TmpClassifierNotRootNodes[i].getRightNodeDesc();
        if (!nodeRight.isLeaf())
        {
            Ncv32u newOffset = nodeRight.getNextNodeOffset() + offsetRoot;
            nodeRight.create(newOffset);
        }
        h_TmpClassifierNotRootNodes[i].setRightNodeDesc(nodeRight);

        haarClassifierNodes.push_back(h_TmpClassifierNotRootNodes[i]);
    }

    return NCV_SUCCESS;
}

////

#else /* loadFromXML implementation switch */

#include "e:/devNPP-OpenCV/src/external/_rapidxml-1.13/rapidxml.hpp"

NCVStatus loadFromXML(const std::string &filename,
                      HaarClassifierCascadeDescriptor &haar,
                      std::vector<HaarStage64> &haarStages,
                      std::vector<HaarClassifierNode128> &haarClassifierNodes,
                      std::vector<HaarFeature64> &haarFeatures)
{
    NCVStatus ncvStat;

    haar.NumStages = 0;
    haar.NumClassifierRootNodes = 0;
    haar.NumClassifierTotalNodes = 0;
    haar.NumFeatures = 0;
    haar.ClassifierSize.width = 0;
    haar.ClassifierSize.height = 0;
    haar.bNeedsTiltedII = false;
    haar.bHasStumpsOnly = false;

    FILE *fp;
    fopen_s(&fp, filename.c_str(), "r");
    ncvAssertReturn(fp != NULL, NCV_FILE_ERROR);

    //get file size
    fseek(fp, 0, SEEK_END);
    Ncv32u xmlSize = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    //load file to vector
    std::vector<char> xmlFileCont;
    xmlFileCont.resize(xmlSize+1);
    memset(&xmlFileCont[0], 0, xmlSize+1);
    fread_s(&xmlFileCont[0], xmlSize, 1, xmlSize, fp);
    fclose(fp);

    haar.bHasStumpsOnly = true;
    haar.bNeedsTiltedII = false;
    Ncv32u curMaxTreeDepth;

    std::vector<HaarClassifierNode128> h_TmpClassifierNotRootNodes;
    haarStages.resize(0);
    haarClassifierNodes.resize(0);
    haarFeatures.resize(0);

    //XML loading and OpenCV XML classifier syntax verification
    try
    {
        rapidxml::xml_document<> doc;
        doc.parse<0>(&xmlFileCont[0]);

        //opencv_storage
        rapidxml::xml_node<> *parserGlobal = doc.first_node();
        ncvAssertReturn(!strcmp(parserGlobal->name(), "opencv_storage"), NCV_HAAR_XML_LOADING_EXCEPTION);

        //classifier type
        parserGlobal = parserGlobal->first_node();
        ncvAssertReturn(parserGlobal, NCV_HAAR_XML_LOADING_EXCEPTION);
        rapidxml::xml_attribute<> *attr = parserGlobal->first_attribute("type_id");
        ncvAssertReturn(!strcmp(attr->value(), "opencv-haar-classifier"), NCV_HAAR_XML_LOADING_EXCEPTION);

        //classifier size
        parserGlobal = parserGlobal->first_node("size");
        ncvAssertReturn(parserGlobal, NCV_HAAR_XML_LOADING_EXCEPTION);
        sscanf_s(parserGlobal->value(), "%d %d", &(haar.ClassifierSize.width), &(haar.ClassifierSize.height));

        //parse stages
        parserGlobal = parserGlobal->next_sibling("stages");
        ncvAssertReturn(parserGlobal, NCV_HAAR_XML_LOADING_EXCEPTION);
        parserGlobal = parserGlobal->first_node("_");
        ncvAssertReturn(parserGlobal, NCV_HAAR_XML_LOADING_EXCEPTION);

        while (parserGlobal)
        {
            HaarStage64 curStage;
            curStage.setStartClassifierRootNodeOffset(haarClassifierNodes.size());
            Ncv32u tmpNumClassifierRootNodes = 0;

            rapidxml::xml_node<> *parserStageThreshold = parserGlobal->first_node("stage_threshold");
            ncvAssertReturn(parserStageThreshold, NCV_HAAR_XML_LOADING_EXCEPTION);
            Ncv32f tmpStageThreshold;
            sscanf_s(parserStageThreshold->value(), "%f", &tmpStageThreshold);
            curStage.setStageThreshold(tmpStageThreshold);

            //parse trees
            rapidxml::xml_node<> *parserTree;
            parserTree = parserGlobal->first_node("trees");
            ncvAssertReturn(parserTree, NCV_HAAR_XML_LOADING_EXCEPTION);
            parserTree = parserTree->first_node("_");
            ncvAssertReturn(parserTree, NCV_HAAR_XML_LOADING_EXCEPTION);

            while (parserTree)
            {
                rapidxml::xml_node<> *parserNode;
                parserNode = parserTree->first_node("_");
                ncvAssertReturn(parserNode, NCV_HAAR_XML_LOADING_EXCEPTION);
                Ncv32u nodeId = 0;

                while (parserNode)
                {
                    HaarClassifierNode128 curNode;

                    rapidxml::xml_node<> *parserNodeThreshold = parserNode->first_node("threshold");
                    ncvAssertReturn(parserNodeThreshold, NCV_HAAR_XML_LOADING_EXCEPTION);
                    Ncv32f tmpThreshold;
                    sscanf_s(parserNodeThreshold->value(), "%f", &tmpThreshold);
                    curNode.setThreshold(tmpThreshold);

                    rapidxml::xml_node<> *parserNodeLeft = parserNode->first_node("left_val");
                    HaarClassifierNodeDescriptor32 nodeLeft;
                    if (parserNodeLeft)
                    {
                        Ncv32f leftVal;
                        sscanf_s(parserNodeLeft->value(), "%f", &leftVal);
                        ncvStat = nodeLeft.create(leftVal);
                        ncvAssertReturn(ncvStat == NCV_SUCCESS, ncvStat);
                    }
                    else
                    {
                        parserNodeLeft = parserNode->first_node("left_node");
                        ncvAssertReturn(parserNodeLeft, NCV_HAAR_XML_LOADING_EXCEPTION);
                        Ncv32u leftNodeOffset;
                        sscanf_s(parserNodeLeft->value(), "%d", &leftNodeOffset);
                        nodeLeft.create(h_TmpClassifierNotRootNodes.size() + leftNodeOffset - 1);
                        haar.bHasStumpsOnly = false;
                    }
                    curNode.setLeftNodeDesc(nodeLeft);

                    rapidxml::xml_node<> *parserNodeRight = parserNode->first_node("right_val");
                    HaarClassifierNodeDescriptor32 nodeRight;
                    if (parserNodeRight)
                    {
                        Ncv32f rightVal;
                        sscanf_s(parserNodeRight->value(), "%f", &rightVal);
                        ncvStat = nodeRight.create(rightVal);
                        ncvAssertReturn(ncvStat == NCV_SUCCESS, ncvStat);
                    }
                    else
                    {
                        parserNodeRight = parserNode->first_node("right_node");
                        ncvAssertReturn(parserNodeRight, NCV_HAAR_XML_LOADING_EXCEPTION);
                        Ncv32u rightNodeOffset;
                        sscanf_s(parserNodeRight->value(), "%d", &rightNodeOffset);
                        nodeRight.create(h_TmpClassifierNotRootNodes.size() + rightNodeOffset - 1);
                        haar.bHasStumpsOnly = false;
                    }
                    curNode.setRightNodeDesc(nodeRight);

                    rapidxml::xml_node<> *parserNodeFeatures = parserNode->first_node("feature");
                    ncvAssertReturn(parserNodeFeatures, NCV_HAAR_XML_LOADING_EXCEPTION);

                    rapidxml::xml_node<> *parserNodeFeaturesTilted = parserNodeFeatures->first_node("tilted");
                    ncvAssertReturn(parserNodeFeaturesTilted, NCV_HAAR_XML_LOADING_EXCEPTION);
                    Ncv32u tiltedVal;
                    sscanf_s(parserNodeFeaturesTilted->value(), "%d", &tiltedVal);
                    haar.bNeedsTiltedII = (tiltedVal != 0);

                    rapidxml::xml_node<> *parserNodeFeaturesRects = parserNodeFeatures->first_node("rects");
                    ncvAssertReturn(parserNodeFeaturesRects, NCV_HAAR_XML_LOADING_EXCEPTION);
                    parserNodeFeaturesRects = parserNodeFeaturesRects->first_node("_");
                    ncvAssertReturn(parserNodeFeaturesRects, NCV_HAAR_XML_LOADING_EXCEPTION);
                    Ncv32u featureId = 0;

                    while (parserNodeFeaturesRects)
                    {
                        Ncv32u rectX, rectY, rectWidth, rectHeight;
                        Ncv32f rectWeight;
                        sscanf_s(parserNodeFeaturesRects->value(), "%d %d %d %d %f", &rectX, &rectY, &rectWidth, &rectHeight, &rectWeight);
                        HaarFeature64 curFeature;
                        ncvStat = curFeature.setRect(rectX, rectY, rectWidth, rectHeight, haar.ClassifierSize.width, haar.ClassifierSize.height);
                        curFeature.setWeight(rectWeight);
                        ncvAssertReturn(NCV_SUCCESS == ncvStat, ncvStat);
                        haarFeatures.push_back(curFeature);

                        parserNodeFeaturesRects = parserNodeFeaturesRects->next_sibling("_");
                        featureId++;
                    }

                    HaarFeatureDescriptor32 tmpFeatureDesc;
                    ncvStat = tmpFeatureDesc.create(haar.bNeedsTiltedII, featureId, haarFeatures.size() - featureId);
                    ncvAssertReturn(NCV_SUCCESS == ncvStat, ncvStat);
                    curNode.setFeatureDesc(tmpFeatureDesc);

                    if (!nodeId)
                    {
                        //root node
                        haarClassifierNodes.push_back(curNode);
                        curMaxTreeDepth = 1;
                    }
                    else
                    {
                        //other node
                        h_TmpClassifierNotRootNodes.push_back(curNode);
                        curMaxTreeDepth++;
                    }

                    parserNode = parserNode->next_sibling("_");
                    nodeId++;
                }

                parserTree = parserTree->next_sibling("_");
                tmpNumClassifierRootNodes++;
            }

            curStage.setNumClassifierRootNodes(tmpNumClassifierRootNodes);
            haarStages.push_back(curStage);

            parserGlobal = parserGlobal->next_sibling("_");
        }
    }
    catch (...)
    {
        return NCV_HAAR_XML_LOADING_EXCEPTION;
    }

    //fill in cascade stats
    haar.NumStages = haarStages.size();
    haar.NumClassifierRootNodes = haarClassifierNodes.size();
    haar.NumClassifierTotalNodes = haar.NumClassifierRootNodes + h_TmpClassifierNotRootNodes.size();
    haar.NumFeatures = haarFeatures.size();

    //merge root and leaf nodes in one classifiers array
    Ncv32u offsetRoot = haarClassifierNodes.size();
    for (Ncv32u i=0; i<haarClassifierNodes.size(); i++)
    {
        HaarClassifierNodeDescriptor32 nodeLeft = haarClassifierNodes[i].getLeftNodeDesc();
        if (!nodeLeft.isLeaf())
        {
            Ncv32u newOffset = nodeLeft.getNextNodeOffset() + offsetRoot;
            nodeLeft.create(newOffset);
        }
        haarClassifierNodes[i].setLeftNodeDesc(nodeLeft);

        HaarClassifierNodeDescriptor32 nodeRight = haarClassifierNodes[i].getRightNodeDesc();
        if (!nodeRight.isLeaf())
        {
            Ncv32u newOffset = nodeRight.getNextNodeOffset() + offsetRoot;
            nodeRight.create(newOffset);
        }
        haarClassifierNodes[i].setRightNodeDesc(nodeRight);
    }
    for (Ncv32u i=0; i<h_TmpClassifierNotRootNodes.size(); i++)
    {
        HaarClassifierNodeDescriptor32 nodeLeft = h_TmpClassifierNotRootNodes[i].getLeftNodeDesc();
        if (!nodeLeft.isLeaf())
        {
            Ncv32u newOffset = nodeLeft.getNextNodeOffset() + offsetRoot;
            nodeLeft.create(newOffset);
        }
        h_TmpClassifierNotRootNodes[i].setLeftNodeDesc(nodeLeft);

        HaarClassifierNodeDescriptor32 nodeRight = h_TmpClassifierNotRootNodes[i].getRightNodeDesc();
        if (!nodeRight.isLeaf())
        {
            Ncv32u newOffset = nodeRight.getNextNodeOffset() + offsetRoot;
            nodeRight.create(newOffset);
        }
        h_TmpClassifierNotRootNodes[i].setRightNodeDesc(nodeRight);

        haarClassifierNodes.push_back(h_TmpClassifierNotRootNodes[i]);
    }

    return NCV_SUCCESS;
}

#endif /* loadFromXML implementation switch */

#endif /* HAVE_CUDA */


