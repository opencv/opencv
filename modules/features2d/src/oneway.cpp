/*
 *  cvoneway.cpp
 *  one_way_sample
 *
 *  Created by Victor  Eruhimov on 3/23/10.
 *  Copyright 2010 Argus Corp. All rights reserved.
 *
 */

#include "precomp.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdio.h>

namespace cv{
    
    inline int round(float value)
    {
        if(value > 0)
        {
            return int(value + 0.5f);
        }
        else
        {
            return int(value - 0.5f);
        }
    }
    
    inline CvRect resize_rect(CvRect rect, float alpha)
    {
        return cvRect(rect.x + round((float)(0.5*(1 - alpha)*rect.width)), rect.y + round((float)(0.5*(1 - alpha)*rect.height)),
                      round(rect.width*alpha), round(rect.height*alpha));
    }
    
    CvMat* ConvertImageToMatrix(IplImage* patch);
    
    class CvCameraPose
        {
        public:
            CvCameraPose()
            {
                m_rotation = cvCreateMat(1, 3, CV_32FC1);
                m_translation = cvCreateMat(1, 3, CV_32FC1);
            };
            
            ~CvCameraPose()
            {
                cvReleaseMat(&m_rotation);
                cvReleaseMat(&m_translation);
            };
            
            void SetPose(CvMat* rotation, CvMat* translation)
            {
                cvCopy(rotation, m_rotation);
                cvCopy(translation, m_translation);
            };
            
            CvMat* GetRotation() {return m_rotation;};
            CvMat* GetTranslation() {return m_translation;};
            
        protected:
            CvMat* m_rotation;
            CvMat* m_translation;
        };
    
    // AffineTransformPatch: generates an affine transformed image patch.
    // - src: source image (roi is supported)
    // - dst: output image. ROI of dst image should be 2 times smaller than ROI of src.
    // - pose: parameters of an affine transformation
    void AffineTransformPatch(IplImage* src, IplImage* dst, CvAffinePose pose);
    
    // GenerateAffineTransformFromPose: generates an affine transformation matrix from CvAffinePose instance
    // - size: the size of image patch
    // - pose: affine transformation
    // - transform: 2x3 transformation matrix
    void GenerateAffineTransformFromPose(CvSize size, CvAffinePose pose, CvMat* transform);
    
    // Generates a random affine pose
    CvAffinePose GenRandomAffinePose();
    
    
    const static int num_mean_components = 500;
    const static float noise_intensity = 0.15f;
    
    
    static inline CvPoint rect_center(CvRect rect)
    {
        return cvPoint(rect.x + rect.width/2, rect.y + rect.height/2);
    }
    
    void homography_transform(IplImage* frontal, IplImage* result, CvMat* homography)
    {
        cvWarpPerspective(frontal, result, homography);
    }
    
    CvAffinePose perturbate_pose(CvAffinePose pose, float noise)
    {
        // perturbate the matrix
        float noise_mult_factor = 1 + (0.5f - float(rand())/RAND_MAX)*noise;
        float noise_add_factor = noise_mult_factor - 1;
        
        CvAffinePose pose_pert = pose;
        pose_pert.phi += noise_add_factor;
        pose_pert.theta += noise_mult_factor;
        pose_pert.lambda1 *= noise_mult_factor;
        pose_pert.lambda2 *= noise_mult_factor;
        
        return pose_pert;
    }
    
    void generate_mean_patch(IplImage* frontal, IplImage* result, CvAffinePose pose, int pose_count, float noise)
    {
        IplImage* sum = cvCreateImage(cvSize(result->width, result->height), IPL_DEPTH_32F, 1);
        IplImage* workspace = cvCloneImage(result);
        IplImage* workspace_float = cvCloneImage(sum);
        
        cvSetZero(sum);
        for(int i = 0; i < pose_count; i++)
        {
            CvAffinePose pose_pert = perturbate_pose(pose, noise);
            
            AffineTransformPatch(frontal, workspace, pose_pert);
            cvConvertScale(workspace, workspace_float);
            cvAdd(sum, workspace_float, sum);
        }
        
        cvConvertScale(sum, result, 1.0f/pose_count);
        
        cvReleaseImage(&workspace);
        cvReleaseImage(&sum);
        cvReleaseImage(&workspace_float);
    }
    
    void generate_mean_patch_fast(IplImage* /*frontal*/, IplImage* /*result*/, CvAffinePose /*pose*/,
                                  CvMat* /*pca_hr_avg*/, CvMat* /*pca_hr_eigenvectors*/, const OneWayDescriptor* /*pca_descriptors*/)
    {
        /*for(int i = 0; i < pca_hr_eigenvectors->cols; i++)
        {
            
        }*/
    }
    
    void readPCAFeatures(const char *filename, CvMat** avg, CvMat** eigenvectors, const char *postfix = "");
    void readPCAFeatures(const FileNode &fn, CvMat** avg, CvMat** eigenvectors, const char* postfix = "");
    void savePCAFeatures(FileStorage &fs, const char* postfix, CvMat* avg, CvMat* eigenvectors);
    void calcPCAFeatures(vector<IplImage*>& patches, FileStorage &fs, const char* postfix, CvMat** avg,
                         CvMat** eigenvectors);
    void loadPCAFeatures(const char* path, const char* images_list, vector<IplImage*>& patches, CvSize patch_size);
    void generatePCAFeatures(const char* path, const char* img_filename, FileStorage& fs, const char* postfix,
                             CvSize patch_size, CvMat** avg, CvMat** eigenvectors);
    
    void eigenvector2image(CvMat* eigenvector, IplImage* img);

    void FindOneWayDescriptor(int desc_count, const OneWayDescriptor* descriptors, IplImage* patch, int& desc_idx, int& pose_idx, float& distance,
                              CvMat* avg = 0, CvMat* eigenvalues = 0);
    
    void FindOneWayDescriptor(int desc_count, const OneWayDescriptor* descriptors, IplImage* patch, int n,
                              std::vector<int>& desc_idxs, std::vector<int>&  pose_idxs, std::vector<float>& distances,
                              CvMat* avg = 0, CvMat* eigenvalues = 0);
    
    void FindOneWayDescriptor(cv::flann::Index* m_pca_descriptors_tree, CvSize patch_size, int m_pca_dim_low, int m_pose_count, IplImage* patch, int& desc_idx, int& pose_idx, float& distance,
                              CvMat* avg = 0, CvMat* eigenvalues = 0);
    
    void FindOneWayDescriptorEx(int desc_count, const OneWayDescriptor* descriptors, IplImage* patch,
                                float scale_min, float scale_max, float scale_step,
                                int& desc_idx, int& pose_idx, float& distance, float& scale,
                                CvMat* avg, CvMat* eigenvectors);
    
    void FindOneWayDescriptorEx(int desc_count, const OneWayDescriptor* descriptors, IplImage* patch,
                                float scale_min, float scale_max, float scale_step,
                                int n, std::vector<int>& desc_idxs, std::vector<int>& pose_idxs,
                                std::vector<float>& distances, std::vector<float>& scales,
                                CvMat* avg, CvMat* eigenvectors);
    
    void FindOneWayDescriptorEx(cv::flann::Index* m_pca_descriptors_tree, CvSize patch_size, int m_pca_dim_low, int m_pose_count, IplImage* patch,
                                float scale_min, float scale_max, float scale_step,
                                int& desc_idx, int& pose_idx, float& distance, float& scale,
                                CvMat* avg, CvMat* eigenvectors);
    
    inline CvRect fit_rect_roi_fixedsize(CvRect rect, CvRect roi)
    {
        CvRect fit = rect;
        fit.x = MAX(fit.x, roi.x);
        fit.y = MAX(fit.y, roi.y);
        fit.x = MIN(fit.x, roi.x + roi.width - fit.width - 1);
        fit.y = MIN(fit.y, roi.y + roi.height - fit.height - 1);
        return(fit);
    }
    
    inline CvRect fit_rect_fixedsize(CvRect rect, IplImage* img)
    {
        CvRect roi = cvGetImageROI(img);
        return fit_rect_roi_fixedsize(rect, roi);
    }
    
    OneWayDescriptor::OneWayDescriptor()
    {
        m_pose_count = 0;
        m_samples = 0;
        m_input_patch = 0;
        m_train_patch = 0;
        m_pca_coeffs = 0;
        m_affine_poses = 0;
        m_transforms = 0;
        m_pca_dim_low = 100;
        m_pca_dim_high = 100;
    }
    
    OneWayDescriptor::~OneWayDescriptor()
    {
        if(m_pose_count)
        {
            for(int i = 0; i < m_pose_count; i++)
            {
                cvReleaseImage(&m_samples[i]);
                cvReleaseMat(&m_pca_coeffs[i]);
            }
            cvReleaseImage(&m_input_patch);
            cvReleaseImage(&m_train_patch);
            delete []m_samples;
            delete []m_pca_coeffs;
            
            if(!m_transforms)
            {
                delete []m_affine_poses;
            }
        }
    }
    
    void OneWayDescriptor::Allocate(int pose_count, CvSize size, int nChannels)
    {
        m_pose_count = pose_count;
        m_samples = new IplImage* [m_pose_count];
        m_pca_coeffs = new CvMat* [m_pose_count];
        m_patch_size = cvSize(size.width/2, size.height/2);
        
        if(!m_transforms)
        {
            m_affine_poses = new CvAffinePose[m_pose_count];
        }
        
        int length = m_pca_dim_low;//roi.width*roi.height;
        for(int i = 0; i < m_pose_count; i++)
        {
            m_samples[i] = cvCreateImage(cvSize(size.width/2, size.height/2), IPL_DEPTH_32F, nChannels);
            m_pca_coeffs[i] = cvCreateMat(1, length, CV_32FC1);
        }
        
        m_input_patch = cvCreateImage(GetPatchSize(), IPL_DEPTH_8U, 1);
        m_train_patch = cvCreateImage(GetInputPatchSize(), IPL_DEPTH_8U, 1);
    }
    
    void cvmSet2DPoint(CvMat* matrix, int row, int col, CvPoint2D32f point)
    {
        cvmSet(matrix, row, col, point.x);
        cvmSet(matrix, row, col + 1, point.y);
    }
    
    void cvmSet3DPoint(CvMat* matrix, int row, int col, CvPoint3D32f point)
    {
        cvmSet(matrix, row, col, point.x);
        cvmSet(matrix, row, col + 1, point.y);
        cvmSet(matrix, row, col + 2, point.z);
    }
    
    CvAffinePose GenRandomAffinePose()
    {
        const float scale_min = 0.8f;
        const float scale_max = 1.2f;
        CvAffinePose pose;
        pose.theta = float(rand())/RAND_MAX*120 - 60;
        pose.phi = float(rand())/RAND_MAX*360;
        pose.lambda1 = scale_min + float(rand())/RAND_MAX*(scale_max - scale_min);
        pose.lambda2 = scale_min + float(rand())/RAND_MAX*(scale_max - scale_min);
        
        return pose;
    }
    
    void GenerateAffineTransformFromPose(CvSize size, CvAffinePose pose, CvMat* transform)
    {
        CvMat* temp = cvCreateMat(3, 3, CV_32FC1);
        CvMat* final = cvCreateMat(3, 3, CV_32FC1);
        cvmSet(temp, 2, 0, 0.0f);
        cvmSet(temp, 2, 1, 0.0f);
        cvmSet(temp, 2, 2, 1.0f);
        
        CvMat rotation;
        cvGetSubRect(temp, &rotation, cvRect(0, 0, 3, 2));
        
        cv2DRotationMatrix(cvPoint2D32f(size.width/2, size.height/2), pose.phi, 1.0, &rotation);
        cvCopy(temp, final);
        
        cvmSet(temp, 0, 0, pose.lambda1);
        cvmSet(temp, 0, 1, 0.0f);
        cvmSet(temp, 1, 0, 0.0f);
        cvmSet(temp, 1, 1, pose.lambda2);
        cvmSet(temp, 0, 2, size.width/2*(1 - pose.lambda1));
        cvmSet(temp, 1, 2, size.height/2*(1 - pose.lambda2));
        cvMatMul(temp, final, final);
        
        cv2DRotationMatrix(cvPoint2D32f(size.width/2, size.height/2), pose.theta - pose.phi, 1.0, &rotation);
        cvMatMul(temp, final, final);
        
        cvGetSubRect(final, &rotation, cvRect(0, 0, 3, 2));
        cvCopy(&rotation, transform);
        
        cvReleaseMat(&temp);
        cvReleaseMat(&final);
    }
    
    void AffineTransformPatch(IplImage* src, IplImage* dst, CvAffinePose pose)
    {
        CvRect src_large_roi = cvGetImageROI(src);
        
        IplImage* temp = cvCreateImage(cvSize(src_large_roi.width, src_large_roi.height), IPL_DEPTH_32F, src->nChannels);
        cvSetZero(temp);
        IplImage* temp2 = cvCloneImage(temp);
        CvMat* rotation_phi = cvCreateMat(2, 3, CV_32FC1);
        
        CvSize new_size = cvSize(cvRound(temp->width*pose.lambda1), cvRound(temp->height*pose.lambda2));
        IplImage* temp3 = cvCreateImage(new_size, IPL_DEPTH_32F, src->nChannels);
        
        cvConvertScale(src, temp);
        cvResetImageROI(temp);
        
        
        cv2DRotationMatrix(cvPoint2D32f(temp->width/2, temp->height/2), pose.phi, 1.0, rotation_phi);
        cvWarpAffine(temp, temp2, rotation_phi);
        
        cvSetZero(temp);
        
        cvResize(temp2, temp3);
        
        cv2DRotationMatrix(cvPoint2D32f(temp3->width/2, temp3->height/2), pose.theta - pose.phi, 1.0, rotation_phi);
        cvWarpAffine(temp3, temp, rotation_phi);
        
        cvSetImageROI(temp, cvRect(temp->width/2 - src_large_roi.width/4, temp->height/2 - src_large_roi.height/4,
                                   src_large_roi.width/2, src_large_roi.height/2));
        cvConvertScale(temp, dst);
        cvReleaseMat(&rotation_phi);
        
        cvReleaseImage(&temp3);
        cvReleaseImage(&temp2);
        cvReleaseImage(&temp);
    }
    
    void OneWayDescriptor::GenerateSamples(int pose_count, IplImage* frontal, int norm)
    {
        /*    if(m_transforms)
         {
         GenerateSamplesWithTransforms(pose_count, frontal);
         return;
         }
         */
        CvRect roi = cvGetImageROI(frontal);
        IplImage* patch_8u = cvCreateImage(cvSize(roi.width/2, roi.height/2), frontal->depth, frontal->nChannels);
        for(int i = 0; i < pose_count; i++)
        {
            if(!m_transforms)
            {
                m_affine_poses[i] = GenRandomAffinePose();
            }
            //AffineTransformPatch(frontal, patch_8u, m_affine_poses[i]);
            generate_mean_patch(frontal, patch_8u, m_affine_poses[i], num_mean_components, noise_intensity);
            
            double scale = 1.0f;
            if(norm)
            {
                double sum = cvSum(patch_8u).val[0];
                scale = 1/sum;
            }
            cvConvertScale(patch_8u, m_samples[i], scale);
            
#if 0
            double maxval;
            cvMinMaxLoc(m_samples[i], 0, &maxval);
            IplImage* test = cvCreateImage(cvSize(roi.width/2, roi.height/2), IPL_DEPTH_8U, 1);
            cvConvertScale(m_samples[i], test, 255.0/maxval);
            cvNamedWindow("1", 1);
            cvShowImage("1", test);
            cvWaitKey(0);
#endif
        }
        cvReleaseImage(&patch_8u);
    }
    
    void OneWayDescriptor::GenerateSamplesFast(IplImage* frontal, CvMat* pca_hr_avg,
                                               CvMat* pca_hr_eigenvectors, OneWayDescriptor* pca_descriptors)
    {
        CvRect roi = cvGetImageROI(frontal);
        if(roi.width != GetInputPatchSize().width || roi.height != GetInputPatchSize().height)
        {
            cvResize(frontal, m_train_patch);
            frontal = m_train_patch;
        }
        
        CvMat* pca_coeffs = cvCreateMat(1, pca_hr_eigenvectors->cols, CV_32FC1);
        double maxval;
        cvMinMaxLoc(frontal, 0, &maxval);
        CvMat* frontal_data = ConvertImageToMatrix(frontal);
        
        double sum = cvSum(frontal_data).val[0];
        cvConvertScale(frontal_data, frontal_data, 1.0f/sum);
        cvProjectPCA(frontal_data, pca_hr_avg, pca_hr_eigenvectors, pca_coeffs);
        for(int i = 0; i < m_pose_count; i++)
        {
            cvSetZero(m_samples[i]);
            for(int j = 0; j < m_pca_dim_high; j++)
            {
                double coeff = cvmGet(pca_coeffs, 0, j);
                IplImage* patch = pca_descriptors[j + 1].GetPatch(i);
                cvAddWeighted(m_samples[i], 1.0, patch, coeff, 0, m_samples[i]);
                
#if 0
                printf("coeff%d = %f\n", j, coeff);
                IplImage* test = cvCreateImage(cvSize(12, 12), IPL_DEPTH_8U, 1);
                double maxval;
                cvMinMaxLoc(patch, 0, &maxval);
                cvConvertScale(patch, test, 255.0/maxval);
                cvNamedWindow("1", 1);
                cvShowImage("1", test);
                cvWaitKey(0);
#endif
            }
            
            cvAdd(pca_descriptors[0].GetPatch(i), m_samples[i], m_samples[i]);
            double sum = cvSum(m_samples[i]).val[0];
            cvConvertScale(m_samples[i], m_samples[i], 1.0/sum);
            
#if 0
            IplImage* test = cvCreateImage(cvSize(12, 12), IPL_DEPTH_8U, 1);
            /*        IplImage* temp1 = cvCreateImage(cvSize(12, 12), IPL_DEPTH_32F, 1);
             eigenvector2image(pca_hr_avg, temp1);
             IplImage* test = cvCreateImage(cvSize(12, 12), IPL_DEPTH_8U, 1);
             cvAdd(m_samples[i], temp1, temp1);
             cvMinMaxLoc(temp1, 0, &maxval);
             cvConvertScale(temp1, test, 255.0/maxval);*/
            cvMinMaxLoc(m_samples[i], 0, &maxval);
            cvConvertScale(m_samples[i], test, 255.0/maxval);
            
            cvNamedWindow("1", 1);
            cvShowImage("1", frontal);
            cvNamedWindow("2", 1);
            cvShowImage("2", test);
            cvWaitKey(0);
#endif
        }
        
        cvReleaseMat(&pca_coeffs);
        cvReleaseMat(&frontal_data);
    }
    
    void OneWayDescriptor::SetTransforms(CvAffinePose* poses, CvMat** transforms)
    {
        if(m_affine_poses)
        {
            delete []m_affine_poses;
        }
        
        m_affine_poses = poses;
        m_transforms = transforms;
    }
    
    void OneWayDescriptor::Initialize(int pose_count, IplImage* frontal, const char* feature_name, int norm)
    {
        m_feature_name = std::string(feature_name);
        CvRect roi = cvGetImageROI(frontal);
        m_center = rect_center(roi);
        
        Allocate(pose_count, cvSize(roi.width, roi.height), frontal->nChannels);
        
        GenerateSamples(pose_count, frontal, norm);
    }
    
    void OneWayDescriptor::InitializeFast(int pose_count, IplImage* frontal, const char* feature_name,
                                          CvMat* pca_hr_avg, CvMat* pca_hr_eigenvectors, OneWayDescriptor* pca_descriptors)
    {
        if(pca_hr_avg == 0)
        {
            Initialize(pose_count, frontal, feature_name, 1);
            return;
        }
        m_feature_name = std::string(feature_name);
        CvRect roi = cvGetImageROI(frontal);
        m_center = rect_center(roi);
        
        Allocate(pose_count, cvSize(roi.width, roi.height), frontal->nChannels);
        
        GenerateSamplesFast(frontal, pca_hr_avg, pca_hr_eigenvectors, pca_descriptors);
    }
    
    void OneWayDescriptor::InitializePCACoeffs(CvMat* avg, CvMat* eigenvectors)
    {
        for(int i = 0; i < m_pose_count; i++)
        {
            ProjectPCASample(m_samples[i], avg, eigenvectors, m_pca_coeffs[i]);
        }
    }
    
    void OneWayDescriptor::ProjectPCASample(IplImage* patch, CvMat* avg, CvMat* eigenvectors, CvMat* pca_coeffs) const
    {
        CvMat* patch_mat = ConvertImageToMatrix(patch);
        //    CvMat eigenvectorsr;
        //    cvGetSubRect(eigenvectors, &eigenvectorsr, cvRect(0, 0, eigenvectors->cols, pca_coeffs->cols));
        CvMat* temp = cvCreateMat(1, eigenvectors->cols, CV_32FC1);
        cvProjectPCA(patch_mat, avg, eigenvectors, temp);
        CvMat temp1;
        cvGetSubRect(temp, &temp1, cvRect(0, 0, pca_coeffs->cols, 1));
        cvCopy(&temp1, pca_coeffs);
        
        cvReleaseMat(&temp);
        cvReleaseMat(&patch_mat);
    }
    
    void OneWayDescriptor::EstimatePosePCA(CvArr* patch, int& pose_idx, float& distance, CvMat* avg, CvMat* eigenvectors) const
    {
        if(avg == 0)
        {
            // do not use pca
            if (!CV_IS_MAT(patch))
            {
                EstimatePose((IplImage*)patch, pose_idx, distance);
            }
            else
            {
                
            }
            return;
        }
        CvRect roi={0,0,0,0};
        if (!CV_IS_MAT(patch))
        {
            roi = cvGetImageROI((IplImage*)patch);
            if(roi.width != GetPatchSize().width || roi.height != GetPatchSize().height)
            {
                cvResize(patch, m_input_patch);
                patch = m_input_patch;
                roi = cvGetImageROI((IplImage*)patch);
            }
        }
        
        CvMat* pca_coeffs = cvCreateMat(1, m_pca_dim_low, CV_32FC1);
        
        if (CV_IS_MAT(patch))
        {
            cvCopy((CvMat*)patch, pca_coeffs);
        }
        else
        {
            IplImage* patch_32f = cvCreateImage(cvSize(roi.width, roi.height), IPL_DEPTH_32F, 1);
            double sum = cvSum(patch).val[0];
            cvConvertScale(patch, patch_32f, 1.0f/sum);
            ProjectPCASample(patch_32f, avg, eigenvectors, pca_coeffs);
            cvReleaseImage(&patch_32f);
        }
        
        
        distance = 1e10;
        pose_idx = -1;
        
        for(int i = 0; i < m_pose_count; i++)
        {
            double dist = cvNorm(m_pca_coeffs[i], pca_coeffs);
            //		float dist = 0;
            //		float data1, data2;
            //		//CvMat* pose_pca_coeffs = m_pca_coeffs[i];
            //		for (int x=0; x < pca_coeffs->width; x++)
            //			for (int y =0 ; y < pca_coeffs->height; y++)
            //			{
            //				data1 = ((float*)(pca_coeffs->data.ptr + pca_coeffs->step*x))[y];
            //				data2 = ((float*)(m_pca_coeffs[i]->data.ptr + m_pca_coeffs[i]->step*x))[y];
            //				dist+=(data1-data2)*(data1-data2);
            //			}
            ////#if 1
            //		for (int j = 0; j < m_pca_dim_low; j++)
            //		{
            //			dist += (pose_pca_coeffs->data.fl[j]- pca_coeffs->data.fl[j])*(pose_pca_coeffs->data.fl[j]- pca_coeffs->data.fl[j]);
            //		}
            //#else
            //		for (int j = 0; j <= m_pca_dim_low - 4; j += 4)
            //		{
            //			dist += (pose_pca_coeffs->data.fl[j]- pca_coeffs->data.fl[j])*
            //				(pose_pca_coeffs->data.fl[j]- pca_coeffs->data.fl[j]);
            //			dist += (pose_pca_coeffs->data.fl[j+1]- pca_coeffs->data.fl[j+1])*
            //				(pose_pca_coeffs->data.fl[j+1]- pca_coeffs->data.fl[j+1]);
            //			dist += (pose_pca_coeffs->data.fl[j+2]- pca_coeffs->data.fl[j+2])*
            //				(pose_pca_coeffs->data.fl[j+2]- pca_coeffs->data.fl[j+2]);
            //			dist += (pose_pca_coeffs->data.fl[j+3]- pca_coeffs->data.fl[j+3])*
            //				(pose_pca_coeffs->data.fl[j+3]- pca_coeffs->data.fl[j+3]);
            //		}
            //#endif
            if(dist < distance)
            {
                distance = (float)dist;
                pose_idx = i;
            }
        }
        
        cvReleaseMat(&pca_coeffs);
    }
    
    void OneWayDescriptor::EstimatePose(IplImage* patch, int& pose_idx, float& distance) const
    {
        distance = 1e10;
        pose_idx = -1;
        
        CvRect roi = cvGetImageROI(patch);
        IplImage* patch_32f = cvCreateImage(cvSize(roi.width, roi.height), IPL_DEPTH_32F, patch->nChannels);
        double sum = cvSum(patch).val[0];
        cvConvertScale(patch, patch_32f, 1/sum);
        
        for(int i = 0; i < m_pose_count; i++)
        {
            if(m_samples[i]->width != patch_32f->width || m_samples[i]->height != patch_32f->height)
            {
                continue;
            }
            double dist = cvNorm(m_samples[i], patch_32f);
            //float dist = 0.0f;
            //float i1,i2;
            
            //for (int y = 0; y<patch_32f->height; y++)
            //	for (int x = 0; x< patch_32f->width; x++)
            //	{
            //		i1 = ((float*)(m_samples[i]->imageData + m_samples[i]->widthStep*y))[x];
            //		i2 = ((float*)(patch_32f->imageData + patch_32f->widthStep*y))[x];
            //		dist+= (i1-i2)*(i1-i2);
            //	}
            
            if(dist < distance)
            {
                distance = (float)dist;
                pose_idx = i;
            }
            
#if 0
            IplImage* img1 = cvCreateImage(cvSize(roi.width, roi.height), IPL_DEPTH_8U, 1);
            IplImage* img2 = cvCreateImage(cvSize(roi.width, roi.height), IPL_DEPTH_8U, 1);
            double maxval;
            cvMinMaxLoc(m_samples[i], 0, &maxval);
            cvConvertScale(m_samples[i], img1, 255.0/maxval);
            cvMinMaxLoc(patch_32f, 0, &maxval);
            cvConvertScale(patch_32f, img2, 255.0/maxval);
            
            cvNamedWindow("1", 1);
            cvShowImage("1", img1);
            cvNamedWindow("2", 1);
            cvShowImage("2", img2);
            printf("Distance = %f\n", dist);
            cvWaitKey(0);
#endif
        }
        
        cvReleaseImage(&patch_32f);
    }
    
    void OneWayDescriptor::Save(const char* path)
    {
        for(int i = 0; i < m_pose_count; i++)
        {
            char buf[1024];
            sprintf(buf, "%s/patch_%04d.jpg", path, i);
            IplImage* patch = cvCreateImage(cvSize(m_samples[i]->width, m_samples[i]->height), IPL_DEPTH_8U, m_samples[i]->nChannels);
            
            double maxval;
            cvMinMaxLoc(m_samples[i], 0, &maxval);
            cvConvertScale(m_samples[i], patch, 255/maxval);
            
            cvSaveImage(buf, patch);
            
            cvReleaseImage(&patch);
        }
    }
    
    void OneWayDescriptor::Write(CvFileStorage* fs, const char* name)
    {
        CvMat* mat = cvCreateMat(m_pose_count, m_samples[0]->width*m_samples[0]->height, CV_32FC1);
        
        // prepare data to write as a single matrix
        for(int i = 0; i < m_pose_count; i++)
        {
            for(int y = 0; y < m_samples[i]->height; y++)
            {
                for(int x = 0; x < m_samples[i]->width; x++)
                {
                    float val = *((float*)(m_samples[i]->imageData + m_samples[i]->widthStep*y) + x);
                    cvmSet(mat, i, y*m_samples[i]->width + x, val);
                }
            }
        }
        
        cvWrite(fs, name, mat);
        
        cvReleaseMat(&mat);
    }
    
    int OneWayDescriptor::ReadByName(const FileNode &parent, const char* name)
    {
        CvMat* mat = reinterpret_cast<CvMat*> (parent[name].readObj ());
        if(!mat)
        {
            return 0;
        }
        
        
        for(int i = 0; i < m_pose_count; i++)
        {
            for(int y = 0; y < m_samples[i]->height; y++)
            {
                for(int x = 0; x < m_samples[i]->width; x++)
                {
                    float val = (float)cvmGet(mat, i, y*m_samples[i]->width + x);
                    *((float*)(m_samples[i]->imageData + y*m_samples[i]->widthStep) + x) = val;
                }
            }
        }
        
        cvReleaseMat(&mat);
        return 1;
    }

    int OneWayDescriptor::ReadByName(CvFileStorage* fs, CvFileNode* parent, const char* name)
    {
        return ReadByName (FileNode (fs, parent), name);
    }
    
    IplImage* OneWayDescriptor::GetPatch(int index)
    {
        return m_samples[index];
    }
    
    CvAffinePose OneWayDescriptor::GetPose(int index) const
    {
        return m_affine_poses[index];
    }
    
    void FindOneWayDescriptor(int desc_count, const OneWayDescriptor* descriptors, IplImage* patch, int& desc_idx, int& pose_idx, float& distance,
                              CvMat* avg, CvMat* eigenvectors)
    {
        desc_idx = -1;
        pose_idx = -1;
        distance = 1e10;
        //--------
        //PCA_coeffs precalculating
        int m_pca_dim_low = descriptors[0].GetPCADimLow();
        CvMat* pca_coeffs = cvCreateMat(1, m_pca_dim_low, CV_32FC1);
        int patch_width = descriptors[0].GetPatchSize().width;
        int patch_height = descriptors[0].GetPatchSize().height;
        if (avg)
        {
            CvRect _roi = cvGetImageROI((IplImage*)patch);
            IplImage* test_img = cvCreateImage(cvSize(patch_width,patch_height), IPL_DEPTH_8U, 1);
            if(_roi.width != patch_width|| _roi.height != patch_height)
            {
                
                cvResize(patch, test_img);
                _roi = cvGetImageROI(test_img);
            }
            else
            {
                cvCopy(patch,test_img);
            }
            IplImage* patch_32f = cvCreateImage(cvSize(_roi.width, _roi.height), IPL_DEPTH_32F, 1);
            double sum = cvSum(test_img).val[0];
            cvConvertScale(test_img, patch_32f, 1.0f/sum);
            
            //ProjectPCASample(patch_32f, avg, eigenvectors, pca_coeffs);
            //Projecting PCA
            CvMat* patch_mat = ConvertImageToMatrix(patch_32f);
            CvMat* temp = cvCreateMat(1, eigenvectors->cols, CV_32FC1);
            cvProjectPCA(patch_mat, avg, eigenvectors, temp);
            CvMat temp1;
            cvGetSubRect(temp, &temp1, cvRect(0, 0, pca_coeffs->cols, 1));
            cvCopy(&temp1, pca_coeffs);
            cvReleaseMat(&temp);
            cvReleaseMat(&patch_mat);
            //End of projecting
            
            cvReleaseImage(&patch_32f);
            cvReleaseImage(&test_img);
        }
        
        //--------
        
        
        
        for(int i = 0; i < desc_count; i++)
        {
            int _pose_idx = -1;
            float _distance = 0;
            
#if 0
            descriptors[i].EstimatePose(patch, _pose_idx, _distance);
#else
            if (!avg)
            {
                descriptors[i].EstimatePosePCA(patch, _pose_idx, _distance, avg, eigenvectors);
            }
            else
            {
                descriptors[i].EstimatePosePCA(pca_coeffs, _pose_idx, _distance, avg, eigenvectors);
            }
#endif
            
            if(_distance < distance)
            {
                desc_idx = i;
                pose_idx = _pose_idx;
                distance = _distance;
            }
        }
        cvReleaseMat(&pca_coeffs);
    }
    
#if defined(_KDTREE)
    
    void FindOneWayDescriptor(cv::flann::Index* m_pca_descriptors_tree, CvSize patch_size, int m_pca_dim_low, int m_pose_count, IplImage* patch, int& desc_idx, int& pose_idx, float& distance,
                              CvMat* avg, CvMat* eigenvectors)
    {
        desc_idx = -1;
        pose_idx = -1;
        distance = 1e10;
        //--------
        //PCA_coeffs precalculating
        CvMat* pca_coeffs = cvCreateMat(1, m_pca_dim_low, CV_32FC1);
        int patch_width = patch_size.width;
        int patch_height = patch_size.height;
        //if (avg)
        //{
		CvRect _roi = cvGetImageROI((IplImage*)patch);
		IplImage* test_img = cvCreateImage(cvSize(patch_width,patch_height), IPL_DEPTH_8U, 1);
		if(_roi.width != patch_width|| _roi.height != patch_height)
		{
            
			cvResize(patch, test_img);
			_roi = cvGetImageROI(test_img);
		}
		else
		{
			cvCopy(patch,test_img);
		}
		IplImage* patch_32f = cvCreateImage(cvSize(_roi.width, _roi.height), IPL_DEPTH_32F, 1);
		float sum = cvSum(test_img).val[0];
		cvConvertScale(test_img, patch_32f, 1.0f/sum);
        
		//ProjectPCASample(patch_32f, avg, eigenvectors, pca_coeffs);
		//Projecting PCA
		CvMat* patch_mat = ConvertImageToMatrix(patch_32f);
		CvMat* temp = cvCreateMat(1, eigenvectors->cols, CV_32FC1);
		cvProjectPCA(patch_mat, avg, eigenvectors, temp);
		CvMat temp1;
		cvGetSubRect(temp, &temp1, cvRect(0, 0, pca_coeffs->cols, 1));
		cvCopy(&temp1, pca_coeffs);
		cvReleaseMat(&temp);
		cvReleaseMat(&patch_mat);
		//End of projecting
        
		cvReleaseImage(&patch_32f);
		cvReleaseImage(&test_img);
        //	}
        
        //--------
        
		//float* target = new float[m_pca_dim_low];
		//::cvflann::KNNResultSet res(1,pca_coeffs->data.fl,m_pca_dim_low);
		//::cvflann::SearchParams params;
		//params.checks = -1;
        
		//int maxDepth = 1000000;
		//int neighbors_count = 1;
		//int* neighborsIdx = new int[neighbors_count];
		//float* distances = new float[neighbors_count];
		//if (m_pca_descriptors_tree->findNearest(pca_coeffs->data.fl,neighbors_count,maxDepth,neighborsIdx,0,distances) > 0)
		//{
		//	desc_idx = neighborsIdx[0] / m_pose_count;
		//	pose_idx = neighborsIdx[0] % m_pose_count;
		//	distance = distances[0];
		//}
		//delete[] neighborsIdx;
		//delete[] distances;
        
		cv::Mat m_object(1, m_pca_dim_low, CV_32F);
		cv::Mat m_indices(1, 1, CV_32S);
		cv::Mat m_dists(1, 1, CV_32F);
        
		float* object_ptr = m_object.ptr<float>(0);
		for (int i=0;i<m_pca_dim_low;i++)
		{
			object_ptr[i] = pca_coeffs->data.fl[i];
		}
        
		m_pca_descriptors_tree->knnSearch(m_object, m_indices, m_dists, 1, cv::flann::SearchParams(-1) );
        
		desc_idx = ((int*)(m_indices.ptr<int>(0)))[0] / m_pose_count;
		pose_idx = ((int*)(m_indices.ptr<int>(0)))[0] % m_pose_count;
		distance = ((float*)(m_dists.ptr<float>(0)))[0];
        
        //	delete[] target;
        
        
        //    for(int i = 0; i < desc_count; i++)
        //    {
        //        int _pose_idx = -1;
        //        float _distance = 0;
        //
        //#if 0
        //        descriptors[i].EstimatePose(patch, _pose_idx, _distance);
        //#else
        //		if (!avg)
        //		{
        //			descriptors[i].EstimatePosePCA(patch, _pose_idx, _distance, avg, eigenvectors);
        //		}
        //		else
        //		{
        //			descriptors[i].EstimatePosePCA(pca_coeffs, _pose_idx, _distance, avg, eigenvectors);
        //		}
        //#endif
        //
        //        if(_distance < distance)
        //        {
        //            desc_idx = i;
        //            pose_idx = _pose_idx;
        //            distance = _distance;
        //        }
        //    }
        cvReleaseMat(&pca_coeffs);
    }
#endif
    //**
    void FindOneWayDescriptor(int desc_count, const OneWayDescriptor* descriptors, IplImage* patch, int n,
                              std::vector<int>& desc_idxs, std::vector<int>&  pose_idxs, std::vector<float>& distances,
                              CvMat* avg, CvMat* eigenvectors)
    {
        for (int i=0;i<n;i++)
        {
            desc_idxs[i] = -1;
            pose_idxs[i] = -1;
            distances[i] = 1e10;
        }
        //--------
        //PCA_coeffs precalculating
        int m_pca_dim_low = descriptors[0].GetPCADimLow();
        CvMat* pca_coeffs = cvCreateMat(1, m_pca_dim_low, CV_32FC1);
        int patch_width = descriptors[0].GetPatchSize().width;
        int patch_height = descriptors[0].GetPatchSize().height;
        if (avg)
        {
            CvRect _roi = cvGetImageROI((IplImage*)patch);
            IplImage* test_img = cvCreateImage(cvSize(patch_width,patch_height), IPL_DEPTH_8U, 1);
            if(_roi.width != patch_width|| _roi.height != patch_height)
            {
                
                cvResize(patch, test_img);
                _roi = cvGetImageROI(test_img);
            }
            else
            {
                cvCopy(patch,test_img);
            }
            IplImage* patch_32f = cvCreateImage(cvSize(_roi.width, _roi.height), IPL_DEPTH_32F, 1);
            double sum = cvSum(test_img).val[0];
            cvConvertScale(test_img, patch_32f, 1.0f/sum);
            
            //ProjectPCASample(patch_32f, avg, eigenvectors, pca_coeffs);
            //Projecting PCA
            CvMat* patch_mat = ConvertImageToMatrix(patch_32f);
            CvMat* temp = cvCreateMat(1, eigenvectors->cols, CV_32FC1);
            cvProjectPCA(patch_mat, avg, eigenvectors, temp);
            CvMat temp1;
            cvGetSubRect(temp, &temp1, cvRect(0, 0, pca_coeffs->cols, 1));
            cvCopy(&temp1, pca_coeffs);
            cvReleaseMat(&temp);
            cvReleaseMat(&patch_mat);
            //End of projecting
            
            cvReleaseImage(&patch_32f);
            cvReleaseImage(&test_img);
        }
        //--------
        
        
        
        for(int i = 0; i < desc_count; i++)
        {
            int _pose_idx = -1;
            float _distance = 0;
            
#if 0
            descriptors[i].EstimatePose(patch, _pose_idx, _distance);
#else
            if (!avg)
            {
                descriptors[i].EstimatePosePCA(patch, _pose_idx, _distance, avg, eigenvectors);
            }
            else
            {
                descriptors[i].EstimatePosePCA(pca_coeffs, _pose_idx, _distance, avg, eigenvectors);
            }
#endif
            
            for (int j=0;j<n;j++)
            {
                if(_distance < distances[j])
                {
                    for (int k=(n-1);k > j;k--)
                    {
                        desc_idxs[k] = desc_idxs[k-1];
                        pose_idxs[k] = pose_idxs[k-1];
                        distances[k] = distances[k-1];
                    }
                    desc_idxs[j] = i;
                    pose_idxs[j] = _pose_idx;
                    distances[j] = _distance;
                    break;
                }
            }
        }
        cvReleaseMat(&pca_coeffs);
    }
    
    void FindOneWayDescriptorEx(int desc_count, const OneWayDescriptor* descriptors, IplImage* patch,
                                float scale_min, float scale_max, float scale_step,
                                int& desc_idx, int& pose_idx, float& distance, float& scale,
                                CvMat* avg, CvMat* eigenvectors)
    {
        CvSize patch_size = descriptors[0].GetPatchSize();
        IplImage* input_patch;
        CvRect roi;
        
        input_patch= cvCreateImage(patch_size, IPL_DEPTH_8U, 1);
        roi = cvGetImageROI((IplImage*)patch);
        
        int _desc_idx, _pose_idx;
        float _distance;
        distance = 1e10;
        for(float cur_scale = scale_min; cur_scale < scale_max; cur_scale *= scale_step)
        {
            //        printf("Scale = %f\n", cur_scale);
            
            CvRect roi_scaled = resize_rect(roi, cur_scale);
            cvSetImageROI(patch, roi_scaled);
            cvResize(patch, input_patch);
            
            
#if 0
            if(roi.x > 244 && roi.y < 200)
            {
                cvNamedWindow("1", 1);
                cvShowImage("1", input_patch);
                cvWaitKey(0);
            }
#endif
            
            FindOneWayDescriptor(desc_count, descriptors, input_patch, _desc_idx, _pose_idx, _distance, avg, eigenvectors);
            if(_distance < distance)
            {
                distance = _distance;
                desc_idx = _desc_idx;
                pose_idx = _pose_idx;
                scale = cur_scale;
            }
        }
        
        
        cvSetImageROI((IplImage*)patch, roi);
        cvReleaseImage(&input_patch);
        
    }
    
    void FindOneWayDescriptorEx(int desc_count, const OneWayDescriptor* descriptors, IplImage* patch,
                                float scale_min, float scale_max, float scale_step,
                                int n, std::vector<int>& desc_idxs, std::vector<int>& pose_idxs,
                                std::vector<float>& distances, std::vector<float>& scales,
                                CvMat* avg, CvMat* eigenvectors)
    {
        CvSize patch_size = descriptors[0].GetPatchSize();
        IplImage* input_patch;
        CvRect roi;
        
        input_patch= cvCreateImage(patch_size, IPL_DEPTH_8U, 1);
        roi = cvGetImageROI((IplImage*)patch);
        
        //  float min_distance = 1e10;
        std::vector<int> _desc_idxs;
        _desc_idxs.resize(n);
        std::vector<int> _pose_idxs;
        _pose_idxs.resize(n);
        std::vector<float> _distances;
        _distances.resize(n);
        
        
        for (int i=0;i<n;i++)
        {
            distances[i] = 1e10;
        }
        
        for(float cur_scale = scale_min; cur_scale < scale_max; cur_scale *= scale_step)
        {
            
            CvRect roi_scaled = resize_rect(roi, cur_scale);
            cvSetImageROI(patch, roi_scaled);
            cvResize(patch, input_patch);
            
            
            
            FindOneWayDescriptor(desc_count, descriptors, input_patch, n,_desc_idxs, _pose_idxs, _distances, avg, eigenvectors);
            for (int i=0;i<n;i++)
            {
                if(_distances[i] < distances[i])
                {
                    distances[i] = _distances[i];
                    desc_idxs[i] = _desc_idxs[i];
                    pose_idxs[i] = _pose_idxs[i];
                    scales[i] = cur_scale;
                }
            }
        }
        
        
        
        cvSetImageROI((IplImage*)patch, roi);
        cvReleaseImage(&input_patch);
    }
    
#if defined(_KDTREE)
    void FindOneWayDescriptorEx(cv::flann::Index* m_pca_descriptors_tree, CvSize patch_size, int m_pca_dim_low,
                                int m_pose_count, IplImage* patch,
                                float scale_min, float scale_max, float scale_step,
                                int& desc_idx, int& pose_idx, float& distance, float& scale,
                                CvMat* avg, CvMat* eigenvectors)
    {
        IplImage* input_patch;
        CvRect roi;
        
        input_patch= cvCreateImage(patch_size, IPL_DEPTH_8U, 1);
        roi = cvGetImageROI((IplImage*)patch);
        
        int _desc_idx, _pose_idx;
        float _distance;
        distance = 1e10;
        for(float cur_scale = scale_min; cur_scale < scale_max; cur_scale *= scale_step)
        {
            //        printf("Scale = %f\n", cur_scale);
            
            CvRect roi_scaled = resize_rect(roi, cur_scale);
            cvSetImageROI(patch, roi_scaled);
            cvResize(patch, input_patch);
            
            FindOneWayDescriptor(m_pca_descriptors_tree, patch_size, m_pca_dim_low, m_pose_count, input_patch, _desc_idx, _pose_idx, _distance, avg, eigenvectors);
            if(_distance < distance)
            {
                distance = _distance;
                desc_idx = _desc_idx;
                pose_idx = _pose_idx;
                scale = cur_scale;
            }
        }
        
        
        cvSetImageROI((IplImage*)patch, roi);
        cvReleaseImage(&input_patch);
        
    }
#endif
    
    const char* OneWayDescriptor::GetFeatureName() const
    {
        return m_feature_name.c_str();
    }
    
    CvPoint OneWayDescriptor::GetCenter() const
    {
        return m_center;
    }
    
    int OneWayDescriptor::GetPCADimLow() const
    {
        return m_pca_dim_low;
    }
    
    int OneWayDescriptor::GetPCADimHigh() const
    {
        return m_pca_dim_high;
    }

    CvMat* ConvertImageToMatrix(IplImage* patch)
    {
        CvRect roi = cvGetImageROI(patch);
        CvMat* mat = cvCreateMat(1, roi.width*roi.height, CV_32FC1);
        
        if(patch->depth == 32)
        {
            for(int y = 0; y < roi.height; y++)
            {
                for(int x = 0; x < roi.width; x++)
                {
                    mat->data.fl[y*roi.width + x] = *((float*)(patch->imageData + (y + roi.y)*patch->widthStep) + x + roi.x);
                }
            }
        }
        else if(patch->depth == 8)
        {
            for(int y = 0; y < roi.height; y++)
            {
                for(int x = 0; x < roi.width; x++)
                {
                    mat->data.fl[y*roi.width + x] = (float)(unsigned char)patch->imageData[(y + roi.y)*patch->widthStep + x + roi.x];
                }
            }
        }
        else
        {
            printf("Image depth %d is not supported\n", patch->depth);
            return 0;
        }
        
        return mat;
    }
    
    OneWayDescriptorBase::OneWayDescriptorBase(CvSize patch_size, int pose_count, const char* train_path,
                                               const char* pca_config, const char* pca_hr_config,
                                               const char* pca_desc_config, int pyr_levels,
                                               int pca_dim_high, int pca_dim_low)
    : m_pca_dim_high(pca_dim_high), m_pca_dim_low(pca_dim_low), scale_min (0.7f), scale_max(1.5f), scale_step (1.2f)
    {
#if defined(_KDTREE)
        m_pca_descriptors_matrix = 0;
        m_pca_descriptors_tree = 0;
#endif
        //	m_pca_descriptors_matrix = 0;
        m_patch_size = patch_size;
        m_pose_count = pose_count;
        m_pyr_levels = pyr_levels;
        m_poses = 0;
        m_transforms = 0;
        
        m_pca_avg = 0;
        m_pca_eigenvectors = 0;
        m_pca_hr_avg = 0;
        m_pca_hr_eigenvectors = 0;
        m_pca_descriptors = 0;
        
        m_descriptors = 0;
        
        if(train_path == 0 || strlen(train_path) == 0)
        {
            // skip pca loading
            return;
        }
        char pca_config_filename[1024];
        sprintf(pca_config_filename, "%s/%s", train_path, pca_config);
        readPCAFeatures(pca_config_filename, &m_pca_avg, &m_pca_eigenvectors);
        if(pca_hr_config && strlen(pca_hr_config) > 0)
        {
            char pca_hr_config_filename[1024];
            sprintf(pca_hr_config_filename, "%s/%s", train_path, pca_hr_config);
            readPCAFeatures(pca_hr_config_filename, &m_pca_hr_avg, &m_pca_hr_eigenvectors);
        }
        
        m_pca_descriptors = new OneWayDescriptor[m_pca_dim_high + 1];
        
#if !defined(_GH_REGIONS)
        if(pca_desc_config && strlen(pca_desc_config) > 0)
            //    if(0)
        {
            //printf("Loading the descriptors...");
            char pca_desc_config_filename[1024];
            sprintf(pca_desc_config_filename, "%s/%s", train_path, pca_desc_config);
            LoadPCADescriptors(pca_desc_config_filename);
            //printf("done.\n");
        }
        else
        {
            printf("Initializing the descriptors...\n");
            InitializePoseTransforms();
            CreatePCADescriptors();
            SavePCADescriptors("pca_descriptors.yml");
        }
#endif //_GH_REGIONS
        //    SavePCADescriptors("./pca_descriptors.yml");
        
    }

    OneWayDescriptorBase::OneWayDescriptorBase(CvSize patch_size, int pose_count, const string &pca_filename,
                                               const string &train_path, const string &images_list, float _scale_min, float _scale_max,
                                               float _scale_step, int pyr_levels,
                                               int pca_dim_high, int pca_dim_low)
    : m_pca_dim_high(pca_dim_high), m_pca_dim_low(pca_dim_low), scale_min(_scale_min), scale_max(_scale_max), scale_step(_scale_step)
    {
#if defined(_KDTREE)
        m_pca_descriptors_matrix = 0;
        m_pca_descriptors_tree = 0;
#endif
        m_patch_size = patch_size;
        m_pose_count = pose_count;
        m_pyr_levels = pyr_levels;
        m_poses = 0;
        m_transforms = 0;

        m_pca_avg = 0;
        m_pca_eigenvectors = 0;
        m_pca_hr_avg = 0;
        m_pca_hr_eigenvectors = 0;
        m_pca_descriptors = 0;

        m_descriptors = 0;


        if (pca_filename.length() == 0)
        {
            return;
        }

        CvFileStorage* fs = cvOpenFileStorage(pca_filename.c_str(), NULL, CV_STORAGE_READ);
        if (fs != 0)
        {
            cvReleaseFileStorage(&fs);

            readPCAFeatures(pca_filename.c_str(), &m_pca_avg, &m_pca_eigenvectors, "_lr");
            readPCAFeatures(pca_filename.c_str(), &m_pca_hr_avg, &m_pca_hr_eigenvectors, "_hr");
            m_pca_descriptors = new OneWayDescriptor[m_pca_dim_high + 1];
#if !defined(_GH_REGIONS)
            LoadPCADescriptors(pca_filename.c_str());
#endif //_GH_REGIONS
        }
        else
        {
            GeneratePCA(train_path.c_str(), images_list.c_str());
            m_pca_descriptors = new OneWayDescriptor[m_pca_dim_high + 1];
            char pca_default_filename[1024];
            sprintf(pca_default_filename, "%s/%s", train_path.c_str(), GetPCAFilename().c_str());
            LoadPCADescriptors(pca_default_filename);
        }
    }

    void OneWayDescriptorBase::Read (const FileNode &fn)
    {
        clear ();

        m_pose_count = fn["poseCount"];
        int patch_width = fn["patchWidth"];
        int patch_height = fn["patchHeight"];
        m_patch_size = cvSize (patch_width, patch_height);
        m_pyr_levels = fn["pyrLevels"];
        m_pca_dim_high = fn["pcaDimHigh"];
        m_pca_dim_low = fn["pcaDimLow"];
        scale_min = fn["minScale"];
        scale_max = fn["maxScale"];
        scale_step = fn["stepScale"];
	
	LoadPCAall (fn);
    }

    void OneWayDescriptorBase::LoadPCAall (const FileNode &fn)
    {
        readPCAFeatures(fn, &m_pca_avg, &m_pca_eigenvectors, "_lr");
        readPCAFeatures(fn, &m_pca_hr_avg, &m_pca_hr_eigenvectors, "_hr");
        m_pca_descriptors = new OneWayDescriptor[m_pca_dim_high + 1];
#if !defined(_GH_REGIONS)
        LoadPCADescriptors(fn);
#endif //_GH_REGIONS
    }

    OneWayDescriptorBase::~OneWayDescriptorBase()
    {
        cvReleaseMat(&m_pca_avg);
        cvReleaseMat(&m_pca_eigenvectors);
        
        if(m_pca_hr_eigenvectors)
        {
            delete[] m_pca_descriptors;
            cvReleaseMat(&m_pca_hr_avg);
            cvReleaseMat(&m_pca_hr_eigenvectors);
        }
        
        
        if(m_descriptors)
            delete []m_descriptors;

        if(m_poses)
            delete []m_poses;
        
        if (m_transforms)
        {
            for(int i = 0; i < m_pose_count; i++)
            {
                cvReleaseMat(&m_transforms[i]);
            }
            delete []m_transforms;
        }
#if defined(_KDTREE)
        if (m_pca_descriptors_matrix)
        {
            cvReleaseMat(&m_pca_descriptors_matrix);
        }
        if (m_pca_descriptors_tree)
        {
            delete m_pca_descriptors_tree;
        }
#endif
    }
    
    void OneWayDescriptorBase::clear(){
        if (m_descriptors)
        {
            delete []m_descriptors;
            m_descriptors = 0;
        }

#if defined(_KDTREE)
        if (m_pca_descriptors_matrix)
        {
            cvReleaseMat(&m_pca_descriptors_matrix);
            m_pca_descriptors_matrix = 0;
        }
        if (m_pca_descriptors_tree)
        {
            delete m_pca_descriptors_tree;
            m_pca_descriptors_tree = 0;
        }
#endif
    }

    void OneWayDescriptorBase::InitializePoses()
    {
        m_poses = new CvAffinePose[m_pose_count];
        for(int i = 0; i < m_pose_count; i++)
        {
            m_poses[i] = GenRandomAffinePose();
        }
    }
    
    void OneWayDescriptorBase::InitializeTransformsFromPoses()
    {
        m_transforms = new CvMat*[m_pose_count];
        for(int i = 0; i < m_pose_count; i++)
        {
            m_transforms[i] = cvCreateMat(2, 3, CV_32FC1);
            GenerateAffineTransformFromPose(cvSize(m_patch_size.width*2, m_patch_size.height*2), m_poses[i], m_transforms[i]);
        }
    }
    
    void OneWayDescriptorBase::InitializePoseTransforms()
    {
        InitializePoses();
        InitializeTransformsFromPoses();
    }
    
    void OneWayDescriptorBase::InitializeDescriptor(int desc_idx, IplImage* train_image, const KeyPoint& keypoint, const char* feature_label)
    {
        
        // TBD add support for octave != 0
        CvPoint center = keypoint.pt;
        
        CvRect roi = cvRect(center.x - m_patch_size.width/2, center.y - m_patch_size.height/2, m_patch_size.width, m_patch_size.height);
        cvResetImageROI(train_image);
        roi = fit_rect_fixedsize(roi, train_image);
        cvSetImageROI(train_image, roi);
        if(roi.width != m_patch_size.width || roi.height != m_patch_size.height)
        {
            return;
        }
        
        InitializeDescriptor(desc_idx, train_image, feature_label);
        cvResetImageROI(train_image);
    }
    
    void OneWayDescriptorBase::InitializeDescriptor(int desc_idx, IplImage* train_image, const char* feature_label)
    {
        m_descriptors[desc_idx].SetPCADimHigh(m_pca_dim_high);
        m_descriptors[desc_idx].SetPCADimLow(m_pca_dim_low);
        m_descriptors[desc_idx].SetTransforms(m_poses, m_transforms);
        
        if(!m_pca_hr_eigenvectors)
        {
            m_descriptors[desc_idx].Initialize(m_pose_count, train_image, feature_label);
        }
        else
        {
            m_descriptors[desc_idx].InitializeFast(m_pose_count, train_image, feature_label,
                                                   m_pca_hr_avg, m_pca_hr_eigenvectors, m_pca_descriptors);
        }
        
        if(m_pca_avg)
        {
            m_descriptors[desc_idx].InitializePCACoeffs(m_pca_avg, m_pca_eigenvectors);
        }
    }
    
    void OneWayDescriptorBase::FindDescriptor(IplImage* src, cv::Point2f pt, int& desc_idx, int& pose_idx, float& distance) const
    {
        CvRect roi = cvRect(cvRound(pt.x - m_patch_size.width/4),
                            cvRound(pt.y - m_patch_size.height/4),
                            m_patch_size.width/2, m_patch_size.height/2);
        cvSetImageROI(src, roi);
        
        FindDescriptor(src, desc_idx, pose_idx, distance);
        cvResetImageROI(src);   
    }
    
    void OneWayDescriptorBase::FindDescriptor(IplImage* patch, int& desc_idx, int& pose_idx, float& distance, float* _scale, float* scale_ranges) const
    {
#if 0
        ::FindOneWayDescriptor(m_train_feature_count, m_descriptors, patch, desc_idx, pose_idx, distance, m_pca_avg, m_pca_eigenvectors);
#else
        float min = scale_min;
        float max = scale_max;
        float step = scale_step;
        
        if (scale_ranges)
        {
            min = scale_ranges[0];
            max = scale_ranges[1];
        }
        
        float scale = 1.0f;
        
#if !defined(_KDTREE)
        cv::FindOneWayDescriptorEx(m_train_feature_count, m_descriptors, patch,
                                   min, max, step, desc_idx, pose_idx, distance, scale,
                                   m_pca_avg, m_pca_eigenvectors);
#else
        cv::FindOneWayDescriptorEx(m_pca_descriptors_tree, m_descriptors[0].GetPatchSize(), m_descriptors[0].GetPCADimLow(), m_pose_count, patch,
                                   min, max, step, desc_idx, pose_idx, distance, scale,
                                   m_pca_avg, m_pca_eigenvectors);
#endif
        
        if (_scale)
            *_scale = scale;
        
#endif
    }
    
    void OneWayDescriptorBase::FindDescriptor(IplImage* patch, int n, std::vector<int>& desc_idxs, std::vector<int>& pose_idxs,
                                              std::vector<float>& distances, std::vector<float>& _scales, float* scale_ranges) const
    {
        float min = scale_min;
        float max = scale_max;
        float step = scale_step;
        
        if (scale_ranges)
        {
            min = scale_ranges[0];
            max = scale_ranges[1];
        }
        
        distances.resize(n);
        _scales.resize(n);
        desc_idxs.resize(n);
        pose_idxs.resize(n);
        /*float scales = 1.0f;*/
        
        cv::FindOneWayDescriptorEx(m_train_feature_count, m_descriptors, patch,
                                   min, max, step ,n, desc_idxs, pose_idxs, distances, _scales,
                                   m_pca_avg, m_pca_eigenvectors);
        
    }
    
    void OneWayDescriptorBase::SetPCAHigh(CvMat* avg, CvMat* eigenvectors)
    {
        m_pca_hr_avg = cvCloneMat(avg);
        m_pca_hr_eigenvectors = cvCloneMat(eigenvectors);
    }
    
    void OneWayDescriptorBase::SetPCALow(CvMat* avg, CvMat* eigenvectors)
    {
        m_pca_avg = cvCloneMat(avg);
        m_pca_eigenvectors = cvCloneMat(eigenvectors);
    }
    
    void OneWayDescriptorBase::AllocatePCADescriptors()
    {
        m_pca_descriptors = new OneWayDescriptor[m_pca_dim_high + 1];
        for(int i = 0; i < m_pca_dim_high + 1; i++)
        {
            m_pca_descriptors[i].SetPCADimHigh(m_pca_dim_high);
            m_pca_descriptors[i].SetPCADimLow(m_pca_dim_low);
        }
    }
    
    void OneWayDescriptorBase::CreatePCADescriptors()
    {
        if(m_pca_descriptors == 0)
        {
            AllocatePCADescriptors();
        }
        IplImage* frontal = cvCreateImage(m_patch_size, IPL_DEPTH_32F, 1);
        
        eigenvector2image(m_pca_hr_avg, frontal);
        m_pca_descriptors[0].SetTransforms(m_poses, m_transforms);
        m_pca_descriptors[0].Initialize(m_pose_count, frontal, "", 0);
        
        for(int j = 0; j < m_pca_dim_high; j++)
        {
            CvMat eigenvector;
            cvGetSubRect(m_pca_hr_eigenvectors, &eigenvector, cvRect(0, j, m_pca_hr_eigenvectors->cols, 1));
            eigenvector2image(&eigenvector, frontal);
            
            m_pca_descriptors[j + 1].SetTransforms(m_poses, m_transforms);
            m_pca_descriptors[j + 1].Initialize(m_pose_count, frontal, "", 0);
            
            printf("Created descriptor for PCA component %d\n", j);
        }
        
        cvReleaseImage(&frontal);
    }
    
    
    int OneWayDescriptorBase::LoadPCADescriptors(const char* filename)
    {
        FileStorage fs = FileStorage (filename, FileStorage::READ);
        if(!fs.isOpened ())
        {
            printf("File %s not found...\n", filename);
            return 0;
        }

        LoadPCADescriptors (fs.root ());

        printf("Successfully read %d pca components\n", m_pca_dim_high);
        fs.release ();
        
        return 1;
    }

    int OneWayDescriptorBase::LoadPCADescriptors(const FileNode &fn)
    {
        // read affine poses
//            FileNode* node = cvGetFileNodeByName(fs, 0, "affine poses");
        CvMat* poses = reinterpret_cast<CvMat*> (fn["affine_poses"].readObj ());
        if (poses == 0)
        {
            poses = reinterpret_cast<CvMat*> (fn["affine poses"].readObj ());
            if (poses == 0)
                return 0;
        }


        if(m_poses)
        {
            delete m_poses;
        }
        m_poses = new CvAffinePose[m_pose_count];
        for(int i = 0; i < m_pose_count; i++)
        {
            m_poses[i].phi = (float)cvmGet(poses, i, 0);
            m_poses[i].theta = (float)cvmGet(poses, i, 1);
            m_poses[i].lambda1 = (float)cvmGet(poses, i, 2);
            m_poses[i].lambda2 = (float)cvmGet(poses, i, 3);
        }
        cvReleaseMat(&poses);

        // now initialize pose transforms
        InitializeTransformsFromPoses();

        m_pca_dim_high = (int) fn["pca_components_number"];
        if (m_pca_dim_high == 0)
        {
            m_pca_dim_high = (int) fn["pca components number"];
        }
        if(m_pca_descriptors)
        {
            delete []m_pca_descriptors;
        }
        AllocatePCADescriptors();
        for(int i = 0; i < m_pca_dim_high + 1; i++)
        {
            m_pca_descriptors[i].Allocate(m_pose_count, m_patch_size, 1);
            m_pca_descriptors[i].SetTransforms(m_poses, m_transforms);
            char buf[1024];
            sprintf(buf, "descriptor_for_pca_component_%d", i);

            if (! m_pca_descriptors[i].ReadByName(fn, buf))
            {
                char buf[1024];
                sprintf(buf, "descriptor for pca component %d", i);
                m_pca_descriptors[i].ReadByName(fn, buf);
            }
        }
        return 1;
    }


    void savePCAFeatures(FileStorage &fs, const char* postfix, CvMat* avg, CvMat* eigenvectors)
    {
        char buf[1024];
        sprintf(buf, "avg_%s", postfix);
        fs.writeObj(buf, avg);
        sprintf(buf, "eigenvectors_%s", postfix);
        fs.writeObj(buf, eigenvectors);
    }

    void calcPCAFeatures(vector<IplImage*>& patches, FileStorage &fs, const char* postfix, CvMat** avg,
                         CvMat** eigenvectors)
    {
        int width = patches[0]->width;
        int height = patches[0]->height;
        int length = width * height;
        int patch_count = (int)patches.size();

        CvMat* data = cvCreateMat(patch_count, length, CV_32FC1);
        *avg = cvCreateMat(1, length, CV_32FC1);
        CvMat* eigenvalues = cvCreateMat(1, length, CV_32FC1);
        *eigenvectors = cvCreateMat(length, length, CV_32FC1);

        for (int i = 0; i < patch_count; i++)
        {
            float sum = cvSum(patches[i]).val[0];
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    *((float*)(data->data.ptr + data->step * i) + y * width + x)
                            = (float)(unsigned char)patches[i]->imageData[y * patches[i]->widthStep + x] / sum;
                }
            }
        }

        //printf("Calculating PCA...");
        cvCalcPCA(data, *avg, eigenvalues, *eigenvectors, CV_PCA_DATA_AS_ROW);
        //printf("done\n");

        // save pca data
        savePCAFeatures(fs, postfix, *avg, *eigenvectors);

        cvReleaseMat(&data);
        cvReleaseMat(&eigenvalues);
    }

    void extractPatches (IplImage *img, vector<IplImage*>& patches, CvSize patch_size)
    {
        vector<KeyPoint> features;
        SURF surf_extractor(1.0f);
        //printf("Extracting SURF features...");
        surf_extractor(img, Mat(), features);
        //printf("done\n");

        for (int j = 0; j < (int)features.size(); j++)
        {
            int patch_width = patch_size.width;
            int patch_height = patch_size.height;

            CvPoint center = features[j].pt;

            CvRect roi = cvRect(center.x - patch_width / 2, center.y - patch_height / 2, patch_width, patch_height);
            cvSetImageROI(img, roi);
            roi = cvGetImageROI(img);
            if (roi.width != patch_width || roi.height != patch_height)
            {
                continue;
            }

            IplImage* patch = cvCreateImage(cvSize(patch_width, patch_height), IPL_DEPTH_8U, 1);
            cvCopy(img, patch);
            patches.push_back(patch);
            cvResetImageROI(img);
        }
        //printf("Completed file, extracted %d features\n", (int)features.size());
    }

/*
    void loadPCAFeatures(const FileNode &fn, vector<IplImage*>& patches, CvSize patch_size)
    {
        FileNodeIterator begin = fn.begin();
        for (FileNodeIterator i = fn.begin(); i != fn.end(); i++)
        {
            IplImage *img = reinterpret_cast<IplImage*> ((*i).readObj());
            extractPatches (img, patches, patch_size);
            cvReleaseImage(&img);
        }
    }
*/

    void loadPCAFeatures(const char* path, const char* images_list, vector<IplImage*>& patches, CvSize patch_size)
    {
        char images_filename[1024];
        sprintf(images_filename, "%s/%s", path, images_list);
        FILE *pFile = fopen(images_filename, "r");
        if (pFile == 0)
        {
            printf("Cannot open images list file %s\n", images_filename);
            return;
        }
        while (!feof(pFile))
        {
            char imagename[1024];
            if (fscanf(pFile, "%s", imagename) <= 0)
            {
                break;
            }

            char filename[1024];
            sprintf(filename, "%s/%s", path, imagename);

            //printf("Reading image %s...", filename);
            IplImage* img = cvLoadImage(filename, CV_LOAD_IMAGE_GRAYSCALE);
            //printf("done\n");

            extractPatches (img, patches, patch_size);

            cvReleaseImage(&img);
        }
        fclose(pFile);
    }

    void generatePCAFeatures(const char* path, const char* img_filename, FileStorage& fs, const char* postfix,
                             CvSize patch_size, CvMat** avg, CvMat** eigenvectors)
    {
        vector<IplImage*> patches;
        loadPCAFeatures(path, img_filename, patches, patch_size);
        calcPCAFeatures(patches, fs, postfix, avg, eigenvectors);
    }

/*
    void generatePCAFeatures(const FileNode &fn, const char* postfix,
                             CvSize patch_size, CvMat** avg, CvMat** eigenvectors)
    {
        vector<IplImage*> patches;
        loadPCAFeatures(fn, patches, patch_size);
        calcPCAFeatures(patches, fs, postfix, avg, eigenvectors);
    }


    void OneWayDescriptorBase::GeneratePCA(const FileNode &fn, int pose_count)
    {
        generatePCAFeatures(fn, "hr", m_patch_size, &m_pca_hr_avg, &m_pca_hr_eigenvectors);
        generatePCAFeatures(fn, "lr", cvSize(m_patch_size.width / 2, m_patch_size.height / 2),
                            &m_pca_avg, &m_pca_eigenvectors);


        OneWayDescriptorBase descriptors(m_patch_size, pose_count);
        descriptors.SetPCAHigh(m_pca_hr_avg, m_pca_hr_eigenvectors);
        descriptors.SetPCALow(m_pca_avg, m_pca_eigenvectors);

        printf("Calculating %d PCA descriptors (you can grab a coffee, this will take a while)...\n",
               descriptors.GetPCADimHigh());
        descriptors.InitializePoseTransforms();
        descriptors.CreatePCADescriptors();
        descriptors.SavePCADescriptors(*fs);
    }
*/

    void OneWayDescriptorBase::GeneratePCA(const char* img_path, const char* images_list, int pose_count)
    {
        char pca_filename[1024];
        sprintf(pca_filename, "%s/%s", img_path, GetPCAFilename().c_str());
        FileStorage fs = FileStorage(pca_filename, FileStorage::WRITE);

        generatePCAFeatures(img_path, images_list, fs, "hr", m_patch_size, &m_pca_hr_avg, &m_pca_hr_eigenvectors);
        generatePCAFeatures(img_path, images_list, fs, "lr", cvSize(m_patch_size.width / 2, m_patch_size.height / 2),
                            &m_pca_avg, &m_pca_eigenvectors);

        OneWayDescriptorBase descriptors(m_patch_size, pose_count);
        descriptors.SetPCAHigh(m_pca_hr_avg, m_pca_hr_eigenvectors);
        descriptors.SetPCALow(m_pca_avg, m_pca_eigenvectors);

        printf("Calculating %d PCA descriptors (you can grab a coffee, this will take a while)...\n",
               descriptors.GetPCADimHigh());
        descriptors.InitializePoseTransforms();
        descriptors.CreatePCADescriptors();
        descriptors.SavePCADescriptors(*fs);

        fs.release();
    }

    void OneWayDescriptorBase::Write (FileStorage &fs) const
    {
        fs << "poseCount" << m_pose_count;
        fs << "patchWidth" << m_patch_size.width;
        fs << "patchHeight" << m_patch_size.height;
        fs << "minScale" << scale_min;
        fs << "maxScale" << scale_max;
        fs << "stepScale" << scale_step;
        fs << "pyrLevels" << m_pyr_levels;
        fs << "pcaDimHigh" << m_pca_dim_high;
        fs << "pcaDimLow" << m_pca_dim_low;

        SavePCAall (fs);
    }

    void OneWayDescriptorBase::SavePCAall (FileStorage &fs) const
    {
        savePCAFeatures(fs, "hr", m_pca_hr_avg, m_pca_hr_eigenvectors);
        savePCAFeatures(fs, "lr", m_pca_avg, m_pca_eigenvectors);
        SavePCADescriptors(*fs);
    }

    void OneWayDescriptorBase::SavePCADescriptors(const char* filename)
    {
        CvMemStorage* storage = cvCreateMemStorage();
        CvFileStorage* fs = cvOpenFileStorage(filename, storage, CV_STORAGE_WRITE);
        
        SavePCADescriptors (fs);
        
        cvReleaseMemStorage(&storage);
        cvReleaseFileStorage(&fs);
    }
    
    void OneWayDescriptorBase::SavePCADescriptors(CvFileStorage *fs) const
    {
        cvWriteInt(fs, "pca_components_number", m_pca_dim_high);
        cvWriteComment(
                       fs,
                       "The first component is the average Vector, so the total number of components is <pca components number> + 1",
                       0);
        cvWriteInt(fs, "patch_width", m_patch_size.width);
        cvWriteInt(fs, "patch_height", m_patch_size.height);

        // pack the affine transforms into a single CvMat and write them
        CvMat* poses = cvCreateMat(m_pose_count, 4, CV_32FC1);
        for (int i = 0; i < m_pose_count; i++)
        {
            cvmSet(poses, i, 0, m_poses[i].phi);
            cvmSet(poses, i, 1, m_poses[i].theta);
            cvmSet(poses, i, 2, m_poses[i].lambda1);
            cvmSet(poses, i, 3, m_poses[i].lambda2);
        }
        cvWrite(fs, "affine_poses", poses);
        cvReleaseMat(&poses);

        for (int i = 0; i < m_pca_dim_high + 1; i++)
        {
            char buf[1024];
            sprintf(buf, "descriptor_for_pca_component_%d", i);
            m_pca_descriptors[i].Write(fs, buf);
        }
    }


    void OneWayDescriptorBase::Allocate(int train_feature_count)
    {
        m_train_feature_count = train_feature_count;
        m_descriptors = new OneWayDescriptor[m_train_feature_count];
        for(int i = 0; i < m_train_feature_count; i++)
        {
            m_descriptors[i].SetPCADimHigh(m_pca_dim_high);
            m_descriptors[i].SetPCADimLow(m_pca_dim_low);
        }
    }
    
    void OneWayDescriptorBase::InitializeDescriptors(IplImage* train_image, const vector<KeyPoint>& features,
                                                     const char* feature_label, int desc_start_idx)
    {
        for(int i = 0; i < (int)features.size(); i++)
        {
            InitializeDescriptor(desc_start_idx + i, train_image, features[i], feature_label);
            
        }
        cvResetImageROI(train_image);
        
#if defined(_KDTREE)
        ConvertDescriptorsArrayToTree();
#endif
    }
    
    void OneWayDescriptorBase::CreateDescriptorsFromImage(IplImage* src, const std::vector<KeyPoint>& features)
    {
        m_train_feature_count = (int)features.size();
        
        m_descriptors = new OneWayDescriptor[m_train_feature_count];
        
        InitializeDescriptors(src, features);
        
    }
    
#if defined(_KDTREE)
    void OneWayDescriptorBase::ConvertDescriptorsArrayToTree()
    {
        int n = this->GetDescriptorCount();
        if (n<1)
            return;
        int pca_dim_low = this->GetDescriptor(0)->GetPCADimLow();
        
        //if (!m_pca_descriptors_matrix)
        //	m_pca_descriptors_matrix = new ::cvflann::Matrix<float>(n*m_pose_count,pca_dim_low);
        //else
        //{
        //	if ((m_pca_descriptors_matrix->cols != pca_dim_low)&&(m_pca_descriptors_matrix->rows != n*m_pose_count))
        //	{
        //		delete m_pca_descriptors_matrix;
        //		m_pca_descriptors_matrix = new ::cvflann::Matrix<float>(n*m_pose_count,pca_dim_low);
        //	}
        //}
        
        m_pca_descriptors_matrix = cvCreateMat(n*m_pose_count,pca_dim_low,CV_32FC1);
        for (int i=0;i<n;i++)
        {
            CvMat** pca_coeffs = m_descriptors[i].GetPCACoeffs();
            for (int j = 0;j<m_pose_count;j++)
            {
                for (int k=0;k<pca_dim_low;k++)
                {
                    m_pca_descriptors_matrix->data.fl[(i*m_pose_count+j)*m_pca_dim_low + k] = pca_coeffs[j]->data.fl[k];
                }
            }
        }
        cv::Mat pca_descriptors_mat(m_pca_descriptors_matrix,false);
        
        //::cvflann::KDTreeIndexParams params;
        //params.trees = 1;
        //m_pca_descriptors_tree = new KDTree(pca_descriptors_mat);
        m_pca_descriptors_tree = new cv::flann::Index(pca_descriptors_mat,cv::flann::KDTreeIndexParams(1));
        //cvReleaseMat(&m_pca_descriptors_matrix);
        //m_pca_descriptors_tree->buildIndex();
    }
#endif
    
    void OneWayDescriptorObject::Allocate(int train_feature_count, int object_feature_count)
    {
        OneWayDescriptorBase::Allocate(train_feature_count);
        m_object_feature_count = object_feature_count;
        
        m_part_id = new int[m_object_feature_count];
    }
    
    
    void OneWayDescriptorObject::InitializeObjectDescriptors(IplImage* train_image, const vector<KeyPoint>& features,
                                                             const char* feature_label, int desc_start_idx, float scale, int is_background)
    {
        InitializeDescriptors(train_image, features, feature_label, desc_start_idx);
        
        for(int i = 0; i < (int)features.size(); i++)
        {
            CvPoint center = features[i].pt;
            
            if(!is_background)
            {
                // remember descriptor part id
                CvPoint center_scaled = cvPoint(round(center.x*scale), round(center.y*scale));
                m_part_id[i + desc_start_idx] = MatchPointToPart(center_scaled);
            }
        }
        cvResetImageROI(train_image);
    }
    
    int OneWayDescriptorObject::IsDescriptorObject(int desc_idx) const
    {
        return desc_idx < m_object_feature_count ? 1 : 0;
    }
    
    int OneWayDescriptorObject::MatchPointToPart(CvPoint pt) const
    {
        int idx = -1;
        const int max_dist = 10;
        for(int i = 0; i < (int)m_train_features.size(); i++)
        {
            if(norm(Point2f(pt) - m_train_features[i].pt) < max_dist)
            {
                idx = i;
                break;
            }
        }
        
        return idx;
    }
    
    int OneWayDescriptorObject::GetDescriptorPart(int desc_idx) const
    {
        //    return MatchPointToPart(GetDescriptor(desc_idx)->GetCenter());
        return desc_idx < m_object_feature_count ? m_part_id[desc_idx] : -1;
    }
    
    OneWayDescriptorObject::OneWayDescriptorObject(CvSize patch_size, int pose_count, const char* train_path,
                                                   const char* pca_config, const char* pca_hr_config, const char* pca_desc_config, int pyr_levels) :
    OneWayDescriptorBase(patch_size, pose_count, train_path, pca_config, pca_hr_config, pca_desc_config, pyr_levels)
    {
        m_part_id = 0;
    }
    
    OneWayDescriptorObject::OneWayDescriptorObject(CvSize patch_size, int pose_count, const string &pca_filename,
                                                   const string &train_path, const string &images_list, float _scale_min, float _scale_max, float _scale_step, int pyr_levels) :
    OneWayDescriptorBase(patch_size, pose_count, pca_filename, train_path, images_list, _scale_min, _scale_max, _scale_step, pyr_levels)
    {
        m_part_id = 0;
    }

    OneWayDescriptorObject::~OneWayDescriptorObject()
    {
        if (m_part_id)
            delete []m_part_id;
    }
    
    vector<KeyPoint> OneWayDescriptorObject::_GetLabeledFeatures() const
    {
        vector<KeyPoint> features;
        for(size_t i = 0; i < m_train_features.size(); i++)
        {
            features.push_back(m_train_features[i]);
        }
        
        return features;
    }
    
    void eigenvector2image(CvMat* eigenvector, IplImage* img)
    {
        CvRect roi = cvGetImageROI(img);
        if(img->depth == 32)
        {
            for(int y = 0; y < roi.height; y++)
            {
                for(int x = 0; x < roi.width; x++)
                {
                    float val = (float)cvmGet(eigenvector, 0, roi.width*y + x);
                    *((float*)(img->imageData + (roi.y + y)*img->widthStep) + roi.x + x) = val;
                }
            }
        }
        else
        {
            for(int y = 0; y < roi.height; y++)
            {
                for(int x = 0; x < roi.width; x++)
                {
                    float val = (float)cvmGet(eigenvector, 0, roi.width*y + x);
                    img->imageData[(roi.y + y)*img->widthStep + roi.x + x] = (unsigned char)val;
                }
            }
        }
    }

    void readPCAFeatures(const char* filename, CvMat** avg, CvMat** eigenvectors, const char* postfix)
    {
        FileStorage fs = FileStorage(filename, FileStorage::READ);
        if (!fs.isOpened ())
        {
            printf("Cannot open file %s! Exiting!", filename);
        }

        readPCAFeatures (fs.root (), avg, eigenvectors, postfix);
        fs.release ();
    }

    void readPCAFeatures(const FileNode &fn, CvMat** avg, CvMat** eigenvectors, const char* postfix)
    {
        std::string str = std::string ("avg") + postfix;
        CvMat* _avg = reinterpret_cast<CvMat*> (fn[str].readObj());
        if (_avg != 0)
        {
            *avg = cvCloneMat(_avg);
            cvReleaseMat(&_avg);
        }

        str = std::string ("eigenvectors") + postfix;
        CvMat* _eigenvectors = reinterpret_cast<CvMat*> (fn[str].readObj());
        if (_eigenvectors != 0)
        {
            *eigenvectors = cvCloneMat(_eigenvectors);
            cvReleaseMat(&_eigenvectors);
        }
    }
}
