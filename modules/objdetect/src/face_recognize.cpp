// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

#include "opencv2/dnn.hpp"

#include <algorithm>

namespace cv
{

class FaceRecognizerSFImpl : public FaceRecognizerSF
{
public:
    FaceRecognizerSFImpl(const String& model, const String& config, int backend_id, int target_id)
    {
        net = dnn::readNet(model, config);
        CV_Assert(!net.empty());

        net.setPreferableBackend(backend_id);
        net.setPreferableTarget(target_id);
    };
    void alignCrop(InputArray _src_img, InputArray _face_mat, OutputArray _aligned_img) const override
    {
        Mat face_mat = _face_mat.getMat();
        float src_point[5][2];
        for (int row = 0; row < 5; ++row)
        {
            for(int col = 0; col < 2; ++col)
            {
                src_point[row][col] = face_mat.at<float>(0, row*2+col+4);
            }
        }
        Mat warp_mat = getSimilarityTransformMatrix(src_point);
        warpAffine(_src_img, _aligned_img, warp_mat, Size(112, 112), INTER_LINEAR);
    };
    void feature(InputArray _aligned_img, OutputArray _face_feature) override
    {
        Mat inputBolb = dnn::blobFromImage(_aligned_img, 1, Size(112, 112), Scalar(0, 0, 0), true, false);
        net.setInput(inputBolb);
        net.forward(_face_feature);
    };
    double match(InputArray _face_feature1, InputArray _face_feature2, int dis_type) const override
    {
        Mat face_feature1 = _face_feature1.getMat(), face_feature2 = _face_feature2.getMat();
        face_feature1 /= norm(face_feature1);
        face_feature2 /= norm(face_feature2);

        if(dis_type == DisType::FR_COSINE){
            return sum(face_feature1.mul(face_feature2))[0];
        }else if(dis_type == DisType::FR_NORM_L2){
            return norm(face_feature1, face_feature2);
        }else{
            throw std::invalid_argument("invalid parameter " + std::to_string(dis_type));
        }

    };

private:
    Mat getSimilarityTransformMatrix(float src[5][2]) const {
        float dst[5][2] = { {38.2946f, 51.6963f}, {73.5318f, 51.5014f}, {56.0252f, 71.7366f}, {41.5493f, 92.3655f}, {70.7299f, 92.2041f} };
        float avg0 = (src[0][0] + src[1][0] + src[2][0] + src[3][0] + src[4][0]) / 5;
        float avg1 = (src[0][1] + src[1][1] + src[2][1] + src[3][1] + src[4][1]) / 5;
        //Compute mean of src and dst.
        float src_mean[2] = { avg0, avg1 };
        float dst_mean[2] = { 56.0262f, 71.9008f };
        //Subtract mean from src and dst.
        float src_demean[5][2];
        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 5; j++)
            {
                src_demean[j][i] = src[j][i] - src_mean[i];
            }
        }
        float dst_demean[5][2];
        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 5; j++)
            {
                dst_demean[j][i] = dst[j][i] - dst_mean[i];
            }
        }
        double A00 = 0.0, A01 = 0.0, A10 = 0.0, A11 = 0.0;
        for (int i = 0; i < 5; i++)
            A00 += dst_demean[i][0] * src_demean[i][0];
        A00 = A00 / 5;
        for (int i = 0; i < 5; i++)
            A01 += dst_demean[i][0] * src_demean[i][1];
        A01 = A01 / 5;
        for (int i = 0; i < 5; i++)
            A10 += dst_demean[i][1] * src_demean[i][0];
        A10 = A10 / 5;
        for (int i = 0; i < 5; i++)
            A11 += dst_demean[i][1] * src_demean[i][1];
        A11 = A11 / 5;
        Mat A = (Mat_<double>(2, 2) << A00, A01, A10, A11);
        double d[2] = { 1.0, 1.0 };
        double detA = A00 * A11 - A01 * A10;
        if (detA < 0)
            d[1] = -1;
        double T[3][3] = { {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0} };
        Mat s, u, vt, v;
        SVD::compute(A, s, u, vt);
        double smax = s.ptr<double>(0)[0]>s.ptr<double>(1)[0] ? s.ptr<double>(0)[0] : s.ptr<double>(1)[0];
        double tol = smax * 2 * FLT_MIN;
        int rank = 0;
        if (s.ptr<double>(0)[0]>tol)
            rank += 1;
        if (s.ptr<double>(1)[0]>tol)
            rank += 1;
        double arr_u[2][2] = { {u.ptr<double>(0)[0], u.ptr<double>(0)[1]}, {u.ptr<double>(1)[0], u.ptr<double>(1)[1]} };
        double arr_vt[2][2] = { {vt.ptr<double>(0)[0], vt.ptr<double>(0)[1]}, {vt.ptr<double>(1)[0], vt.ptr<double>(1)[1]} };
        double det_u = arr_u[0][0] * arr_u[1][1] - arr_u[0][1] * arr_u[1][0];
        double det_vt = arr_vt[0][0] * arr_vt[1][1] - arr_vt[0][1] * arr_vt[1][0];
        if (rank == 1)
        {
            if ((det_u*det_vt) > 0)
            {
                Mat uvt = u*vt;
                T[0][0] = uvt.ptr<double>(0)[0];
                T[0][1] = uvt.ptr<double>(0)[1];
                T[1][0] = uvt.ptr<double>(1)[0];
                T[1][1] = uvt.ptr<double>(1)[1];
            }
            else
            {
                double temp = d[1];
                d[1] = -1;
                Mat D = (Mat_<double>(2, 2) << d[0], 0.0, 0.0, d[1]);
                Mat Dvt = D*vt;
                Mat uDvt = u*Dvt;
                T[0][0] = uDvt.ptr<double>(0)[0];
                T[0][1] = uDvt.ptr<double>(0)[1];
                T[1][0] = uDvt.ptr<double>(1)[0];
                T[1][1] = uDvt.ptr<double>(1)[1];
                d[1] = temp;
            }
        }
        else
        {
            Mat D = (Mat_<double>(2, 2) << d[0], 0.0, 0.0, d[1]);
            Mat Dvt = D*vt;
            Mat uDvt = u*Dvt;
            T[0][0] = uDvt.ptr<double>(0)[0];
            T[0][1] = uDvt.ptr<double>(0)[1];
            T[1][0] = uDvt.ptr<double>(1)[0];
            T[1][1] = uDvt.ptr<double>(1)[1];
        }
        double var1 = 0.0;
        for (int i = 0; i < 5; i++)
            var1 += src_demean[i][0] * src_demean[i][0];
        var1 = var1 / 5;
        double var2 = 0.0;
        for (int i = 0; i < 5; i++)
            var2 += src_demean[i][1] * src_demean[i][1];
        var2 = var2 / 5;
        double scale = 1.0 / (var1 + var2)* (s.ptr<double>(0)[0] * d[0] + s.ptr<double>(1)[0] * d[1]);
        double TS[2];
        TS[0] = T[0][0] * src_mean[0] + T[0][1] * src_mean[1];
        TS[1] = T[1][0] * src_mean[0] + T[1][1] * src_mean[1];
        T[0][2] = dst_mean[0] - scale*TS[0];
        T[1][2] = dst_mean[1] - scale*TS[1];
        T[0][0] *= scale;
        T[0][1] *= scale;
        T[1][0] *= scale;
        T[1][1] *= scale;
        Mat transform_mat = (Mat_<double>(2, 3) << T[0][0], T[0][1], T[0][2], T[1][0], T[1][1], T[1][2]);
        return transform_mat;
    }
private:
    dnn::Net net;
};

Ptr<FaceRecognizerSF> FaceRecognizerSF::create(const String& model, const String& config, int backend_id, int target_id)
{
    return makePtr<FaceRecognizerSFImpl>(model, config, backend_id, target_id);
}

} // namespace cv
