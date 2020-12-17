// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include <iostream>
#include <unordered_map>

#include "opencv2/core/base.hpp"
#include "opencv2/core/types.hpp"

#if defined(HAVE_EIGEN)
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

#include "opencv2/core/eigen.hpp"
#endif

namespace cv
{
namespace kinfu
{
/*!
 * \class BlockSparseMat
 * Naive implementation of Sparse Block Matrix
 */
template<typename _Tp, int blockM, int blockN>
struct BlockSparseMat
{
    struct Point2iHash
    {
        size_t operator()(const cv::Point2i& point) const noexcept
        {
            size_t seed                     = 0;
            constexpr uint32_t GOLDEN_RATIO = 0x9e3779b9;
            seed ^= std::hash<int>()(point.x) + GOLDEN_RATIO + (seed << 6) + (seed >> 2);
            seed ^= std::hash<int>()(point.y) + GOLDEN_RATIO + (seed << 6) + (seed >> 2);
            return seed;
        }
    };
    typedef Matx<_Tp, blockM, blockN> MatType;
    typedef std::unordered_map<Point2i, MatType, Point2iHash> IDtoBlockValueMap;

    BlockSparseMat(int _nBlocks) : nBlocks(_nBlocks), ijValue() {}

    MatType& refBlock(int i, int j)
    {
        Point2i p(i, j);
        auto it = ijValue.find(p);
        if (it == ijValue.end())
        {
            it = ijValue.insert({ p, Matx<_Tp, blockM, blockN>::zeros() }).first;
        }
        return it->second;
    }

    Mat diagonal()
    {
        // Diagonal max length is the number of columns in the sparse matrix
        int diagLength = blockN * nBlocks;
        cv::Mat diag   = cv::Mat::zeros(diagLength, 1, CV_32F);

        for (int i = 0; i < diagLength; i++)
        {
            diag.at<float>(i, 0) = refElem(i, i);
        }
        return diag;
    }

    float& refElem(int i, int j)
    {
        Point2i ib(i / blockM, j / blockN), iv(i % blockM, j % blockN);
        return refBlock(ib.x, ib.y)(iv.x, iv.y);
    }

#if defined(HAVE_EIGEN)
    Eigen::SparseMatrix<_Tp> toEigen() const
    {
        std::vector<Eigen::Triplet<double>> tripletList;
        tripletList.reserve(ijValue.size() * blockM * blockN);
        for (auto ijv : ijValue)
        {
            int xb = ijv.first.x, yb = ijv.first.y;
            MatType vblock = ijv.second;
            for (int i = 0; i < blockM; i++)
            {
                for (int j = 0; j < blockN; j++)
                {
                    float val = vblock(i, j);
                    if (abs(val) >= NON_ZERO_VAL_THRESHOLD)
                    {
                        tripletList.push_back(Eigen::Triplet<double>(blockM * xb + i, blockN * yb + j, val));
                    }
                }
            }
        }
        Eigen::SparseMatrix<_Tp> EigenMat(blockM * nBlocks, blockN * nBlocks);
        EigenMat.setFromTriplets(tripletList.begin(), tripletList.end());
        EigenMat.makeCompressed();

        return EigenMat;
    }
#endif
    size_t nonZeroBlocks() const { return ijValue.size(); }

    static constexpr float NON_ZERO_VAL_THRESHOLD = 0.0001f;
    int nBlocks;
    IDtoBlockValueMap ijValue;
};

//! Function to solve a sparse linear system of equations HX = B
//! Requires Eigen
static bool sparseSolve(const BlockSparseMat<float, 6, 6>& H, const Mat& B, Mat& X, Mat& predB)
{
    bool result = false;
#if defined(HAVE_EIGEN)
    Eigen::SparseMatrix<float> bigA = H.toEigen();
    Eigen::VectorXf bigB;
    cv2eigen(B, bigB);

    Eigen::SparseMatrix<float> bigAtranspose = bigA.transpose();
    if(!bigA.isApprox(bigAtranspose))
    {
        CV_Error(Error::StsBadArg, "H matrix is not symmetrical");
        return result;
    }

    Eigen::SimplicialLDLT<Eigen::SparseMatrix<float>> solver;

    solver.compute(bigA);
    if (solver.info() != Eigen::Success)
    {
        std::cout << "failed to eigen-decompose" << std::endl;
        result = false;
    }
    else
    {
        Eigen::VectorXf solutionX = solver.solve(bigB);
        Eigen::VectorXf predBEigen = bigA * solutionX;
        if (solver.info() != Eigen::Success)
        {
            std::cout << "failed to eigen-solve" << std::endl;
            result = false;
        }
        else
        {
            eigen2cv(solutionX, X);
            eigen2cv(predBEigen, predB);
            result = true;
        }
    }
#else
    std::cout << "no eigen library" << std::endl;
    CV_Error(Error::StsNotImplemented, "Eigen library required for matrix solve, dense solver is not implemented");
#endif
    return result;
}
}  // namespace kinfu
}  // namespace cv
