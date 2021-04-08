// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include <iostream>
#include <unordered_map>

#include "opencv2/core/base.hpp"
#include "opencv2/core/types.hpp"
#include "opencv2/core/utils/logger.hpp"

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
template<typename _Tp, size_t blockM, size_t blockN>
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

    BlockSparseMat(size_t _nBlocks) : nBlocks(_nBlocks), ijValue() {}

    void clear()
    {
        ijValue.clear();
    }

    inline MatType& refBlock(size_t i, size_t j)
    {
        Point2i p((int)i, (int)j);
        auto it = ijValue.find(p);
        if (it == ijValue.end())
        {
            it = ijValue.insert({ p, MatType::zeros() }).first;
        }
        return it->second;
    }

    inline _Tp& refElem(size_t i, size_t j)
    {
        Point2i ib((int)(i / blockM), (int)(j / blockN));
        Point2i iv((int)(i % blockM), (int)(j % blockN));
        return refBlock(ib.x, ib.y)(iv.x, iv.y);
    }

    inline MatType valBlock(size_t i, size_t j) const
    {
        Point2i p((int)i, (int)j);
        auto it = ijValue.find(p);
        if (it == ijValue.end())
            return MatType::zeros();
        else
            return it->second;
    }

    inline _Tp valElem(size_t i, size_t j) const
    {
        Point2i ib((int)(i / blockM), (int)(j / blockN));
        Point2i iv((int)(i % blockM), (int)(j % blockN));
        return valBlock(ib.x, ib.y)(iv.x, iv.y);
    }

    Mat diagonal() const
    {
        // Diagonal max length is the number of columns in the sparse matrix
        int diagLength = blockN * nBlocks;
        cv::Mat diag   = cv::Mat::zeros(diagLength, 1, cv::DataType<_Tp>::type);

        for (int i = 0; i < diagLength; i++)
        {
            diag.at<_Tp>(i, 0) = valElem(i, i);
        }
        return diag;
    }

#if defined(HAVE_EIGEN)
    Eigen::SparseMatrix<_Tp> toEigen() const
    {
        std::vector<Eigen::Triplet<_Tp>> tripletList;
        tripletList.reserve(ijValue.size() * blockM * blockN);
        for (const auto& ijv : ijValue)
        {
            int xb = ijv.first.x, yb = ijv.first.y;
            MatType vblock = ijv.second;
            for (size_t i = 0; i < blockM; i++)
            {
                for (size_t j = 0; j < blockN; j++)
                {
                    _Tp val = vblock((int)i, (int)j);
                    if (abs(val) >= NON_ZERO_VAL_THRESHOLD)
                    {
                        tripletList.push_back(Eigen::Triplet<_Tp>((int)(blockM * xb + i), (int)(blockN * yb + j), val));
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
    inline size_t nonZeroBlocks() const { return ijValue.size(); }

    BlockSparseMat<_Tp, blockM, blockN>& operator+=(const BlockSparseMat<_Tp, blockM, blockN>& other)
    {
        for (const auto& oijv : other.ijValue)
        {
            Point2i p = oijv.first;
            MatType vblock = oijv.second;
            this->refBlock(p.x, p.y) += vblock;
        }

        return *this;
    }

#if defined(HAVE_EIGEN)
    //! Function to solve a sparse linear system of equations HX = B
    //! Requires Eigen
    bool sparseSolve(InputArray B, OutputArray X, bool checkSymmetry = true, OutputArray predB = cv::noArray()) const
    {
        Eigen::SparseMatrix<_Tp> bigA = toEigen();
        Mat mb = B.getMat().t();
        Eigen::Matrix<_Tp, -1, 1> bigB;
        cv2eigen(mb, bigB);

        Eigen::SparseMatrix<_Tp> bigAtranspose = bigA.transpose();
        if(checkSymmetry && !bigA.isApprox(bigAtranspose))
        {
            CV_Error(Error::StsBadArg, "H matrix is not symmetrical");
            return false;
        }

        Eigen::SimplicialLDLT<Eigen::SparseMatrix<_Tp>> solver;

        solver.compute(bigA);
        if (solver.info() != Eigen::Success)
        {
            CV_LOG_INFO(NULL, "Failed to eigen-decompose");
            return false;
        }
        else
        {
            Eigen::Matrix<_Tp, -1, 1> solutionX = solver.solve(bigB);
            if (solver.info() != Eigen::Success)
            {
                CV_LOG_INFO(NULL, "Failed to eigen-solve");
                return false;
            }
            else
            {
                eigen2cv(solutionX, X);
                if (predB.needed())
                {
                    Eigen::Matrix<_Tp, -1, 1> predBEigen = bigA * solutionX;
                    eigen2cv(predBEigen, predB);
                }
                return true;
            }
        }
    }
#else
    bool sparseSolve(InputArray /*B*/, OutputArray /*X*/, bool /*checkSymmetry*/ = true, OutputArray /*predB*/ = cv::noArray()) const
    {
        CV_Error(Error::StsNotImplemented, "Eigen library required for matrix solve, dense solver is not implemented");
    }
#endif

    static constexpr _Tp NON_ZERO_VAL_THRESHOLD = _Tp(0.0001);
    size_t nBlocks;
    IDtoBlockValueMap ijValue;
};

}  // namespace kinfu
}  // namespace cv
