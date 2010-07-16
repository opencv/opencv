/*********************************************************************
* Software License Agreement (BSD License)
*
*  Copyright (c) 2009, Willow Garage, Inc.
*  All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
*
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above
*     copyright notice, this list of conditions and the following
*     disclaimer in the documentation and/or other materials provided
*     with the distribution.
*   * Neither the name of the Willow Garage nor the names of its
*     contributors may be used to endorse or promote products derived
*     from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*********************************************************************/

#include "precomp.hpp"
#include "flann/flann.hpp"

namespace cv
{

namespace flann {

::cvflann::Index* LinearIndexParams::createIndex(const Mat& dataset) const
{
	CV_Assert(dataset.type() == CV_32F);
	CV_Assert(dataset.isContinuous());

	// TODO: fix ::cvflann::Matrix class so it can be constructed with a const float*
	::cvflann::Matrix<float> mat(dataset.rows, dataset.cols, (float*)dataset.ptr<float>(0));

	return new ::cvflann::Index(mat, ::cvflann::LinearIndexParams());
}

::cvflann::Index* KDTreeIndexParams::createIndex(const Mat& dataset) const
{
	CV_Assert(dataset.type() == CV_32F);
	CV_Assert(dataset.isContinuous());

	// TODO: fix ::cvflann::Matrix class so it can be constructed with a const float*
	::cvflann::Matrix<float> mat(dataset.rows, dataset.cols, (float*)dataset.ptr<float>(0));

	return new ::cvflann::Index(mat, ::cvflann::KDTreeIndexParams(trees));
}

::cvflann::Index* KMeansIndexParams::createIndex(const Mat& dataset) const
{
	CV_Assert(dataset.type() == CV_32F);
	CV_Assert(dataset.isContinuous());

	// TODO: fix ::cvflann::Matrix class so it can be constructed with a const float*
	::cvflann::Matrix<float> mat(dataset.rows, dataset.cols, (float*)dataset.ptr<float>(0));

	return new ::cvflann::Index(mat, ::cvflann::KMeansIndexParams(branching,iterations, (::flann_centers_init_t)centers_init, cb_index));
}

::cvflann::Index* CompositeIndexParams::createIndex(const Mat& dataset) const
{
	CV_Assert(dataset.type() == CV_32F);
	CV_Assert(dataset.isContinuous());

	// TODO: fix ::cvflann::Matrix class so it can be constructed with a const float*
	::cvflann::Matrix<float> mat(dataset.rows, dataset.cols, (float*)dataset.ptr<float>(0));

	return new ::cvflann::Index(mat, ::cvflann::CompositeIndexParams(trees, branching, iterations, (::flann_centers_init_t)centers_init, cb_index));
}

::cvflann::Index* AutotunedIndexParams::createIndex(const Mat& dataset) const
{
	CV_Assert(dataset.type() == CV_32F);
	CV_Assert(dataset.isContinuous());

	// TODO: fix ::cvflann::Matrix class so it can be constructed with a const float*
	::cvflann::Matrix<float> mat(dataset.rows, dataset.cols, (float*)dataset.ptr<float>(0));

	return new ::cvflann::Index(mat, ::cvflann::AutotunedIndexParams(target_precision, build_weight, memory_weight, sample_fraction));
}

::cvflann::Index* SavedIndexParams::createIndex(const Mat& dataset) const
{
	CV_Assert(dataset.type() == CV_32F);
	CV_Assert(dataset.isContinuous());

	// TODO: fix ::cvflann::Matrix class so it can be constructed with a const float*
	::cvflann::Matrix<float> mat(dataset.rows, dataset.cols, (float*)dataset.ptr<float>(0));

	return new ::cvflann::Index(mat, ::cvflann::SavedIndexParams(filename));
}



Index::Index(const Mat& dataset, const IndexParams& params)
{
	nnIndex = params.createIndex(dataset);
}

Index::~Index()
{
	delete nnIndex;
}

void Index::knnSearch(const vector<float>& query, vector<int>& indices, vector<float>& dists, int knn, const SearchParams& searchParams)
{

	::cvflann::Matrix<float> m_query(1, (int)query.size(), (float*)&query[0]);
	::cvflann::Matrix<int> m_indices(1, (int)indices.size(), &indices[0]);
	::cvflann::Matrix<float> m_dists(1, (int)dists.size(), &dists[0]);

	nnIndex->knnSearch(m_query,m_indices,m_dists,knn,::cvflann::SearchParams(searchParams.checks));
}


void Index::knnSearch(const Mat& queries, Mat& indices, Mat& dists, int knn, const SearchParams& searchParams)
{

	CV_Assert(queries.type() == CV_32F);
	CV_Assert(queries.isContinuous());
	::cvflann::Matrix<float> m_queries(queries.rows, queries.cols, (float*)queries.ptr<float>(0));

	CV_Assert(indices.type() == CV_32S);
	CV_Assert(indices.isContinuous());
	::cvflann::Matrix<int> m_indices(indices.rows, indices.cols, (int*)indices.ptr<int>(0));

	CV_Assert(dists.type() == CV_32F);
	CV_Assert(dists.isContinuous());
	::cvflann::Matrix<float> m_dists(dists.rows, dists.cols, (float*)dists.ptr<float>(0));

	nnIndex->knnSearch(m_queries,m_indices,m_dists,knn,::cvflann::SearchParams(searchParams.checks));
}

int Index::radiusSearch(const vector<float>& query, vector<int>& indices, vector<float>& dists, float radius, const SearchParams& searchParams)
{
	::cvflann::Matrix<float> m_query(1, (int)query.size(), (float*)&query[0]);
	::cvflann::Matrix<int> m_indices(1, (int)indices.size(), &indices[0]);
	::cvflann::Matrix<float> m_dists(1, (int)dists.size(), &dists[0]);

	return nnIndex->radiusSearch(m_query,m_indices,m_dists,radius,::cvflann::SearchParams(searchParams.checks));
}


int Index::radiusSearch(const Mat& query, Mat& indices, Mat& dists, float radius, const SearchParams& searchParams)
{
	CV_Assert(query.type() == CV_32F);
	CV_Assert(query.isContinuous());
	::cvflann::Matrix<float> m_query(query.rows, query.cols, (float*)query.ptr<float>(0));

	CV_Assert(indices.type() == CV_32S);
	CV_Assert(indices.isContinuous());
	::cvflann::Matrix<int> m_indices(indices.rows, indices.cols, (int*)indices.ptr<int>(0));

	CV_Assert(dists.type() == CV_32F);
	CV_Assert(dists.isContinuous());
	::cvflann::Matrix<float> m_dists(dists.rows, dists.cols, (float*)dists.ptr<float>(0));

	return nnIndex->radiusSearch(m_query,m_indices,m_dists,radius,::cvflann::SearchParams(searchParams.checks));
}


void Index::save(string filename)
{
	nnIndex->save(filename);
}

int Index::size() const
{
	return nnIndex->size();
}

int Index::veclen() const
{
	return nnIndex->veclen();
}


int hierarchicalClustering(const Mat& features, Mat& centers, const KMeansIndexParams& params)
{
	CV_Assert(features.type() == CV_32F);
	CV_Assert(features.isContinuous());
	::cvflann::Matrix<float> m_features(features.rows, features.cols, (float*)features.ptr<float>(0));

	CV_Assert(features.type() == CV_32F);
	CV_Assert(features.isContinuous());
	::cvflann::Matrix<float> m_centers(centers.rows, centers.cols, (float*)centers.ptr<float>(0));

	return ::cvflann::hierarchicalClustering(m_features, m_centers, ::cvflann::KMeansIndexParams(params.branching, params.iterations,
			(::flann_centers_init_t)params.centers_init, params.cb_index));
}


}

}
