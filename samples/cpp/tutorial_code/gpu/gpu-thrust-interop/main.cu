#include "Thrust_interop.hpp"
#include <opencv2/core/cuda_stream_accessor.hpp>

#include <thrust/transform.h>
#include <thrust/random.h>
#include <thrust/sort.h>
#include <thrust/system/cuda/execution_policy.h>
//! [prg]
struct prg
{
	float a, b;

	__host__ __device__
		prg(float _a = 0.f, float _b = 1.f) : a(_a), b(_b) {};

	__host__ __device__
		float operator()(const unsigned int n) const
	{
		thrust::default_random_engine rng;
		thrust::uniform_real_distribution<float> dist(a, b);
		rng.discard(n);

		return dist(rng);
	}
};
//! [prg]


//! [pred_greater]
template<typename T> struct pred_greater
{
	T value;
	__host__ __device__ pred_greater(T value_) : value(value_){}
	__host__ __device__ bool operator()(const T& val) const
	{
		return val > value;
	}
};
//! [pred_greater]


int main(void)
{
	// Generate a 2 channel row matrix with 100 elements.  Set the first channel to be the element index, and the second to be a randomly
	// generated value.  Sort by the randomly generated value while maintaining index association.
	//! [sort]
	{
		cv::cuda::GpuMat d_idx(1, 100, CV_32SC2);

		auto keyBegin = GpuMatBeginItr<int>(d_idx, 1);
		auto keyEnd = GpuMatEndItr<int>(d_idx, 1);

		auto idxBegin = GpuMatBeginItr<int>(d_idx, 0);
		auto idxEnd = GpuMatEndItr<int>(d_idx, 0);

		thrust::sequence(idxBegin, idxEnd);
		thrust::transform(idxBegin, idxEnd, keyBegin, prg(0, 10));
		thrust::sort_by_key(keyBegin, keyEnd, idxBegin);

		cv::Mat h_idx(d_idx);
	}
	//! [sort]

	// Randomly fill a row matrix with 100 elements between -1 and 1
	//! [random]
	{
		cv::cuda::GpuMat d_value(1, 100, CV_32F);
		auto valueBegin = GpuMatBeginItr<float>(d_value);
		auto valueEnd = GpuMatEndItr<float>(d_value);
		thrust::transform(thrust::make_counting_iterator(0), thrust::make_counting_iterator(d_value.cols), valueBegin, prg(-1, 1));

		cv::Mat h_value(d_value);
	}
	//! [random]

	// OpenCV has count non zero, but what if you want to count a specific value?
	//! [count_value]
	{
		cv::cuda::GpuMat d_value(1, 100, CV_32S);
		d_value.setTo(cv::Scalar(0));
		d_value.colRange(10, 50).setTo(cv::Scalar(15));
		auto count = thrust::count(GpuMatBeginItr<int>(d_value), GpuMatEndItr<int>(d_value), 15);
		std::cout << count << std::endl;
	}
	//! [count_value]

	// Randomly fill an array then copy only values greater than 0.  Perform these tasks on a stream.
	//! [copy_greater]
	{
		cv::cuda::GpuMat d_value(1, 100, CV_32F);
		auto valueBegin = GpuMatBeginItr<float>(d_value);
		auto valueEnd = GpuMatEndItr<float>(d_value);
		cv::cuda::Stream stream;
		//! [random_gen_stream]
		thrust::transform(thrust::system::cuda::par.on(cv::cuda::StreamAccessor::getStream(stream)), thrust::make_counting_iterator(0), thrust::make_counting_iterator(d_value.cols), valueBegin, prg(-1, 1));
		//! [random_gen_stream]
		int count = thrust::count_if(thrust::system::cuda::par.on(cv::cuda::StreamAccessor::getStream(stream)), valueBegin, valueEnd, pred_greater<float>(0.0));
		cv::cuda::GpuMat d_valueGreater(1, count, CV_32F);
		thrust::copy_if(thrust::system::cuda::par.on(cv::cuda::StreamAccessor::getStream(stream)), valueBegin, valueEnd, GpuMatBeginItr<float>(d_valueGreater), pred_greater<float>(0.0));
		cv::Mat h_greater(d_valueGreater);
	}
	//! [copy_greater]
	
	return 0;
}
