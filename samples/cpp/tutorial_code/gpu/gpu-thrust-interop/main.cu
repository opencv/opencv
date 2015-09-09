#include "Thrust_interop.hpp"

#include <thrust/transform.h>
#include <thrust/random.h>
#include <thrust/sort.h>
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

template<typename T> struct pred_eq
{
	T value;
	int channel;
	__host__ __device__
		pred_eq(T value_, int channel_ = 0) :value(value_), channel(channel_){}
	__host__ __device__
		bool operator()(const T val) const
	{
		return val == value;
	}
	template<int N>
	__host__ __device__ bool operator()(const cv::Vec<T, N>& val)
	{
		if (channel < N)
			return val.val[channel] == value;
		return false;
	}
};


int main(void)
{
	// Generate a 2 channel row matrix with 100 elements.  Set the first channel to be the element index, and the second to be a randomly
	// generated value.  Sort by the randomly generated value while maintaining index association.
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

	// Randomly fill a row matrix with 100 elements between -1 and 1
	{
		cv::cuda::GpuMat d_value(1, 100, CV_32F);
		auto valueBegin = GpuMatBeginItr<float>(d_value);
		auto valueEnd = GpuMatEndItr<float>(d_value);
		thrust::transform(thrust::make_counting_iterator(0), thrust::make_counting_iterator(d_value.cols), valueBegin, prg(-1, 1));

		cv::Mat h_value(d_value);
	}

	// OpenCV has count non zero, but what if you want to count a specific value?
	{
		cv::cuda::GpuMat d_value(1, 100, CV_32S);
		d_value.setTo(cv::Scalar(0));
		d_value.colRange(10, 50).setTo(cv::Scalar(15));
		auto count = thrust::count(GpuMatBeginItr<int>(d_value), GpuMatEndItr<int>(d_value), 15);
		std::cout << count << std::endl;
	}

	


	return 0;
}
