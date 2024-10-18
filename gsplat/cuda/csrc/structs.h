#ifndef GSPLAT_STRUCTS_H_INCLUDED
#define GSPLAT_STRUCTS_H_INCLUDED

#include <iostream>
#include <vector>
#include <cuda_runtime_api.h>

namespace PerGaussian
{
	template <typename T>
	static void obtain(char*& chunk, T*& ptr, std::size_t count, std::size_t alignment)
	{
		std::size_t offset = (reinterpret_cast<std::uintptr_t>(chunk) + alignment - 1) & ~(alignment - 1);
		ptr = reinterpret_cast<T*>(offset);
		chunk = reinterpret_cast<char*>(ptr + count);
	}

    inline std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
        auto lambda = [&t](size_t N) {
            t.resize_({(long long)N});
            return reinterpret_cast<char*>(t.contiguous().data_ptr());
        };
        return lambda;
    }

    template <uint32_t CDIM>
	struct ImageState
	{
		uint32_t *bucket_count;
		uint32_t *bucket_offsets;
		size_t bucket_count_scan_size;
		char* bucket_count_scanning_space;
		float* pixel_colors;
		int32_t* max_contrib;

		size_t scan_size;
		uint2* ranges;
		uint32_t* n_contrib;
		float* accum_alpha;
		char* contrib_scan;

        static ImageState<CDIM> fromChunk(char*& chunk, size_t N);
	};

    template <uint32_t CDIM>
	struct SampleState
	{
		uint32_t *bucket_to_tile;
		float *T;
		float *ar;

        static SampleState<CDIM> fromChunk(char*& chunk, size_t C);
	};

	template<typename T> 
	size_t required(size_t P)
	{
		char* size = nullptr;
		T::fromChunk(size, P);
		return ((size_t)size) + 128;
	}
};

#define CHECK_CUDA(A, debug) \
A; if(debug) { \
auto ret = cudaDeviceSynchronize(); \
if (ret != cudaSuccess) { \
std::cerr << "\n[CUDA ERROR] in " << __FILE__ << "\nLine " << __LINE__ << ": " << cudaGetErrorString(ret); \
throw std::runtime_error(cudaGetErrorString(ret)); \
} \
}

#endif