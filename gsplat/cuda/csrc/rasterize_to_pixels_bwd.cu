#include "bindings.h"
#include "helpers.cuh"
#include "types.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include "structs.h"


namespace gsplat {

namespace cg = cooperative_groups;
namespace pg = PerGaussian;
/****************************************************************************
 * Rasterization to Pixels Backward Pass
 ****************************************************************************/

template <uint32_t COLOR_DIM, typename S>
__global__ void rasterize_to_pixels_bwd_kernel(
    const uint32_t C,
    const uint32_t N,
    const uint32_t n_isects,
    const bool packed,
    // fwd inputs
    const vec2<S> *__restrict__ means2d, // [C, N, 2] or [nnz, 2]
    const vec3<S> *__restrict__ conics,  // [C, N, 3] or [nnz, 3]
    const S *__restrict__ colors,      // [C, N, COLOR_DIM] or [nnz, COLOR_DIM]
    const S *__restrict__ opacities,   // [C, N] or [nnz]
    const S *__restrict__ backgrounds, // [C, COLOR_DIM] or [nnz, COLOR_DIM]
    const bool *__restrict__ masks,    // [C, tile_height, tile_width]
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const int32_t *__restrict__ tile_offsets, // [C, tile_height, tile_width]
    const int32_t *__restrict__ flatten_ids,  // [n_isects]
    // fwd outputs
    const S *__restrict__ render_alphas,  // [C, image_height, image_width, 1]
    const int32_t *__restrict__ last_ids, // [C, image_height, image_width]
    // grad outputs
    const S *__restrict__ v_render_colors, // [C, image_height, image_width,
                                           // COLOR_DIM]
    const S *__restrict__ v_render_alphas, // [C, image_height, image_width, 1]
    // grad inputs
    vec2<S> *__restrict__ v_means2d_abs, // [C, N, 2] or [nnz, 2]
    vec2<S> *__restrict__ v_means2d,     // [C, N, 2] or [nnz, 2]
    vec3<S> *__restrict__ v_conics,      // [C, N, 3] or [nnz, 3]
    S *__restrict__ v_colors,   // [C, N, COLOR_DIM] or [nnz, COLOR_DIM]
    S *__restrict__ v_opacities, // [C, N] or [nnz]
    int B,
	const uint32_t* __restrict__ per_tile_bucket_offset,
	const uint32_t* __restrict__ bucket_to_tile,
	const float* __restrict__ sampled_T, const float* __restrict__ sampled_ar, 
    const int32_t* __restrict__ max_contrib, const float* __restrict__ pixel_colors
) {
    auto block = cg::this_thread_block();
	auto my_warp = cg::tiled_partition<32>(block);
	uint32_t global_bucket_idx = block.group_index().x * my_warp.meta_group_size() + my_warp.meta_group_rank();
	bool valid_bucket = global_bucket_idx < (uint32_t) B;
	if (!valid_bucket) return;

    bool valid_splat = false;

    int num_splats_in_tile, bucket_idx_in_tile;
	int splat_idx_in_tile,  splat_idx_global;

	uint32_t tile_id = bucket_to_tile[global_bucket_idx];
    int32_t range_start = tile_offsets[tile_id];
    uint32_t camera_id = tile_id / (tile_width * tile_height);
    int32_t range_end =
        (camera_id == C - 1) && (tile_id == tile_width * tile_height - 1)
            ? n_isects
            : tile_offsets[tile_id + 1];
	num_splats_in_tile = range_end - range_start;

	// What is the number of buckets before me? what is my offset?
    uint32_t real_tile_id = tile_id - camera_id * (tile_width * tile_height);
	uint32_t bbm = real_tile_id == 0 ? 0 : per_tile_bucket_offset[real_tile_id - 1];
	bucket_idx_in_tile = global_bucket_idx - bbm;
	splat_idx_in_tile  = bucket_idx_in_tile * 32 + my_warp.thread_rank();
	splat_idx_global   = range_start + splat_idx_in_tile;
	valid_splat = (splat_idx_in_tile < num_splats_in_tile);

	// if first gaussian in bucket is useless, then others are also useless
	if (bucket_idx_in_tile * 32 >= max_contrib[tile_id] || !valid_splat) {
		return;
	}

	// Load Gaussian properties into registers
	int gaussian_idx = 0;
	vec2<S> xy = {0.0f, 0.0f};
    vec3<S> conic = {0.0f, 0.0f, 0.0f};
    S opac = 0.0f ;
	float c[COLOR_DIM] = {0.0f};
	if (valid_splat) {
		gaussian_idx = flatten_ids[splat_idx_global];
		xy    = means2d[gaussian_idx];
        opac  = opacities[gaussian_idx];
		conic = conics[gaussian_idx];
		for (int ch = 0; ch < COLOR_DIM; ++ch)
			c[ch] = colors[gaussian_idx * COLOR_DIM + ch];
	}

    S v_rgb_local[COLOR_DIM] = {0.f};         //Register_dL_dcolors
    vec3<S> v_conic_local = {0.f, 0.f, 0.f};  //Register_dL_dconic2D
    vec2<S> v_xy_local = {0.f, 0.f};          //Register_dL_dmean2D
    vec2<S> v_xy_abs_local = {0.f, 0.f};      //Register_dL_abs_dmean2D
    S v_opacity_local = 0.f;                  //Register_dL_dopacity

	const uint32_t horizontal_blocks = (image_width + tile_size - 1) / tile_size;
	const uint2 tile = {real_tile_id % horizontal_blocks, real_tile_id / horizontal_blocks};
	const uint2 pix_min = {tile.x * tile_size, tile.y * tile_size};

	S T;
	S T_final;
	int32_t last_contributor;
	S ar[COLOR_DIM];
	S v_render_c[COLOR_DIM];
    S v_render_a;
	const S ddelx_dx = 0.5 * image_width;
	const S ddely_dy = 0.5 * image_height;
    const uint32_t block_size = tile_size * tile_size;

    // block.sync();
	// iterate over all pixels in the tile
	for (int i = 0; i < block_size + 31; ++i) { // 256 + 31 = 287
		// SHUFFLING
		// At this point, T already has my (1 - alpha) multiplied.
		// So pass this ready-made T value to next thread.
		T                 = my_warp.shfl_up(T, 1);
		T_final           = my_warp.shfl_up(T_final, 1);
		last_contributor  = my_warp.shfl_up(last_contributor, 1);
		for (int ch = 0; ch < COLOR_DIM; ++ch) {
			ar[ch]        = my_warp.shfl_up(ar[ch], 1);
			v_render_c[ch] = my_warp.shfl_up(v_render_c[ch], 1);
		}
        v_render_a = my_warp.shfl_up(v_render_a, 1);

		// which pixel index should this thread deal with?
		int idx = i - my_warp.thread_rank();
		const uint2    pix    = {pix_min.x + idx % tile_size, pix_min.y + idx / tile_size};
		const uint32_t pix_id = image_width * pix.y + pix.x;
		const float2   pixf   = {(float) pix.x + 0.5f, (float) pix.y + 0.5f};
		bool valid_pixel = pix.x < image_width && pix.y < image_height;

		// every 32nd thread should read the stored state from memory
		if (valid_splat && valid_pixel && my_warp.thread_rank() == 0 && idx < block_size) {
			T = sampled_T[global_bucket_idx * block_size + idx];
			T_final = 1.0f - render_alphas[pix_id];
			for (int ch = 0; ch < COLOR_DIM; ++ch)
				ar[ch] = -(pixel_colors[pix_id * COLOR_DIM + ch]) + sampled_ar[global_bucket_idx * block_size * COLOR_DIM + ch * block_size + idx];
				//ar[ch] = sampled_ar[global_bucket_idx * block_size * COLOR_DIM + ch * block_size + idx];
			last_contributor = last_ids[pix_id];
			for (int ch = 0; ch < COLOR_DIM; ++ch) {
				v_render_c[ch] = v_render_colors[pix_id * COLOR_DIM + ch];
			}
            v_render_a = v_render_alphas[pix_id];
		}

		// do work | IDX 必须大于0 小于BLOCK_SIZE !!
		if (valid_splat && valid_pixel && idx >= 0 && idx < block_size) {
			if (image_width <= pix.x || image_height <= pix.y) continue;

			if (splat_idx_in_tile >= last_contributor) continue;

			// compute blending values
			const vec2<S> delta = { xy.x - pixf.x, xy.y - pixf.y };
            float power = -0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) -
                                   conic.y * delta.x * delta.y;            
			if (power > 0.0f) continue;
			const float vis   = exp(power);
			const float alpha = min(0.99f, opac * vis);
			if (alpha < 1.0f / 255.0f) continue;

            float v_alpha =  0.0f;
			for (int ch = 0; ch < COLOR_DIM; ++ch) {
				v_rgb_local[ch] += T * alpha * v_render_c[ch];
				// v_alpha += (c[ch] * T - ar[ch] * T) * v_render_c[ch];
                // ar[ch] = alpha * c[ch] + (1.0f - alpha) * ar[ch];
				ar[ch] += T * alpha * c[ch];
                v_alpha += ((c[ch] * T) - (1.0f / (1.0f - alpha)) * (-ar[ch])) * v_render_c[ch];
			}
            v_alpha -= T_final / (1.0f - alpha) * v_render_a;

			T *= (1.0f - alpha);

            if (opac * vis <= 0.999f) {
                v_xy_local.x += - opac * vis * v_alpha * (conic.x * delta.x + conic.y * delta.y) * 0.5 * image_width;
                v_xy_local.y += - opac * vis * v_alpha * (conic.y * delta.x + conic.z * delta.y) * 0.5 * image_height;

                v_conic_local.x += - 0.5f * opac * vis * v_alpha * delta.x * delta.x;
                v_conic_local.y += - 1.0f * opac * vis * v_alpha * delta.x * delta.y;
                v_conic_local.z += - 0.5f * opac * vis * v_alpha * delta.y * delta.y;

                v_opacity_local += vis * v_alpha;
            }
		}
	}

    // finally add the gradients using atomics
    if (valid_splat) {
        int32_t g = gaussian_idx; // flatten index in [C * N] or [nnz]
        S *v_rgb_ptr = (S *)(v_colors) + COLOR_DIM * g;
        GSPLAT_PRAGMA_UNROLL
        for (uint32_t k = 0; k < COLOR_DIM; ++k) {
            gpuAtomicAdd(v_rgb_ptr + k, v_rgb_local[k]);
        }

        S *v_conic_ptr = (S *)(v_conics) + 3 * g;
        gpuAtomicAdd(v_conic_ptr, v_conic_local.x);
        gpuAtomicAdd(v_conic_ptr + 1, v_conic_local.y);
        gpuAtomicAdd(v_conic_ptr + 2, v_conic_local.z);

        S *v_xy_ptr = (S *)(v_means2d) + 2 * g;
        gpuAtomicAdd(v_xy_ptr,     v_xy_local.x);
        gpuAtomicAdd(v_xy_ptr + 1, v_xy_local.y);

        if (v_means2d_abs != nullptr) {
            S *v_xy_abs_ptr = (S *)(v_means2d_abs) + 2 * g;
            gpuAtomicAdd(v_xy_abs_ptr,     v_xy_abs_local.x);
            gpuAtomicAdd(v_xy_abs_ptr + 1, v_xy_abs_local.y);
        }

        gpuAtomicAdd(v_opacities + g, v_opacity_local);        
    }    
}

template <uint32_t CDIM>
std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
call_kernel_with_dim(
    // Gaussian parameters
    const torch::Tensor &means2d,                   // [C, N, 2] or [nnz, 2]
    const torch::Tensor &conics,                    // [C, N, 3] or [nnz, 3]
    const torch::Tensor &colors,                    // [C, N, 3] or [nnz, 3]
    const torch::Tensor &opacities,                 // [C, N] or [nnz]
    const at::optional<torch::Tensor> &backgrounds, // [C, 3]
    const at::optional<torch::Tensor> &masks, // [C, tile_height, tile_width]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // intersections
    const torch::Tensor &tile_offsets, // [C, tile_height, tile_width]
    const torch::Tensor &flatten_ids,  // [n_isects]
    // forward outputs
    const torch::Tensor &render_alphas, // [C, image_height, image_width, 1]
    const torch::Tensor &last_ids,      // [C, image_height, image_width]
    // gradients of outputs
    const torch::Tensor &v_render_colors, // [C, image_height, image_width, 3]
    const torch::Tensor &v_render_alphas, // [C, image_height, image_width, 1]
    // options
    bool absgrad,
	const int B,
	const torch::Tensor& imageBuffer,
	const torch::Tensor& sampleBuffer      
) {

    GSPLAT_DEVICE_GUARD(means2d);
    GSPLAT_CHECK_INPUT(means2d);
    GSPLAT_CHECK_INPUT(conics);
    GSPLAT_CHECK_INPUT(colors);
    GSPLAT_CHECK_INPUT(opacities);
    GSPLAT_CHECK_INPUT(tile_offsets);
    GSPLAT_CHECK_INPUT(flatten_ids);
    GSPLAT_CHECK_INPUT(render_alphas);
    GSPLAT_CHECK_INPUT(last_ids);
    GSPLAT_CHECK_INPUT(v_render_colors);
    GSPLAT_CHECK_INPUT(v_render_alphas);
    if (backgrounds.has_value()) {
        GSPLAT_CHECK_INPUT(backgrounds.value());
    }
    if (masks.has_value()) {
        GSPLAT_CHECK_INPUT(masks.value());
    }

    bool packed = means2d.dim() == 2;

    uint32_t C = tile_offsets.size(0);         // number of cameras
    uint32_t N = packed ? 0 : means2d.size(1); // number of gaussians
    uint32_t n_isects = flatten_ids.size(0);
    uint32_t COLOR_DIM = colors.size(-1);
    uint32_t tile_height = tile_offsets.size(1);
    uint32_t tile_width = tile_offsets.size(2);

    // Each block covers a tile on the image. In total there are
    // C * tile_height * tile_width blocks.
    // dim3 threads = {tile_size, tile_size, 1};
    // dim3 blocks = {C, tile_height, tile_width};
    const int THREADS = 32;

    torch::Tensor v_means2d = torch::zeros_like(means2d);
    torch::Tensor v_conics  = torch::zeros_like(conics);
    torch::Tensor v_colors  = torch::zeros_like(colors);
    torch::Tensor v_opacities = torch::zeros_like(opacities);
    torch::Tensor v_means2d_abs;
    if (absgrad) {
        v_means2d_abs = torch::zeros_like(means2d);
    }
    char* image_chunkptr  = reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr());
    char* sample_chunkptr = reinterpret_cast<char*>(sampleBuffer.contiguous().data_ptr());
    pg::ImageState<CDIM>  imgState    = pg::ImageState<CDIM>::fromChunk(image_chunkptr, image_width * image_height);
    pg::SampleState<CDIM> sampleState = pg::SampleState<CDIM>::fromChunk(sample_chunkptr, B);

    if (n_isects) {
        const uint32_t shared_mem =
            tile_size * tile_size *
            (sizeof(int32_t) + sizeof(vec3<float>) + sizeof(vec3<float>) +
             sizeof(float) * COLOR_DIM);
        at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

        if (cudaFuncSetAttribute(
                rasterize_to_pixels_bwd_kernel<CDIM, float>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                shared_mem
            ) != cudaSuccess) {
            AT_ERROR(
                "Failed to set maximum shared memory size (requested ",
                shared_mem,
                " bytes), try lowering tile_size."
            );
        }
        rasterize_to_pixels_bwd_kernel<CDIM, float>
            <<<((B*32) + THREADS - 1) / THREADS, THREADS, shared_mem, stream>>>(
                C,
                N,
                n_isects,
                packed,
                reinterpret_cast<vec2<float> *>(means2d.data_ptr<float>()),
                reinterpret_cast<vec3<float> *>(conics.data_ptr<float>()),
                colors.data_ptr<float>(),
                opacities.data_ptr<float>(),
                backgrounds.has_value() ? backgrounds.value().data_ptr<float>()
                                        : nullptr,
                masks.has_value() ? masks.value().data_ptr<bool>() : nullptr,
                image_width,
                image_height,
                tile_size,
                tile_width,
                tile_height,
                tile_offsets.data_ptr<int32_t>(),
                flatten_ids.data_ptr<int32_t>(),
                render_alphas.data_ptr<float>(),
                last_ids.data_ptr<int32_t>(),
                v_render_colors.data_ptr<float>(),
                v_render_alphas.data_ptr<float>(),
                absgrad ? reinterpret_cast<vec2<float> *>(
                              v_means2d_abs.data_ptr<float>()
                          )
                        : nullptr,
                reinterpret_cast<vec2<float> *>(v_means2d.data_ptr<float>()),
                reinterpret_cast<vec3<float> *>(v_conics.data_ptr<float>()),
                v_colors.data_ptr<float>(),
                v_opacities.data_ptr<float>(),
                B,
                imgState.bucket_offsets,
                sampleState.bucket_to_tile,
                sampleState.T,
                sampleState.ar,
                imgState.max_contrib,
                imgState.pixel_colors
            );
    }

    return std::make_tuple(
        v_means2d_abs, v_means2d, v_conics, v_colors, v_opacities
    );
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
rasterize_to_pixels_bwd_tensor(
    // Gaussian parameters
    const torch::Tensor &means2d,                   // [C, N, 2] or [nnz, 2]
    const torch::Tensor &conics,                    // [C, N, 3] or [nnz, 3]
    const torch::Tensor &colors,                    // [C, N, 3] or [nnz, 3]
    const torch::Tensor &opacities,                 // [C, N] or [nnz]
    const at::optional<torch::Tensor> &backgrounds, // [C, 3]
    const at::optional<torch::Tensor> &masks, // [C, tile_height, tile_width]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // intersections
    const torch::Tensor &tile_offsets, // [C, tile_height, tile_width]
    const torch::Tensor &flatten_ids,  // [n_isects]
    // forward outputs
    const torch::Tensor &render_alphas, // [C, image_height, image_width, 1]
    const torch::Tensor &last_ids,      // [C, image_height, image_width]
    // gradients of outputs
    const torch::Tensor &v_render_colors, // [C, image_height, image_width, 3]
    const torch::Tensor &v_render_alphas, // [C, image_height, image_width, 1]
    // options
    bool absgrad,
	const int B,
	const torch::Tensor& imageBuffer,
	const torch::Tensor& sampleBuffer  
) {

    GSPLAT_CHECK_INPUT(colors);
    uint32_t COLOR_DIM = colors.size(-1);

#define __GS__CALL_(N)                                                         \
    case N:                                                                    \
        return call_kernel_with_dim<N>(                                        \
            means2d,                                                           \
            conics,                                                            \
            colors,                                                            \
            opacities,                                                         \
            backgrounds,                                                       \
            masks,                                                             \
            image_width,                                                       \
            image_height,                                                      \
            tile_size,                                                         \
            tile_offsets,                                                      \
            flatten_ids,                                                       \
            render_alphas,                                                     \
            last_ids,                                                          \
            v_render_colors,                                                   \
            v_render_alphas,                                                   \
            absgrad,                                                           \
            B,                                                                 \
            imageBuffer,                                                       \
            sampleBuffer                                                       \
        );

    switch (COLOR_DIM) {
        __GS__CALL_(1)
        __GS__CALL_(2)
        __GS__CALL_(3)
        __GS__CALL_(4)
        __GS__CALL_(5)
        __GS__CALL_(8)
        __GS__CALL_(9)
        __GS__CALL_(16)
        __GS__CALL_(17)
        __GS__CALL_(32)
        __GS__CALL_(33)
        __GS__CALL_(64)
        __GS__CALL_(65)
        __GS__CALL_(128)
        __GS__CALL_(129)
        __GS__CALL_(256)
        __GS__CALL_(257)
        __GS__CALL_(512)
        __GS__CALL_(513)
    default:
        AT_ERROR("Unsupported number of channels: ", COLOR_DIM);
    }
}

} // namespace gsplat