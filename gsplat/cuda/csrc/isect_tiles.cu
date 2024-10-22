#include "bindings.h"
#include "helpers.cuh"
#include "types.cuh"
#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include <cuda_runtime.h>

namespace gsplat {

namespace cg = cooperative_groups;

/****************************************************************************
 * Gaussian Tile Intersection
 ****************************************************************************/


__device__ inline float evaluate_opacity_factor(const float dx, const float dy, const vec4<float> co) {
	return 0.5f * (co.x * dx * dx + co.z * dy * dy) + co.y * dx * dy;
}

template<uint32_t PATCH_WIDTH, uint32_t PATCH_HEIGHT, typename T>
__device__ inline float max_contrib_power_rect_gaussian_float(
	const vec4<float> co,
	const vec2<T> mean_t,
	const glm::vec2 rect_min,
	const glm::vec2 rect_max,
	glm::vec2& max_pos
)
{
    const vec2<float> mean = static_cast<vec2<float>>(mean_t);
	const float x_min_diff = rect_min.x - mean.x;
	const float x_left = x_min_diff > 0.0f;
	// const float x_left = mean.x < rect_min.x;
	const float not_in_x_range = (mean.x < rect_min.x) + (mean.x > rect_max.x); // 0 or 1

	const float y_min_diff = rect_min.y - mean.y;
	const float y_above =  y_min_diff > 0.0f;
	// const float y_above = mean.y < rect_min.y;
	const float not_in_y_range = (mean.y < rect_min.y) + (mean.y > rect_max.y); // 0 or 1

	max_pos = {mean.x, mean.y};
	float max_contrib_power = 0.0f;

    // 初始化 ttx 和 tty
    float ttx = 0.0f;
    float tty = 0.0f;

    float px1 = 0.0f, py1 = 0.0f;
    float diffx = 0.0f, diffy = 0.0f;
    float dx = 0.0f, dy = 0.0f;
    float rcp_dxdxcox = 0.0f, rcp_dydycoz = 0.0f;
    float2 max_pos_diff = {0.0f, 0.0f};

	if ((not_in_y_range + not_in_x_range) > 0.0f) //OUTSIDE
	{
		px1 = x_left  > 0.0f ? rect_min.x : rect_max.x; //中心离tile最近的边
		py1 = y_above > 0.0f ? rect_min.y : rect_max.y;
		diffx = mean.x - px1; //中心离tile的距离
		diffy = mean.y - py1; //中心离tile的距离

		//copysign 是一个标准的数学函数，用于将第二个参数的符号（正或负）复制到第一个参数上，而保持第一个参数的绝对值不变。
		dx = copysign(float(PATCH_WIDTH),  x_min_diff);  //PATCH_WIDTH  左边左15 左边右-15
		dy = copysign(float(PATCH_HEIGHT), y_min_diff); //PATCH_HEIGHT  上边下15 上边上-15
		//__frcp_rn 是一个内置的设备函数，用于计算浮点数的倒数（reciprocal），并且结果会按照最近偶数（round-to-nearest-even）的模式进行舍入
		rcp_dxdxcox = __frcp_rn(PATCH_WIDTH  * PATCH_WIDTH  * co.x); // = 1.0 / (dx*dx*co.x)
		rcp_dydycoz = __frcp_rn(PATCH_HEIGHT * PATCH_HEIGHT * co.z); // = 1.0 / (dy*dy*co.z)
		//__saturatef 是一个内置的设备函数，用于将浮点数的值限制（或“饱和”）在 0.0 到 1.0 的范围内。
		ttx = not_in_y_range * __saturatef((dx*co.x*(mean.x - px1) + dx*co.y*(mean.y - py1)) * rcp_dxdxcox);
		tty = not_in_x_range * __saturatef((dy*co.y*(mean.x - px1) + dy*co.z*(mean.y - py1)) * rcp_dydycoz);

		max_pos = {px1 + ttx * dx, py1 + tty * dy};
		max_pos_diff.x = mean.x - max_pos.x;
		max_pos_diff.y = mean.y - max_pos.y;
		max_contrib_power = evaluate_opacity_factor(max_pos_diff.x, max_pos_diff.y, co);
		//evaluate_opacity_factor 0.5f * (co.x * max_pos_diff.x**2 + co.z * max_pos_diff.y**2) + co.y * max_pos_diff.x * max_pos_diff.y;
	}

	return max_contrib_power;
}

template <typename T>
__global__ void isect_tiles(
    // if the data is [C, N, ...] or [nnz, ...] (packed)
    const bool packed,
    // parallelize over C * N, only used if packed is False
    const uint32_t C,
    const uint32_t N,
    // parallelize over nnz, only used if packed is True
    const uint32_t nnz,
    const int64_t *__restrict__ camera_ids,   // [nnz] optional
    const int64_t *__restrict__ gaussian_ids, // [nnz] optional
    // data
    const T *__restrict__ means2d,                   // [C, N, 2] or [nnz, 2]
    const int32_t *__restrict__ radii,               // [C, N] or [nnz]
    const T *__restrict__ depths,                    // [C, N] or [nnz]
    const int64_t *__restrict__ cum_tiles_per_gauss, // [C, N] or [nnz]
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const uint32_t tile_n_bits,
    int32_t *__restrict__ tiles_per_gauss, // [C, N] or [nnz]
    int64_t *__restrict__ isect_ids,       // [n_isects]
    int32_t *__restrict__ flatten_ids,      // [n_isects]
    const float *__restrict__ conics,                   // [C, N, 3] or [nnz, 2]
    const float *__restrict__ opacities               // [C, N] or [nnz]

) {
    // For now we'll upcast float16 and bfloat16 to float32
    using OpT = typename OpType<T>::type;

    // parallelize over C * N.
    uint32_t idx = cg::this_grid().thread_rank();
    bool first_pass = cum_tiles_per_gauss == nullptr;
    if (idx >= (packed ? nnz : C * N)) {
        return;
    }

    const OpT radius = radii[idx];
    if (radius <= 0) {
        if (first_pass) {
            tiles_per_gauss[idx] = 0;
        }
        return;
    }

    vec2<OpT> mean2d = glm::make_vec2(means2d + 2 * idx);

    OpT tile_radius = radius / static_cast<OpT>(tile_size);
    OpT tile_x = mean2d.x / static_cast<OpT>(tile_size);
    OpT tile_y = mean2d.y / static_cast<OpT>(tile_size);

    // tile_min is inclusive, tile_max is exclusive
    uint2 tile_min, tile_max;
    tile_min.x = min(max(0, (uint32_t)floor(tile_x - tile_radius)), tile_width);
    tile_min.y =
        min(max(0, (uint32_t)floor(tile_y - tile_radius)), tile_height);
    tile_max.x = min(max(0, (uint32_t)ceil(tile_x + tile_radius)), tile_width);
    tile_max.y = min(max(0, (uint32_t)ceil(tile_y + tile_radius)), tile_height);

    if (first_pass) {
        // first pass only writes out tiles_per_gauss
        tiles_per_gauss[idx] = static_cast<int32_t>(
            (tile_max.y - tile_min.y) * (tile_max.x - tile_min.x)
        );
        return;
    }

    int64_t cid; // camera id
    if (packed) {
        // parallelize over nnz
        cid = camera_ids[idx];
        // gid = gaussian_ids[idx];
    } else {
        // parallelize over C * N
        cid = idx / N;
        // gid = idx % N;
    }
    const int64_t cid_enc = cid << (32 + tile_n_bits);

    int64_t depth_id_enc = (int64_t) * (int32_t *)&(depths[idx]);
    int64_t cur_idx = (idx == 0) ? 0 : cum_tiles_per_gauss[idx - 1];
    int64_t offset_idx = cum_tiles_per_gauss[idx];

    vec4<float> co = {0.0f, 0.0f, 0.0f, 0.0f};
    co.x = conics[3 * idx];
    co.y = conics[3 * idx + 1];
    co.z = conics[3 * idx + 2];
    co.w = opacities[idx];
    const float opacity_threshold = 1.0f / 255.0f;
    const float opacity_factor_threshold = logf(co.w / opacity_threshold);

    for (int32_t i = tile_min.y; i < tile_max.y; ++i) {
        for (int32_t j = tile_min.x; j < tile_max.x; ++j) {
            int64_t tile_id = i * tile_width + j;

            const glm::vec2 _min(j * 16, i * 16);
            const glm::vec2 _max((j + 1) * 16 - 1, (i + 1) * 16 - 1);

            glm::vec2 max_pos;
            float max_opac_factor = 0.0f;
            max_opac_factor = max_contrib_power_rect_gaussian_float<15, 15, OpT>(co, mean2d, _min, _max, max_pos);

            if (max_opac_factor <= opacity_factor_threshold) {
                isect_ids[cur_idx] = cid_enc | (tile_id << 32) | depth_id_enc;
                flatten_ids[cur_idx] = static_cast<int32_t>(idx);
                ++cur_idx;
           }
        }
    }

    for (; cur_idx < offset_idx; ++cur_idx) {
        int64_t tile_key = (int64_t) pow(2, tile_n_bits) - 1;
        tile_key <<= 32;
        int64_t cam_key = (int64_t) 0;
        cam_key <<= (32 + tile_n_bits);
        const int64_t depth = (int64_t) 0xFFFFFFFF;
        int64_t key = cam_key | tile_key | depth;
        isect_ids[cur_idx] = key;
        flatten_ids[cur_idx] = (int32_t) INT32_MAX;
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> isect_tiles_tensor(
    const torch::Tensor &means2d,                    // [C, N, 2] or [nnz, 2]
    const torch::Tensor &radii,                      // [C, N] or [nnz]
    const torch::Tensor &depths,                     // [C, N] or [nnz]
    const at::optional<torch::Tensor> &camera_ids,   // [nnz]
    const at::optional<torch::Tensor> &gaussian_ids, // [nnz]
    const uint32_t C,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const bool sort,
    const bool double_buffer,
    const torch::Tensor &conics,                    // [C, N, 3] or [nnz, 3]
    const torch::Tensor &opacities                // [C, N] or [nnz]       
) {
    GSPLAT_DEVICE_GUARD(means2d);
    GSPLAT_CHECK_INPUT(means2d);
    GSPLAT_CHECK_INPUT(radii);
    GSPLAT_CHECK_INPUT(depths);
    GSPLAT_CHECK_INPUT(conics);
    GSPLAT_CHECK_INPUT(opacities);    
    if (camera_ids.has_value()) {
        GSPLAT_CHECK_INPUT(camera_ids.value());
    }
    if (gaussian_ids.has_value()) {
        GSPLAT_CHECK_INPUT(gaussian_ids.value());
    }
    bool packed = means2d.dim() == 2;

    uint32_t N = 0, nnz = 0, total_elems = 0;
    int64_t *camera_ids_ptr = nullptr;
    int64_t *gaussian_ids_ptr = nullptr;
    if (packed) {
        nnz = means2d.size(0);
        total_elems = nnz;
        TORCH_CHECK(
            camera_ids.has_value() && gaussian_ids.has_value(),
            "When packed is set, camera_ids and gaussian_ids must be provided."
        );
        camera_ids_ptr = camera_ids.value().data_ptr<int64_t>();
        gaussian_ids_ptr = gaussian_ids.value().data_ptr<int64_t>();
    } else {
        N = means2d.size(1); // number of gaussians
        total_elems = C * N;
    }

    uint32_t n_tiles = tile_width * tile_height;
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    // the number of bits needed to encode the camera id and tile id
    // Note: std::bit_width requires C++20
    // uint32_t tile_n_bits = std::bit_width(n_tiles);
    // uint32_t cam_n_bits = std::bit_width(C);
    uint32_t tile_n_bits = (uint32_t)floor(log2(n_tiles)) + 1;
    uint32_t cam_n_bits = (uint32_t)floor(log2(C)) + 1;
    // the first 32 bits are used for the camera id and tile id altogether, so
    // check if we have enough bits for them.
    assert(tile_n_bits + cam_n_bits <= 32);

    // first pass: compute number of tiles per gaussian
    torch::Tensor tiles_per_gauss =
        torch::empty_like(depths, depths.options().dtype(torch::kInt32));

    int64_t n_isects;
    torch::Tensor cum_tiles_per_gauss;
    if (total_elems) {
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            means2d.scalar_type(),
            "isect_tiles_total_elems",
            [&]() {
                isect_tiles<<<
                    (total_elems + GSPLAT_N_THREADS - 1) / GSPLAT_N_THREADS,
                    GSPLAT_N_THREADS,
                    0,
                    stream>>>(
                    packed,
                    C,
                    N,
                    nnz,
                    camera_ids_ptr,
                    gaussian_ids_ptr,
                    reinterpret_cast<scalar_t *>(means2d.data_ptr<scalar_t>()),
                    radii.data_ptr<int32_t>(),
                    depths.data_ptr<scalar_t>(),
                    nullptr,
                    tile_size,
                    tile_width,
                    tile_height,
                    tile_n_bits,
                    tiles_per_gauss.data_ptr<int32_t>(),
                    nullptr,
                    nullptr,
                    conics.data_ptr<float>(),
                    opacities.data_ptr<float>()                   
                );
            }
        );
        cum_tiles_per_gauss = torch::cumsum(tiles_per_gauss.view({-1}), 0);
        n_isects = cum_tiles_per_gauss[-1].item<int64_t>();
    } else {
        n_isects = 0;
    }

    // second pass: compute isect_ids and flatten_ids as a packed tensor
    torch::Tensor isect_ids =
        torch::empty({n_isects}, depths.options().dtype(torch::kInt64));
    torch::Tensor flatten_ids =
        torch::empty({n_isects}, depths.options().dtype(torch::kInt32));
    if (n_isects) {
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            means2d.scalar_type(),
            "isect_tiles_n_isects",
            [&]() {
                isect_tiles<<<
                    (total_elems + GSPLAT_N_THREADS - 1) / GSPLAT_N_THREADS,
                    GSPLAT_N_THREADS,
                    0,
                    stream>>>(
                    packed,
                    C,
                    N,
                    nnz,
                    camera_ids_ptr,
                    gaussian_ids_ptr,
                    reinterpret_cast<scalar_t *>(means2d.data_ptr<scalar_t>()),
                    radii.data_ptr<int32_t>(),
                    depths.data_ptr<scalar_t>(),
                    cum_tiles_per_gauss.data_ptr<int64_t>(),
                    tile_size,
                    tile_width,
                    tile_height,
                    tile_n_bits,
                    nullptr,
                    isect_ids.data_ptr<int64_t>(),
                    flatten_ids.data_ptr<int32_t>(),
                    conics.data_ptr<float>(),
                    opacities.data_ptr<float>()                    
                );
            }
        );
    }

    // optionally sort the Gaussians by isect_ids
    if (n_isects && sort) {
        torch::Tensor isect_ids_sorted = torch::empty_like(isect_ids);
        torch::Tensor flatten_ids_sorted = torch::empty_like(flatten_ids);

        // https://nvidia.github.io/cccl/cub/api/structcub_1_1DeviceRadixSort.html
        // DoubleBuffer reduce the auxiliary memory usage from O(N+P) to O(P)
        if (double_buffer) {
            // Create a set of DoubleBuffers to wrap pairs of device pointers
            cub::DoubleBuffer<int64_t> d_keys(
                isect_ids.data_ptr<int64_t>(),
                isect_ids_sorted.data_ptr<int64_t>()
            );
            cub::DoubleBuffer<int32_t> d_values(
                flatten_ids.data_ptr<int32_t>(),
                flatten_ids_sorted.data_ptr<int32_t>()
            );
            GSPLAT_CUB_WRAPPER(
                cub::DeviceRadixSort::SortPairs,
                d_keys,
                d_values,
                n_isects,
                0,
                32 + tile_n_bits + cam_n_bits,
                stream
            );
            switch (d_keys.selector) {
            case 0: // sorted items are stored in isect_ids
                isect_ids_sorted = isect_ids;
                break;
            case 1: // sorted items are stored in isect_ids_sorted
                break;
            }
            switch (d_values.selector) {
            case 0: // sorted items are stored in flatten_ids
                flatten_ids_sorted = flatten_ids;
                break;
            case 1: // sorted items are stored in flatten_ids_sorted
                break;
            }
            // printf("DoubleBuffer d_keys selector: %d\n", d_keys.selector);
            // printf("DoubleBuffer d_values selector: %d\n",
            // d_values.selector);
        } else {
            GSPLAT_CUB_WRAPPER(
                cub::DeviceRadixSort::SortPairs,
                isect_ids.data_ptr<int64_t>(),
                isect_ids_sorted.data_ptr<int64_t>(),
                flatten_ids.data_ptr<int32_t>(),
                flatten_ids_sorted.data_ptr<int32_t>(),
                n_isects,
                0,
                32 + tile_n_bits + cam_n_bits,
                stream
            );
        }
        return std::make_tuple(
            tiles_per_gauss, isect_ids_sorted, flatten_ids_sorted
        );
    } else {
        return std::make_tuple(tiles_per_gauss, isect_ids, flatten_ids);
    }
}

__global__ void isect_offset_encode(
    const uint32_t n_isects,
    const int64_t *__restrict__ isect_ids,
    const uint32_t C,
    const uint32_t n_tiles,
    const uint32_t tile_n_bits,
    int32_t *__restrict__ offsets, // [C, n_tiles]
    uint32_t *__restrict__ valid_isects
) {
    // e.g., ids: [1, 1, 1, 3, 3], n_tiles = 6
    // counts: [0, 3, 0, 2, 0, 0]
    // cumsum: [0, 3, 3, 5, 5, 5]
    // offsets: [0, 0, 3, 3, 5, 5]
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= n_isects)
        return;

    int64_t isect_id_curr = isect_ids[idx] >> 32;
    int64_t cid_curr = isect_id_curr >> tile_n_bits;
    int64_t tid_curr = isect_id_curr & ((1 << tile_n_bits) - 1);
    int64_t id_curr = cid_curr * n_tiles + tid_curr;

    if (idx == 0) {
        // write out the offsets until the first valid tile (inclusive)
        for (uint32_t i = 0; i < id_curr + 1; ++i)
            offsets[i] = static_cast<int32_t>(0);
        return;
    }

    bool valid_tile_curr = (isect_ids[idx] & 0xFFFFFFFF) != (int64_t)0xFFFFFFFF;
    bool valid_tile_prev = (isect_ids[idx - 1] & 0xFFFFFFFF) != (int64_t)0xFFFFFFFF;

    int64_t isect_id_prev, cid_prev, tid_prev, id_prev;
    if (!valid_tile_curr && valid_tile_prev) {
        isect_id_prev = isect_ids[idx - 1] >> 32; // shift out the depth
        cid_prev = isect_id_prev >> tile_n_bits;
        tid_prev = isect_id_prev & ((1 << tile_n_bits) - 1);
        id_prev  = cid_prev * n_tiles + tid_prev;
        for (uint32_t i = id_prev + 1; i < C * n_tiles; ++i)
            offsets[i] = static_cast<int32_t>(idx);  
        *valid_isects = static_cast<uint32_t>(idx);              
    }

    if (!valid_tile_curr) return;

    if (idx == n_isects - 1) {
        // write out the rest of the offsets
        for (uint32_t i = id_curr + 1; i < C * n_tiles; ++i)
            offsets[i] = static_cast<int32_t>(n_isects);
    }

    if (idx > 0) {
        isect_id_prev = isect_ids[idx - 1] >> 32; // shift out the depth
        if (isect_id_prev == isect_id_curr) return;

        cid_prev = isect_id_prev >> tile_n_bits;
        tid_prev = isect_id_prev & ((1 << tile_n_bits) - 1);
        id_prev = cid_prev * n_tiles + tid_prev;
        for (uint32_t i = id_prev + 1; i < id_curr + 1; ++i)
            offsets[i] = static_cast<int32_t>(idx);
    }


    // if (idx == 0) {
        // write out the offsets until the first valid tile (inclusive)
    //     for (uint32_t i = 0; i < id_curr + 1; ++i)
    //         offsets[i] = static_cast<int32_t>(idx);
    // }
    // if (idx == n_isects - 1) {
    //     // write out the rest of the offsets
    //     for (uint32_t i = id_curr + 1; i < C * n_tiles; ++i)
    //         offsets[i] = static_cast<int32_t>(n_isects);
    // }

    // if (idx > 0) {
    //     // visit the current and previous isect_id and check if the (cid,
    //     // tile_id) pair changes.
    //     int64_t isect_id_prev = isect_ids[idx - 1] >> 32; // shift out the depth
    //     if (isect_id_prev == isect_id_curr)
    //         return;

    //     // write out the offsets between the previous and current tiles
    //     int64_t cid_prev = isect_id_prev >> tile_n_bits;
    //     int64_t tid_prev = isect_id_prev & ((1 << tile_n_bits) - 1);
    //     int64_t id_prev = cid_prev * n_tiles + tid_prev;
    //     for (uint32_t i = id_prev + 1; i < id_curr + 1; ++i)
    //         offsets[i] = static_cast<int32_t>(idx);
    // }

}

std::tuple<torch::Tensor, uint32_t> isect_offset_encode_tensor(
    const torch::Tensor &isect_ids, // [n_isects]
    const uint32_t C,
    const uint32_t tile_width,
    const uint32_t tile_height
) {
    GSPLAT_DEVICE_GUARD(isect_ids);
    GSPLAT_CHECK_INPUT(isect_ids);

    uint32_t n_isects = isect_ids.size(0);
    uint32_t *d_valid_isects;
    cudaMalloc(&d_valid_isects, sizeof(uint32_t));
    uint32_t valid_isects = n_isects;  // 初始值
    cudaMemcpy(d_valid_isects, &valid_isects, sizeof(uint32_t), cudaMemcpyHostToDevice);

    torch::Tensor offsets = torch::empty(
        {C, tile_height, tile_width}, isect_ids.options().dtype(torch::kInt32)
    );
    if (n_isects) {
        uint32_t n_tiles = tile_width * tile_height;
        uint32_t tile_n_bits = (uint32_t)floor(log2(n_tiles)) + 1;
        at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
        isect_offset_encode<<<
            (n_isects + GSPLAT_N_THREADS - 1) / GSPLAT_N_THREADS,
            GSPLAT_N_THREADS,
            0,
            stream>>>(
            n_isects,
            isect_ids.data_ptr<int64_t>(),
            C,
            n_tiles,
            tile_n_bits,
            offsets.data_ptr<int32_t>(),
            d_valid_isects
        );
        cudaMemcpy(&valid_isects, d_valid_isects, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaFree(d_valid_isects);     
    } else {
        offsets.fill_(0);
    }

    return std::make_tuple(offsets, valid_isects);
}

} // namespace gsplat