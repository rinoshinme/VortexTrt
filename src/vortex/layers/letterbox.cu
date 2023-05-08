#include "letterbox.h"
#include <cstdio>
#include <opencv2/highgui/highgui.hpp>
#include "vortex/core/core.h"
#include "affine.h"
#include "vortex/core/image.h"
#include "vortex/core/cuda_utils.h"

namespace vortex
{

#define IMAGE_SAMPLE(image, x, y) \
    (image.data + y * image.bytesPerLine + x * image.channels)

#define BILINEAR_INTERP(v1, v2, v3, v4, alpha, beta) \
    (v1 * (1 - alpha) * (1 - beta) + v2 * (1 - alpha) * beta + v3 * alpha * (1 - beta) + v4 * alpha * beta)

    
    __global__ void WarpImage(const Image src, const Image dst, uint8_t fill_value, const AffineMatrix matrix)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x < 0 || x >= dst.width || y < 0 || y >= dst.height)
            return;
        
        uint8_t* dst_ptr = IMAGE_SAMPLE(dst, x, y);
        dst_ptr[0] = fill_value;
        dst_ptr[1] = fill_value;
        dst_ptr[2] = fill_value;

        float px;
        float py;
        matrix.apply(x * 1.0f, y * 1.0f, &px, &py);

        if (px < 0 || px >= src.width - 1 || py < 0 || py >= src.height - 1)
        {
            // do nothing
            return;
        }

        // do interpolation
        int x1 = int(px);
        int y1 = int(py);
        int x2 = int(px) + 1;
        int y2 = int(py) + 1;
        float alpha = px - x1;
        float beta = py - y1;

        uint8_t* v1 = IMAGE_SAMPLE(src, x1, y1);
        uint8_t* v2 = IMAGE_SAMPLE(src, x1, y2);
        uint8_t* v3 = IMAGE_SAMPLE(src, x2, y1);
        uint8_t* v4 = IMAGE_SAMPLE(src, x2, y2);

#if 1
        dst_ptr[0] = static_cast<uint8_t>(BILINEAR_INTERP(v1[0], v2[0], v3[0], v4[0], alpha, beta));
        dst_ptr[1] = static_cast<uint8_t>(BILINEAR_INTERP(v1[1], v2[1], v3[1], v4[1], alpha, beta));
        dst_ptr[2] = static_cast<uint8_t>(BILINEAR_INTERP(v1[2], v2[2], v3[2], v4[2], alpha, beta));
#else
        dst_ptr[0] = v2[0];
        dst_ptr[1] = v2[1];
        dst_ptr[2] = v2[2];
#endif 
    }

    cv::Mat letterBox(cv::Mat& image, int target_width, int target_height)
    {
        // calculate source and target rectangle 
        uint32_t w = image.cols;
        uint32_t h = image.rows;
        Rect source_rectangle {0, 0, w, h};

        float dx = 0;
        float dy = 0;
        float dw = 0;
        float dh = 0;
        
        float scale = w * 1.0f / h;
        float t = scale * target_height;
        if (t < target_width)
        {
            // width < height
            dx = (target_width - t) / 2;
            dw = t;
            dh = target_height;
        }
        else
        {
            // width > height
            dw = target_width;
            dh = target_width / scale;
            dy = (target_height - dh) / 2;
        }

        Rect target_rectangle = {uint32_t(dx), uint32_t(dy), uint32_t(dw), uint32_t(dh)};

        std::cout << source_rectangle << std::endl;
        std::cout << target_rectangle << std::endl;

        // AffineMatrix matrix(source_rectangle, target_rectangle);
        AffineMatrix matrix(target_rectangle, source_rectangle);

        std::cout << matrix << std::endl;
        
        // allocate memory on device
        Image src(image);
        Image dst(target_width, target_height, 3);

        src.toGpu();
        dst.toGpu();

        // do processing
        dim3 block_size(32, 32);
        dim3 grid_size((target_width + 31) / 32, (target_height + 31) / 32);
        WarpImage<<<block_size, grid_size>>>(src, dst, 114, matrix);
        dst.syncCpu();

        checkRuntime(cudaDeviceSynchronize());
        
        cv::Mat dstImg = dst.toCvMat();
        src.clear();
        dst.clear();

        return dstImg;
    }
}
