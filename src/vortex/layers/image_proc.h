#pragma once

#include <opencv2/core/core.hpp>
#include "affine.h"


namespace vortex
{
    class ResizeNormalize
    {
    private:
        uint32_t m_Width;
        uint32_t m_Height;
        bool m_ZeroToOne;
        std::vector<float> m_Mean;
        std::vector<float> m_Std;

    public:
        ResizeNormalize();
        void operator()(cv::Mat& image);
        void operator()(std::vector<cv::Mat>& images);
    };

    cv::Mat letterBox(cv::Mat& image, int target_width, int target_height);
}
