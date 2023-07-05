#pragma once

#include <vector>
#include <opencv2/core/core.hpp>


namespace vortex
{
    // default input dimension (hwc)
    class ResizeNormalize
    {
    private:
        uint32_t m_Width;
        uint32_t m_Height;
        bool m_ToRGB;
        float m_Mean[3];
        float m_Std[3];

    public:
        ResizeNormalize(int width, int height, 
            const std::vector<float>& mean, 
            const std::vector<float>& std, 
            bool torgb = true);
        ResizeNormalize(int width, int height, float mean, float std, bool torgb = true);

        void Forward(cv::Mat& image, std::vector<float>& output);
    };
    
}
