#include "resize_normalize.h"
#include <opencv2/imgproc/imgproc.hpp>

namespace vortex
{
    ResizeNormalize::ResizeNormalize(int width, int height, 
        const std::vector<float>& mean, const std::vector<float>& std, bool torgb)
    {
        m_Width = width;
        m_Height = height;
        m_ToRGB = torgb;
        for (int i = 0; i < 3; ++i)
        {
            m_Mean[i] = mean[i];
            m_Std[i] = std[i];
        }
    }

    ResizeNormalize::ResizeNormalize(int width, int height, float mean, float std, bool torgb)
    {
        m_Width = width;
        m_Height = height;
        m_ToRGB = torgb;
        for (int i = 0; i < 3; ++i)
        {
            m_Mean[i] = mean;
            m_Std[i] = std;
        }
    }

    void ResizeNormalize::Forward(cv::Mat& image, std::vector<float>& output)
    {
        // resize
        cv::Mat temp;
        cv::resize(image, temp, cv::Size(m_Width, m_Height));

        int output_numel = m_Width * m_Height * 3;
        int image_size = m_Width * m_Height;
        // normalize
        output.resize(output_numel);
        float* output_buffer = output.data();
        float* pRed = output_buffer;
        float* pGreen = output_buffer + image_size;
        float* pBlue = output_buffer + image_size * 2;
        if (m_ToRGB)
        {
            // swap red and blue channel
            float* temp = pRed;
            pRed = pBlue;
            pBlue = temp;
        }
        
        unsigned char* pImage = temp.data;
        for (int i = 0; i < image_size; ++i)
        {
            pRed[i] = (pImage[3 * i + 0] / 255.0 - m_Mean[0]) / m_Std[0];
            pGreen[i] = (pImage[3 * i + 1] / 255.0 - m_Mean[1]) / m_Std[1];
            pBlue[i] = (pImage[3 * i + 2] / 255.0 - m_Mean[2]) / m_Std[2];
        }
    }
}
