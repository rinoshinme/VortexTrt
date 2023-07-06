#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "vortex/engine/simple_infer_engine.h"


namespace vortex
{
    class Unet : SimpleInferEngine
    {
        Unet(const std::string& engine_path)
        {
            BlobInfo input_info = {
                "input", 1, 3, 512, 512
            };

            BlobInfo output_info = {
                "output", 1, 512, 512, 1
            };
            this->LoadEngine(engine_path, input_info, output_info);
        }

        virtual void Preprocess(cv::Mat& image) override
        {
            // custom preprocessing
            cv::Mat temp;
            uint32_t width = m_InputInfo.width;
            uint32_t height = m_InputInfo.height;
            uint32_t channels = m_InputInfo.channels;
            cv::resize(image, temp, cv::Size(width, height));
            uint32_t image_area = width * height;
            uint32_t numel = image_area * channels;

            float* input_buffer = m_InputBlob->dataCpu;
            float* pBlue = input_buffer;
            float* pGreen = input_buffer + image_area;
            float* pRed = input_buffer + image_area * 2;
            unsigned char* pImage = temp.data;

            for (uint32_t i = 0; i < image_area; ++i)
            {
                pRed[i] = pImage[3 * i + 0] / 255.0;
                pGreen[i] = pImage[3 * i + 1] / 255.0;
                pBlue[i] = pImage[3 * i + 2] / 255.0;
            }
        }
    };
}



int main()
{
    return 0;
}

