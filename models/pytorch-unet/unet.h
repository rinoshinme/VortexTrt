#pragma once

#include "core/simple_infer_engine.h"
#include "core/tensor4d.h"
#include "core/device_memory.h"

namespace vortex
{
    class Unet : public SimpleInferEngine
    {
    private:
        std::shared_ptr<Tensor4Df> m_InputBuffer;
        std::shared_ptr<DeviceMemory> m_InputBufferDevice;
        std::shared_ptr<DeviceMemory> m_OutputBufferDevice;
        float m_MaskThreshold;

    public:
        Unet(const std::string& model_path);
        void Infer(cv::Mat& image, std::vector<float>& output);

        void Preprocess(cv::Mat& image);
    };
}
