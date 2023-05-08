#pragma once

#include <opencv2/core/core.hpp>
#include <string>
#include <vector>
#include <set>
#include "NvInfer.h"
#include "vortex/core/logger.h"
#include "vortex/core/blob.h"
#include "vortex/layers/yolo_decode.h"


namespace vortex
{
    // TODO: 
    // 1. integrate into MimoInferEngine
    // 2. CUDA accelerated preprocessing and postprocessing.
    class Yolov5
    {
    private:
        Logger m_Logger;

        nvinfer1::ICudaEngine* m_Engine = nullptr;
        nvinfer1::IExecutionContext* m_Context = nullptr;
        nvinfer1::IRuntime* m_Runtime = nullptr;
        cudaStream_t m_Stream;

        // input&output parameters
        BlobInfo m_InputInfo;
        BlobInfo m_OutputInfo;
        std::shared_ptr<BlobF> m_InputBlob;
        std::shared_ptr<BlobF> m_OutputBlob;

        std::shared_ptr<YoloDecoder> m_Decoder;

    public:
        Yolov5(const std::string& engine_path);
        ~Yolov5();

        void Detect(cv::Mat& image, std::vector<YoloBox>& boxes);

    private:
        void Preprocess(cv::Mat& image);
        bool LoadEngine(const std::string& engine_path, 
            const BlobInfo& input_info, const BlobInfo& output_info);
    };
}
