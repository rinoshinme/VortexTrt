#pragma once

#include <string>
#include <memory>
#include <vector>
#include <opencv2/core/core.hpp>
#include "NvInfer.h"

#include "vortex/core/logger.h"
#include "vortex/core/blob.h"


namespace vortex
{
    // single input/output inference engine
    class SimpleInferEngine
    {
    protected:
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
    
    public:
        SimpleInferEngine() {}
        virtual ~SimpleInferEngine();
        virtual bool LoadEngine(const std::string& engine_path, 
            const BlobInfo& input_info, 
            const BlobInfo& output_info);

        // output fed into the buffer, postprocessing not provided
        virtual void Infer(cv::Mat& image, std::vector<float>& output);

        virtual void Preprocess(cv::Mat& image);
    };
}
