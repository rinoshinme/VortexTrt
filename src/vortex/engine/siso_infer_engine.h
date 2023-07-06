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
    class SisoInferEngine
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
        SisoInferEngine() {}
        virtual ~SisoInferEngine();
        virtual bool LoadEngine(const std::string& engine_path, 
            const BlobInfo& input_info, 
            const BlobInfo& output_info);
        // infer model with blobs
        virtual void InternalInfer(MemoryType input_type = MemoryType::CPU_MEMORY, 
            MemoryType output_type = MemoryType::CPU_MEMORY);
    };
}
