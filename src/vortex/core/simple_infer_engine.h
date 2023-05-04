#pragma once

#include <string>
#include <memory>
#include <vector>
#include "NvInfer.h"
#include "core/logger.h"


namespace vortex
{  
    /* 
        base class for engine inference 
        support batching
    */
    class SimpleInferEngine
    {
    protected:
        Logger m_Logger;
        nvinfer1::ICudaEngine* m_Engine = nullptr;
        nvinfer1::IExecutionContext* m_Context = nullptr;
        nvinfer1::IRuntime* m_Runtime = nullptr;
        cudaStream_t m_Stream;

    public:
        virtual ~SimpleInferEngine();

        virtual bool LoadEngine(const std::string& engine_path);

    private:
        bool LoadEngineData(const std::string& file_path, std::vector<unsigned char>& data);
    };

    // std::shared_ptr<SimpleInferEngine> MakeInferEngine();
}
