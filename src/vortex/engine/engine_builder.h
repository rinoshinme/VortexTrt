/*
    normaly, onnx models can be built into engine files with trtexec when all 
    ops are supported.
*/
#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <memory>
#include "vortex/core/logger.h"
#include "NvInfer.h"


namespace vortex
{
    enum class BuildPrecision
    {
        FP32,
        FP16,
        INT8
    };

    struct BuildOptions
    {
        uint32_t optBatchSize = 1;
        uint32_t maxBatchSize = 16;
        size_t maxWorkspace = 4 << 30;
        BuildPrecision precision = BuildPrecision::FP32;
        uint32_t deviceIndex = 0;
        bool dynamicBatchSize = true;
    };

    /*
    * Sinle Input/Output Engine Builder
    */
    class EngineBuilder
    {
    private:
        BuildOptions m_Options;
        Logger m_Logger;
        
        nvinfer1::ICudaEngine* m_Engine;

    public:
        EngineBuilder(const BuildOptions& options);

        bool Build(const std::string& onnx_path, const std::string& engine_path, const std::vector<uint32_t>& input_dims);

    private:
        
    };

    // TODO
    class Int8EngineBuilder
    {

    };
}
