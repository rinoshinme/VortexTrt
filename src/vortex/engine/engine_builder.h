/*
    normaly, onnx models can be built into engine files with trtexec when all 
    ops are supported.
*/
#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <memory>
#include <map>
#include "vortex/core/logger.h"
#include "NvInfer.h"
#include "vortex/core/common.h"
#include "NvOnnxConfig.h"
#include "NvOnnxParser.h"

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
        uint32_t minBatchSize = 1;
        uint32_t optBatchSize = 16;
        uint32_t maxBatchSize = 16;
        size_t maxWorkspace = 4 << 30;
        bool dynamicBatchSize = true;

        BuildPrecision precision = BuildPrecision::FP32;
    };

    /*
    * Sinle Input/Output Engine Builder
    */
    class OnnxEngineBuilder
    {
    private:
        BuildOptions m_Options;
        Logger m_Logger;
        
        // nvinfer1::ICudaEngine* m_Engine;

    public:
        OnnxEngineBuilder(const BuildOptions& options);

        bool Build(const std::string& onnx_path, const std::string& engine_path, 
            const std::vector<uint32_t>& input_dims, const std::string& input_name);
        
        // support multiple input models
        // bool Build(const std::string& onnx_path, const std::string& engine_path, 
        //     const std::map<std::string, nvinfer1::Dim3>& input_info);

    private:
        bool ConstructNetwork(
            nvinfer1::IBuilder* builder,
            nvinfer1::INetworkDefinition* network, 
            nvinfer1::IBuilderConfig* config,
            nvonnxparser::IParser* parser, 
            const std::string& onnx_path, 
            const std::vector<uint32_t>& input_dims, 
            const std::string& input_name);
        
        bool SaveEngine(const nvinfer1::ICudaEngine& engine, const std::string& filepath);
    };
}
