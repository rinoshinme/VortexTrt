#include "engine_builder.h"
#include <memory>
#include <fstream>
#include <iostream>


namespace vortex
{
    OnnxEngineBuilder::OnnxEngineBuilder(const BuildOptions& options)
    {
        m_Options = options;
        m_Engine = nullptr;
    }

    bool OnnxEngineBuilder::Build(const std::string& onnx_path, const std::string& engine_path, 
        const std::vector<uint32_t>& input_dims, const std::string& input_name)
    {
        nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(m_Logger);
        if (!builder) return false;

        const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        nvinfer1::INetworkDefinition* network = builder->createNetworkV2(explicitBatch);
        if (!network) return false;

        nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
        if (!config) return false;

        nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, m_Logger);
        if (!parser) return false;

        // construct network
        bool constructed = ConstructNetwork(builder, network, config, parser, onnx_path, input_dims, input_name);
        if (!constructed) return false;

        // build
        nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
        if (!engine) return false;

        //serialize into engine file
        SaveEngine(*engine, engine_path);

        return true;
    }

    bool OnnxEngineBuilder::ConstructNetwork(
        nvinfer1::IBuilder* builder,
        nvinfer1::INetworkDefinition* network, 
        nvinfer1::IBuilderConfig* config,
        nvonnxparser::IParser* parser,
        const std::string& onnx_path, 
        const std::vector<uint32_t>& input_dims, 
        const std::string& input_name)
    {
        nvinfer1::ILogger::Severity s = nvinfer1::ILogger::Severity::kERROR;
        auto parsed = parser->parseFromFile(onnx_path.c_str(), static_cast<int>(s));

        for (uint32_t i = 0; i < parser->getNbErrors(); ++i)
        {
            std::cout << parser->getError(i)->desc() << std::endl;
        }

        if (!parsed) return false;

        // set batch size
        if (m_Options.dynamicBatchSize)
        {
            builder->setMaxBatchSize(m_Options.maxBatchSize);
            assert(input_dims.size() == 3);
            uint32_t channels = input_dims[0];
            uint32_t height = input_dims[1];
            uint32_t width = input_dims[2];

            nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
            profile->setDimensions(input_name.c_str(), 
                nvinfer1::OptProfileSelector::kMIN, 
                nvinfer1::Dims4(m_Options.minBatchSize, channels, height, width));
            profile->setDimensions(input_name.c_str(), 
                nvinfer1::OptProfileSelector::kOPT, 
                nvinfer1::Dims4(m_Options.optBatchSize, channels, height, width));
            profile->setDimensions(input_name.c_str(), 
                nvinfer1::OptProfileSelector::kMAX, 
                nvinfer1::Dims4(m_Options.maxBatchSize, channels, height, width));

            config->addOptimizationProfile(profile);
        }
        else
        {
            builder->max_batch_size = 1;
        }

        // todo: set model precision
        assert(m_Options.precision != BuildPrecision::INT8);


        config->setMaxWorkspaceSize(m_Options.maxWorkspace);
        return true;
    }

    bool OnnxEngineBuilder::SaveEngine(const nvinfer1::ICudaEngine& engine, const std::string& filepath)
    {
        std::ofstream engineFile(filepath, std::ios::binary);
        if (!engineFile) return false;

        nvinfer1::IHostMemory* ptr = engine.serialize();
        if (!ptr) return false;

        engineFile.write(reinterpret_cast<const char*>(ptr->data()), ptr->size());
        ptr->destroy();
        engineFile.close();

        return true;
    }
}
