/*
    normaly, onnx models can be built into engine files with trtexec when all 
    ops are supported.

*/
#pragma once
#include <cstdint>
#include <string>

namespace vortex
{
    struct BuildOptions
    {
        uint32_t maxBatchSize;
        size_t maxWorkspace;

        // get default options
        static BuildOptions Default();
    };

    class EngineBuilder
    {
    private:

    public:
        EngineBuilder(const BuildOptions& options);

        bool Build(const std::string& model_path, const std::string& engine_path);
        
    };
}
