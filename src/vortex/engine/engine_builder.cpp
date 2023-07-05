#include "engine_builder.h"


namespace vortex
{
    EngineBuilder::EngineBuilder(const BuildOptions options)
    {
        m_Options = options;
        m_Engine = nullptr;
    }

    bool EngineBuilder::Build(const std::string& onnx_path, const std::string& engine_path, const std::vector<uint32_t>& input_dims)
    {
        

        return true;
    }

    
}
