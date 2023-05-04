#include "simple_infer_engine.h"
#include <fstream>
#include <vector>


namespace vortex
{
    SimpleInferEngine::~SimpleInferEngine()
    {
        if (m_Context != nullptr)
            m_Context->destroy();
        if (m_Engine != nullptr)
            m_Engine->destroy();
        if (m_Runtime != nullptr)
            m_Runtime->destroy();
    }

    bool SimpleInferEngine::LoadEngine(const std::string& engine_path)
    {
        // load data from file
        std::vector<unsigned char> engine_data;
        bool ret = LoadFile(model_path, engine_data);
        if (!ret)
            return false;
        // create engine
        m_Runtime = nvinfer1::createInferRuntime(m_Logger);
        if (m_Runtime == nullptr)
            return false;
        m_Engine = m_Runtime->deserializeCudaEngine(engine_data.data(), engine_data.size());
        if (m_Engine == nullptr)
            return false;
        m_Context = m_Engine->createExecutionContext();
        if (m_Context == nullptr)
            return false;
        
        // create cuda stream
        checkRuntime(cudaStreamCreate(&m_Stream));
        return true;
    }

    bool SimpleInferEngine::LoadEngineData(const std::string& file_path, std::vector<unsigned char>& data)
    {
        std::ifstream file(file_path, std::ios::binary);
        if (file.good())
        {
            file.seekg(0, file.end);
            size_t file_size = file.tellg();
            file.seekg(0, file.beg);
            data.resize(file_size);
            char* ptr = reinterpret_cast<char*>(data.data());
            file.read(ptr, file_size);
            file.close();
            return true;
        }
        return false;
    }

}
