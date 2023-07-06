#include "siso_infer_engine.h"
#include <fstream>
#include <vector>
#include <opencv2/imgproc/imgproc.hpp>
#include "vortex/core/cuda_utils.h"
#include "vortex/utils/fileops.h"


namespace vortex
{
    SisoInferEngine::~SisoInferEngine()
    {
        if (m_Context != nullptr)
            m_Context->destroy();
        if (m_Engine != nullptr)
            m_Engine->destroy();
        if (m_Runtime != nullptr)
            m_Runtime->destroy();
    }

    bool SisoInferEngine::LoadEngine(const std::string& engine_path, 
        const BlobInfo& input_info, const BlobInfo& output_info)
    {
        // load data from file
        std::vector<unsigned char> engine_data;
        bool ret = loadBinaryContent(engine_path, engine_data);
        if (!ret) return false;
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

        // create infer blobs
        m_InputInfo = input_info;
        m_OutputInfo = output_info;
        m_InputBlob = std::make_shared<BlobF>(input_info);
        m_OutputBlob = std::make_shared<BlobF>(output_info);

        return true;
    }

    void SisoInferEngine::InternalInfer(MemoryType input_type, MemoryType output_type)
    {
        // set buffer data
        void* buffers[2] = { nullptr };
        const int inputIndex = m_Engine->getBindingIndex(m_InputBlob->name.c_str());
        const int outputIndex = m_Engine->getBindingIndex(m_OutputBlob->name.c_str());
        buffers[inputIndex] = m_InputBlob->dataGpu;
        buffers[outputIndex] = m_OutputBlob->dataGpu;

        // set binding dimension
        nvinfer1::Dims dims;
        dims.d[0] = m_InputInfo.batch;
        dims.d[1] = m_InputInfo.channels;
        dims.d[2] = m_InputInfo.height,
        dims.d[3] = m_InputInfo.width;
        dims.nbDims = 4;
        m_Context->setBindingDimensions(inputIndex, dims);

        // it's better to wrap pre/post processing here
        // for better performance.
        
        if (input_type == MemoryType::CPU_MEMORY)
            m_InputBlob->ToGpuAsync(m_Stream);
        m_Context->enqueue(1, buffers, m_Stream, nullptr);
        if (output_type == MemoryType::CPU_MEMORY)
            m_OutputBlob->ToCpuAsync(m_Stream);
        
        cudaStreamSynchronize(m_Stream);
    }
}
