#include "mimo_infer_engine.h"
#include "vortex/utils/fileops.h"

namespace vortex
{
    MimoInferEngine::~MimoInferEngine()
    {
        if (m_Context != nullptr)
            m_Context->destroy();
        if (m_Engine != nullptr)
            m_Engine->destroy();
        if (m_Runtime != nullptr)
            m_Runtime->destroy();
    }

    bool MimoInferEngine::LoadEngine(const std::string& engine_path, 
        const std::vector<BlobInfo>& input_info, 
        const std::vector<BlobInfo>& output_info)
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
        for (auto& info : input_info)
            m_InputBlobs.push_back(std::make_shared<BlobF>(info));
        for (auto& info : output_info)
            m_OutputBlobs.push_back(std::make_shared<BlobF>(info));
        return true;
    }

    void MimoInferEngine::InternalInfer()
    {
        // assume the input data is preprocessed and saved into input_blobs
        // set buffer data
        size_t n_inputs = m_InputBlobs.size();
        size_t n_outputs = m_OutputBlobs.size();
        size_t n_ios = n_inputs + n_outputs;
        std::vector<float*> array_buffer;
        array_buffer.resize(n_ios);

        // feed inputs
        for (size_t i = 0; i < n_inputs; ++i)
        {
            const int input_index = m_Engine->getBindingIndex(m_InputBlobs[i]->name.c_str());
            if (input_index >= n_ios) continue;
            array_buffer[input_index] = m_InputBlobs[i]->dataGpu;
            m_InputBlobs[i]->ToGpuAsync(m_Stream);
            // set binding dimension
            nvinfer1::Dims dims;
            dims.d[0] = m_InputInfo[i].batch;
            dims.d[1] = m_InputInfo[i].channels;
            dims.d[2] = m_InputInfo[i].height,
            dims.d[3] = m_InputInfo[i].width;
            dims.nbDims = 4;
            m_Context->setBindingDimensions(input_index, dims);
        }

        for (size_t i = 0; i < n_outputs; ++i)
        {
            const int output_index = m_Engine->getBindingIndex(m_OutputBlobs[i]->name.c_str());
            if (output_index >= n_ios) continue;
            array_buffer[output_index] = m_OutputBlobs[i]->dataGpu;
        }

        // collect output
        m_Context->enqueue(1, (void**)array_buffer.data(), m_Stream, nullptr);
        for (size_t i = 0; i < n_outputs; ++i)
        {
            m_OutputBlobs[i]->ToCpuAsync(m_Stream);
        }

        cudaStreamSynchronize(m_Stream);
    }
}
