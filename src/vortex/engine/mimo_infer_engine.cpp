#include "mimo_infer_engine.h"


namespace vortex
{
    MimoInferEngine::~MimoInferEngine()
    {

    }

    void MimoInferEngine::LoadEngine(const std::string& engine_path, 
        const std::vector<BlobInfo>& input_info, 
        const std::vector<BlobInfo>& output_info)
    {

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

        for (size_t i = 0; i < n_inputs; ++i)
        {
            const int input_index = m_Engine->getBindingIndex(m_InputBlobs[i]->name.c_str());
            if (input_index >= n_ios) continue;
            array_buffer[input_index] = m_InputBlobs[i]->dataGpu;
            m_InputBlobs[i]->ToGpuAsync(m_Stream);
        }
        for (size_t i = 0; i < n_outputs; ++i)
        {
            const int output_index = m_Engine->getBindingIndex(m_OutputBlobs[i]->name.c_str());
            if (output_index >= n_ios) continue;
            array_buffer[output_index] = m_OutputBlobs[i]->dataGpu;
        }

#if 0
        m_Context->enqueue(1, (void*)array_buffer.data(), m_Stream, nullptr);
        // output to where?
        
        m_OutputBlob->ToCpuAsync(m_Stream, output.data());
        cudaStreamSynchronize(m_Stream);
#endif

    }
}
