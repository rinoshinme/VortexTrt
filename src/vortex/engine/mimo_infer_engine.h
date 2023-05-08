/*
 * Multiple-Input/multiple-Output models
 * Each model has different input/output format, we provide 
 * a simple (intermediate) interface wrapping the infer function
 * the Infer interface is not provided.
*/
#include <string>
#include <vector>
#include <map>
#include <memory>
#include "NvInfer.h"
#include "vortex/core/blob.h"
#include "vortex/core/logger.h"

namespace vortex
{
    class MimoInferEngine
    {
    private:
        Logger m_Logger;

        nvinfer1::ICudaEngine* m_Engine = nullptr;
        nvinfer1::IExecutionContext* m_Context = nullptr;
        nvinfer1::IRuntime* m_Runtime = nullptr;
        cudaStream_t m_Stream;

        std::vector<BlobInfo> m_InputInfo;
        std::vector<BlobInfo> m_OutputInfo;
        std::vector<std::shared_ptr<BlobF>> m_InputBlobs;
        std::vector<std::shared_ptr<BlobF>> m_OutputBlobs;

    public:
        MimoInferEngine() {}
        ~MimoInferEngine();

        void LoadEngine(const std::string& engine_path,
            const std::vector<BlobInfo>& input_info, 
            const std::vector<BlobInfo>& output_info);
        
        // InternalInfer take input_blobs as inputs and 
        // put output data into output_blobs
        // preprocess/postprocess not provided
        virtual void InternalInfer();
    };
}
