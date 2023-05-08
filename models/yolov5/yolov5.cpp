#include "yolov5.h"
#include <algorithm>
#include <set>
#include <opencv2/imgproc/imgproc.hpp>
#include "vortex/utils/fileops.h"


namespace vortex
{
    Yolov5::Yolov5(const std::string& engine_path)
    {
        BlobInfo input_info = {
            "images", 640, 640, 3
        };
        BlobInfo output_info = {
            "output", 25200, 85, 1
        };
        this->LoadEngine(engine_path, input_info, output_info);

        m_Decoder = std::make_shared<YoloDecoder>(0.5, 0.5, 80);
    }

    Yolov5::~Yolov5()
    {
        if (m_Context != nullptr)
            m_Context->destroy();
        if (m_Engine != nullptr)
            m_Engine->destroy();
        if (m_Runtime != nullptr)
            m_Runtime->destroy();
    }

    void Yolov5::Detect(cv::Mat& image, std::vector<YoloBox>& boxes)
    {
        this->Preprocess(image);

        // set buffer data
        void* buffers[2] = { nullptr };
        const int inputIndex = m_Engine->getBindingIndex(m_InputBlob->name.c_str());
        const int outputIndex = m_Engine->getBindingIndex(m_OutputBlob->name.c_str());
        buffers[inputIndex] = m_InputBlob->dataGpu;
        buffers[outputIndex] = m_OutputBlob->dataGpu;

        m_InputBlob->ToGpuAsync(m_Stream);
        m_Context->enqueue(1, buffers, m_Stream, nullptr);
        std::vector<float> output;
        output.resize(m_OutputBlob->count);
        m_OutputBlob->ToCpuAsync(m_Stream, output.data());
        cudaStreamSynchronize(m_Stream);

        // parse output info
        int width = image.cols;
        int height = image.rows;
        boxes = m_Decoder->DecodeCpu(output, width, height);
    }

    void Yolov5::Preprocess(cv::Mat& image)
    {
        // simple resize
        // TODO: letterbox
        uint32_t width = m_InputInfo.width;
        uint32_t height = m_InputInfo.height;
        cv::Mat temp;
        cv::resize(image, temp, cv::Size(width, height));

        uint32_t image_area = width * height;
        
        float* input_buffer = m_InputBlob->dataCpu;
        float* pBlue = input_buffer;
        float* pGreen = input_buffer + image_area;
        float* pRed = input_buffer + image_area * 2;

        unsigned char* pImage = temp.data;
        for (uint32_t i = 0; i < image_area; ++i)
        {
            pRed[i] = pImage[3 * i + 0] / 255.0;
            pGreen[i] = pImage[3 * i + 1] / 255.0;
            pBlue[i] = pImage[3 * i + 2] / 255.0;
        }
    }

    bool Yolov5::LoadEngine(const std::string& engine_path, 
        const BlobInfo& input_info, const BlobInfo& output_info)
    {
        // load engine data
        std::vector<unsigned char> engine_data;
        bool ret = loadBinaryContent(engine_path, engine_data);
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

        m_InputInfo = input_info;
        m_OutputInfo = output_info;
        m_InputBlob = std::make_shared<BlobF>(input_info);
        m_OutputBlob = std::make_shared<BlobF>(output_info);
        return true;
    }
}
