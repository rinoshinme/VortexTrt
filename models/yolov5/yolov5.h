#pragma once

#include <opencv2/core/core.hpp>
#include <string>
#include <vector>
#include <set>
#include "NvInfer.h"
#include "vortex/core/logger.h"
#include "vortex/core/blob.h"
#include "vortex/layers/yolo_decode.h"
#include "vortex/engine/siso_infer_engine.h"


namespace vortex
{
    // TODO: 
    // 1. integrate into SisoInferEngine
    // 2. CUDA accelerated preprocessing and postprocessing.
    struct DetBox
    {
        float xmin;
        float ymin;
        float xmax;
        float ymax;
        float score;

        uint32_t class_index;

        float Area() const { return (xmax - xmin) * (ymax - ymin); }
    };

    float BoxOverlap(DetBox& b1, DetBox& b2);

    void sortBoxesByScore(std::vector<DetBox>& boxes);

    class Yolov5 : public SisoInferEngine
    {
    private:
        float m_ConfThresh;
        float m_IouThresh;
        std::shared_ptr<YoloDecoder> m_Decoder;

    public:
        Yolov5(const std::string& engine_path, float conf_thresh = 0.5, float iou_thresh = 0.5);
        ~Yolov5() {}

        void Detect(cv::Mat& image, std::vector<DetBox>& boxes);

        // sample infer
        void Infer(const std::vector<float>& inputs, std::vector<float>& outputs);

    private:
        void Preprocess(cv::Mat& image);
        void Postprocess(std::vector<DetBox>& boxes);
        std::vector<uint32_t> Nms(std::vector<DetBox>& boxes);

    };
}
