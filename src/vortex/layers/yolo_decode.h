#pragma once

#include <vector>
#include <set>

namespace vortex
{
    struct YoloBox
    {
        float top;
        float left;
        float bottom;
        float right;

        float confidence;
        int label;

        YoloBox() = default;
        YoloBox(float t, float l, float b, float r, float conf, int lbl)
            : top(t), left(l), bottom(b), right(r), confidence(conf), label(lbl) {}
    };

    float YoloBoxIou(const YoloBox& box1, const YoloBox& box2);
    bool YoloBoxGreater(const YoloBox& box1, const YoloBox& box2);

    class YoloDecoder
    {
    private:
        float m_ConfThresh;
        float m_NmsThresh;
        int m_NumClasses;

    public:
        YoloDecoder(float conf_thresh, float nms_thresh, int num_classes);

        std::vector<YoloBox> DecodeCpu(const std::vector<float>& pred, int image_width, int image_height);
        // std::vector<YoloBox> DecodeGpu(const std::vector<float>& pred);

    private:
        std::pair<int, float> Argmax(const float* pred, int start, int end);
        std::set<size_t> PerClassNms(const std::vector<YoloBox>& boxes, int class_id);
    };
}
