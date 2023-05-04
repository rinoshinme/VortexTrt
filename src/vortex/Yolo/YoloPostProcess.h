#pragma once
#include <vector>

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

    // post processing for yolo.
    std::vector<YoloBox> cpu_decode(float* pred, int rows, int cols, float conf_thresh=0.25f, float nms_thresh=0.45f);
    std::vector<YoloBox> gpu_decode(float* pred, int rows, int cols, float conf_thresh=0.25f, float nms_thresh=0.45f);
}
