#include "yolo_decode.h"


namespace vortex
{
    std::vector<YoloBox> gpu_decode(float* pred, int rows, int cols, float conf_thresh, float nms_thresh)
    {
        std::vector<YoloBox> boxes;
        return boxes;
    }
}
