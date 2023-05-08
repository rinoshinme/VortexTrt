#pragma once

#include <opencv2/core/core.hpp>

namespace vortex
{
    cv::Mat letterBox(cv::Mat& image, int target_width, int target_height);
}
