#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "yolov5.h"


int main()
{
    std::string engine_path = "../yolov5m.engine";
    auto model = vortex::Yolov5(engine_path);
    std::string image_path = "../../../data/bus.jpg";
    cv::Mat image = cv::imread(image_path);
    std::vector<vortex::YoloBox> boxes;
    model.Detect(image, boxes);
    for (size_t i = 0; i < boxes.size(); ++i)
    {
        auto& box = boxes[i];
        std::cout << "box: " << box.left << ", " << box.top << ", " << box.right << ", " << box.bottom << std::endl;
        std::cout << box.score << ", " << box.class_id << std::endl;
    }

    std::cout << "Bye!\n";
    return 0;
}
