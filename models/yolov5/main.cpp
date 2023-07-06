#include <iostream>
#include <memory>
#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "vortex/utils/fileops.h"
#include "yolov5.h"

using namespace vortex;


void SampleForward(std::shared_ptr<Yolov5> model)
{
    const std::string input_path("images.txt");
    std::vector<float> input_data;
    loadFloats(input_path, input_data);
    
    std::vector<float> output_data;
    model->Infer(input_data, output_data);
    const std::string output_text("output.txt");
    saveFloats(output_text, output_data);
}


int main()
{
    const std::string engine_path = "./yolov5s.engine";
    std::shared_ptr<Yolov5> model = std::make_shared<Yolov5>(engine_path);
    // SampleForward(model);

    const std::string image_path = "../../../data/bus.jpg";
    cv::Mat image = cv::imread(image_path);

    std::vector<DetBox> boxes;
    model->Detect(image, boxes);

    // output boxes
    for (auto& box : boxes)
    {
        printf("[%.03f, %.03f, %.03f, %.03f, %.03f, %d]\n", box.xmin, box.ymin, box.xmax, box.ymax, box.score, box.class_index);
    }

    std::cout << "Bye!\n";
    return 0;
}
