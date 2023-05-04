#include <iostream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "models/arcface.h"
#include "models/unet.h"

using namespace vortex;


void test_arcface()
{
    std::string engine_path("../../sample/arcface_resnet50.engine");
    Arcface model(engine_path);
    std::string image_path("../../sample/sample1.jpg");
    cv::Mat image = cv::imread(image_path);

    std::vector<float> outputs;
    model.Infer(image, outputs);

    for (int i = 0; i < 10; ++i)
    {
        std::cout << outputs[i] << std::endl;
    }
}

void test_unet()
{
    std::string engine_path("../../sample/unet.engine");
    Arcface model(engine_path);
    std::string image_path("../../sample/29bb3ece3180_11.jpg");
    cv::Mat image = cv::imread(image_path);

    std::vector<float> outputs;
    model.Infer(image, outputs);

    for (int i = 0; i < 10; ++i)
    {
        std::cout << outputs[i] << std::endl;
    }
}


int main1()
{
    // test_arcface();
    test_unet();
    return 0;
}
