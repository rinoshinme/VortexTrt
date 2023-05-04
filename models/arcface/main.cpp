#include "arcface.h"


int main()
{
    std::string engine_path("./arcface_resnet50.engine");
    Arcface model(engine_path);
    std::string image_path("../../data/sample1.jpg");
    cv::Mat image = cv::imread(image_path);

    std::vector<float> outputs;
    model.Infer(image, outputs);

    for (int i = 0; i < 10; ++i)
    {
        std::cout << outputs[i] << std::endl;
    }
}
