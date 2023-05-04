#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "alexnet.h"


int main()
{
    std::string engine_path("../alexnet.engine");
    vortex::Alexnet model(engine_path);
    std::string image_path("../../../data/29bb3ece3180_11.jpg");
    cv::Mat image = cv::imread(image_path);

    std::vector<float> outputs;
    model.Infer(image, outputs);

    for (int i = 0; i < 10; ++i)
    {
        std::cout << outputs[i] << std::endl;
    }

    std::cout << "Main\n";
    return 0;
}
