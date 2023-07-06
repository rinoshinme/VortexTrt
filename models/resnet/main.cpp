#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "vortex/engine/simple_infer_engine.h"

namespace vortex
{
    class ResNet : public SimpleInferEngine
    {
    public:
        ResNet(const std::string& engine_path)
        {
            BlobInfo input_info = {
                "input", 1, 3, 224, 224
            };

            BlobInfo output_info = {
                "output", 1, 1000, 1, 1
            };

            this->LoadEngine(engine_path, input_info, output_info);
        }
    };
}


int main()
{
    std::string engine_path("../resnet50.engine");
    vortex::ResNet model(engine_path);
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
