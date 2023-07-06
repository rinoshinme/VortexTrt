#include <iostream>
#include <string>
#include <opencv2/highgui/highgui.hpp>
#include "vortex/layers/image_proc.h"
#include "vortex/engine/engine_builder.h"


void test_letterbox()
{
    std::string image_path("../data/10350842.jpg");
    cv::Mat image = cv::imread(image_path);
    if (!image.isContinuous())
        image = image.clone();
    
    int w = image.cols;
    int h = image.rows;
    w = 640;
    h = 640;
    
    cv::Mat dst;
    dst = vortex::letterBox(image, w, h);

    cv::imwrite("dst.png", dst);
}


void test_onnx_parse()
{
    const std::string onnx_path("./yolov5s.onnx");
    const std::string engine_path("./yolov5s.engine");
    vortex::BuildOptions options;
    vortex::OnnxEngineBuilder builder(options);
    
    std::vector<uint32_t> input_dims = {3, 640, 640};
    std::string input_name("images");
    bool ret = builder.Build(onnx_path, engine_path, input_dims, input_name);
    if (ret)
        std::cout << "Done building engine\n";
    else
        std::cout << "Failed to build engine\n";
}

int main()
{
    test_onnx_parse();

    std::cout << "Finished...\n";
    return 0;
}
