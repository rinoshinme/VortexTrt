#include <iostream>
#include <string>
#include <opencv2/highgui/highgui.hpp>

#include "ImageProc.h"

int main()
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

    std::cout << "Finished...\n";
    return 0;
}
