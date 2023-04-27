#include "unet_trt.h"

int main(int argc, char** argv) {
	// std::string enginefile = "D:/TensorRT-8.6.0.12/bin/unet_road_16.engine";
	std::string enginefile = "D:/python/tensorrt_tutorial/unet_road_int8.engine";
	cv::Mat frame = imread("D:/python/tensorrt_tutorial/qt002.png");
	auto detector = std::make_shared<UNetTRTSegment>();
	detector->initConfig(enginefile);
	detector->segment(frame);
	cv::imshow("UNet + TensorRT8.6 ”Ô“Â∑÷∏Ó—› æ", frame);
	cv::waitKey(0);
	cv::destroyAllWindows();
	return 0;
}