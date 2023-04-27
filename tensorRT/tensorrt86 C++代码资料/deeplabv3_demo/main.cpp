#include "deeplabv3_trt.h"

int main(int argc, char** argv) {
	std::string enginefile = "D:/TensorRT-8.6.0.12/bin/deeplabv3_mobilenet.engine";
	cv::Mat frame = imread("D:/images/messi_player.jpg");
	auto detector = std::make_shared<Deeplabv3TRTSegment>();
	detector->initConfig(enginefile);
	detector->segment(frame);
	cv::imshow("Deeplabv3 + TensorRT8.6 ”Ô“Â∑÷∏Ó—› æ", frame);
	cv::waitKey(0);
	cv::destroyAllWindows();
	return 0;
}