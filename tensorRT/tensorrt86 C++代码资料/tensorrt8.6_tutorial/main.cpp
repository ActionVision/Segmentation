#include <fstream>
#include <iostream>
#include <sstream>
#include <opencv2/opencv.hpp>
#include "NvInfer.h"

using namespace nvinfer1;
using namespace cv;

class Logger : public ILogger
{
	void log(Severity severity, const char* msg)  noexcept
	{
		// suppress info-level messages
		if (severity != Severity::kINFO)
			std::cout << msg << std::endl;
	}
} gLogger;

int main(int argc, char** argv) {
	auto builder = createInferBuilder(gLogger);
	builder->getLogger()->log(nvinfer1::ILogger::Severity::kERROR, "Create Builder...");
	return 0;
}