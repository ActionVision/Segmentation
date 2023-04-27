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

std::string labels_txt_file = "D:/python/pytorch_openvino_demo/imagenet_classes.txt";
std::vector<std::string> readClassNames();
std::vector<std::string> readClassNames()
{
	std::vector<std::string> classNames;

	std::ifstream fp(labels_txt_file);
	if (!fp.is_open())
	{
		printf("could not open file...\n");
		exit(-1);
	}
	std::string name;
	while (!fp.eof())
	{
		std::getline(fp, name);
		if (name.length())
			classNames.push_back(name);
	}
	fp.close();
	return classNames;
}

int main(int argc, char** argv) {
	std::vector<std::string> labels = readClassNames();
	std::string enginepath = "D:/TensorRT-8.6.0.12/bin/resnet18.engine";
	std::ifstream file(enginepath, std::ios::binary);
	char* trtModelStream = NULL;
	int size = 0;
	if (file.good()) {
		file.seekg(0, file.end);
		size = file.tellg();
		file.seekg(0, file.beg);
		trtModelStream = new char[size];
		assert(trtModelStream);
		file.read(trtModelStream, size);
		file.close();
	}

	// 初始化几个对象
	auto runtime = createInferRuntime(gLogger);
	assert(runtime != nullptr);
	auto engine = runtime->deserializeCudaEngine(trtModelStream, size);
	assert(this->engine != nullptr);
	auto context = engine->createExecutionContext();
	assert(this->context != nullptr);
	delete[] trtModelStream;

	void* buffers[2] = { NULL, NULL };
	std::vector<float> prob;
	cudaStream_t stream;

	int input_index = engine->getBindingIndex("input.1");
	int output_index = engine->getBindingIndex("191");

	// 获取输入维度信息 NCHW
	int input_h = engine->getBindingDimensions(input_index).d[2];
	int input_w = engine->getBindingDimensions(input_index).d[3];
	printf("inputH : %d, inputW: %d \n", input_h, input_w);

	// 获取输出维度信息
	int output_h = engine->getBindingDimensions(output_index).d[0];
	int output_w = engine->getBindingDimensions(output_index).d[1];
	std::cout << "out data format: " << output_h << "x" << output_w << std::endl;

	// 创建GPU显存输入/输出缓冲区
	std::cout << " input/outpu : " << engine->getNbBindings() << std::endl;
	cudaMalloc(&buffers[input_index], input_h * input_w * 3 * sizeof(float));
	cudaMalloc(&buffers[output_index], output_h *output_w * sizeof(float));

	// 创建临时缓存输出
	prob.resize(output_h * output_w);

	// 创建cuda流
	cudaStreamCreate(&stream);

	cv::Mat image = cv::imread("D:/images/space_shuttle.jpg");
	cv::Mat rgb, blob;
	cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);
	cv::resize(rgb, blob, cv::Size(224, 224));
	blob.convertTo(blob, CV_32F);
	blob = blob / 255.0;
	cv::subtract(blob, cv::Scalar(0.485, 0.456, 0.406), blob);
	cv::divide(blob, cv::Scalar(0.229, 0.224, 0.225), blob);

	// HWC => CHW
	cv::Mat tensor = ::dnn::blobFromImage(blob);

	// 内存到GPU显存
	cudaMemcpyAsync(buffers[0], tensor.ptr<float>(), input_h * input_w * 3 * sizeof(float), cudaMemcpyHostToDevice, stream);

	// 推理
	context->enqueueV2(buffers, stream, nullptr);

	// GPU显存到内存
	cudaMemcpyAsync(prob.data(),  buffers[1], output_h *output_w * sizeof(float), cudaMemcpyDeviceToHost, stream);

	// 后处理
	cv::Mat probmat(output_h, output_w, CV_32F, (float*)prob.data());
	cv::Point maxL, minL;
	double maxv, minv;
	cv::minMaxLoc(probmat, &minv, &maxv, &minL, &maxL);
	int max_index = maxL.x;
	std::cout << "label id: " << max_index << std::endl;
	cv::putText(image, labels[max_index], cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2, 8);
	cv::imshow("输入图像", image);
	cv::waitKey(0);

	// 同步结束，释放资源
	cudaStreamSynchronize(stream);
	cudaStreamDestroy(stream);

	if (!context) {
		context->destroy();
	}
	if (!engine) {
		engine->destroy();
	}
	if (!runtime) {
		runtime->destroy();
	}
	if (!buffers[0]) {
		delete[] buffers;
	}

	return 0;
}