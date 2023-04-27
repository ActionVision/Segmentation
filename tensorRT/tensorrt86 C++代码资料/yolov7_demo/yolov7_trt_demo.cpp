# include "yolov7_trt_demo.h"
class Logger : public ILogger
{
	void log(Severity severity, const char* msg)  noexcept
	{
		// suppress info-level messages
		if (severity != Severity::kINFO)
			std::cout << msg << std::endl;
	}
} gLogger;

YOLOv7TRTDetector::~YOLOv7TRTDetector() {
	// ͬ���������ͷ���Դ
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
}

void YOLOv7TRTDetector::initConfig(std::string enginefile, float conf_thresholod, float score_thresholod) {
	std::ifstream file(enginefile, std::ios::binary);
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

	// ��ʼ����������
	this->runtime = createInferRuntime(gLogger);
	assert(this->runtime != nullptr);
	this->engine = runtime->deserializeCudaEngine(trtModelStream, size);
	assert(this->engine != nullptr);
	this->context = engine->createExecutionContext();
	assert(this->context != nullptr);
	delete[] trtModelStream;

	int input_index = engine->getBindingIndex("images");
	int output_index = engine->getBindingIndex("output");

	// ��ȡ����ά����Ϣ NCHW=1x3x640x640
	this->input_h = engine->getBindingDimensions(input_index).d[2];
	this->input_w = engine->getBindingDimensions(input_index).d[3];
	printf("inputH : %d, inputW: %d \n", input_h, input_w);

	// ��ȡ���ά����Ϣ
	this->output_h = engine->getBindingDimensions(output_index).d[1];
	this->output_w = engine->getBindingDimensions(output_index).d[2];
	std::cout << "out data format: " << output_h << "x" << output_w << std::endl;

	// ����GPU�Դ�����/���������
	std::cout << " input/outpu : " << engine->getNbBindings() << std::endl;
	cudaMalloc(&buffers[input_index], this->input_h * this->input_w * 3 * sizeof(float));
	cudaMalloc(&buffers[1], 19200 *this->output_w * sizeof(float));
	cudaMalloc(&buffers[2], 4800 *this->output_w * sizeof(float));
	cudaMalloc(&buffers[3], 1200 *this->output_w * sizeof(float));
	cudaMalloc(&buffers[output_index], this->output_h *this->output_w * sizeof(float));

	// ������ʱ�������
	prob.resize(output_h * output_w);

	// ����cuda��
	cudaStreamCreate(&stream);
}

void YOLOv7TRTDetector::detect(cv::Mat &frame, std::vector<DetectResult> &results) {
	int64 start = cv::getTickCount();
	// ͼ��Ԥ���� - ��ʽ������
	int w = frame.cols;
	int h = frame.rows;
	int _max = std::max(h, w);
	cv::Mat image = cv::Mat::zeros(cv::Size(_max, _max), CV_8UC3);
	cv::Rect roi(0, 0, w, h);
	frame.copyTo(image(roi));
	// HWC => CHW
	float x_factor = image.cols / static_cast<float>(this->input_w);
	float y_factor = image.rows / static_cast<float>(this->input_h);
	cv::Mat tensor = cv::dnn::blobFromImage(image, 1.0f / 225.f, cv::Size(input_w, input_h), cv::Scalar(), true);

	// �ڴ浽GPU�Դ�
	cudaMemcpyAsync(buffers[0], tensor.ptr<float>(), input_h * input_w * 3 * sizeof(float), cudaMemcpyHostToDevice, stream);

	// ����
	context->enqueueV2(buffers, stream, nullptr);

	// GPU�Դ浽�ڴ�
	cudaMemcpyAsync(prob.data(), buffers[4], output_h *output_w * sizeof(float), cudaMemcpyDeviceToHost, stream);

	// ����
		// ����, 1x25200x85
	std::vector<cv::Rect> boxes;
	std::vector<int> classIds;
	std::vector<float> confidences;
	cv::Mat det_output(output_h, output_w, CV_32F, (float*)prob.data());

	for (int i = 0; i < det_output.rows; i++) {
		float confidence = det_output.at<float>(i, 4);
		if (confidence < this->conf_thresholod) {
			continue;
		}
		cv::Mat classes_scores = det_output.row(i).colRange(5, output_w);
		cv::Point classIdPoint;
		double score;
		minMaxLoc(classes_scores, 0, &score, 0, &classIdPoint);

		// ���Ŷ� 0��1֮��
		if (score > this->score_thresholod)
		{
			float cx = det_output.at<float>(i, 0);
			float cy = det_output.at<float>(i, 1);
			float ow = det_output.at<float>(i, 2);
			float oh = det_output.at<float>(i, 3);
			int x = static_cast<int>((cx - 0.5 * ow) * x_factor);
			int y = static_cast<int>((cy - 0.5 * oh) * y_factor);
			int width = static_cast<int>(ow * x_factor);
			int height = static_cast<int>(oh * y_factor);
			cv::Rect box;
			box.x = x;
			box.y = y;
			box.width = width;
			box.height = height;

			boxes.push_back(box);
			classIds.push_back(classIdPoint.x);
			confidences.push_back(score);
		}
	}

	// NMS
	std::vector<int> indexes;
	cv::dnn::NMSBoxes(boxes, confidences, 0.25, 0.45, indexes);
	for (size_t i = 0; i < indexes.size(); i++) {
		int idx = indexes[i];
		int cid = classIds[idx];
		//cv::rectangle(frame, boxes[idx], Scalar(0, 0, 255), 2, 8, 0);
		//putText(frame, labels[cid].c_str(), boxes[idx].tl(), cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(255, 0, 0), 1, 8);
		DetectResult dr;
		dr.classId = classIds[idx];
		dr.conf = confidences[idx];
		dr.box = boxes[idx];
		cv::rectangle(frame, boxes[idx], cv::Scalar(0, 0, 255), 1, 8);
		cv::rectangle(frame, cv::Point(boxes[idx].tl().x, boxes[idx].tl().y - 20),
			cv::Point(boxes[idx].br().x, boxes[idx].tl().y), cv::Scalar(0, 255, 255), -1);
		results.emplace_back(dr);
	}

	// ����FPS render it
	float t = (cv::getTickCount() - start) / static_cast<float>(cv::getTickFrequency());
	putText(frame, cv::format("FPS: %.2f", 1.0 / t), cv::Point(20, 40), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255, 0, 0), 2, 8);
}
