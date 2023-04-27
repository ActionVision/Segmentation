
import os
import tensorrt as trt
from my_calibration import YOLOEntropyCalibrator
import pycuda.driver as cuda
import pycuda.autoinit


TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
# https://github.com/NVIDIA/TensorRT/issues/1634
# calib_images = "D:/facedb/CrackForest-dataset/image"
# calib_images = "D:/python/yolov5-7.0/uav_bird_training/data/images/train"
calib_images = "D:/python/datasets/dm_training/data/images/train"
onnx_model_path = "dm_best.onnx"


# create tensorrt-engine
def get_engine(onnx_file_path="", engine_file_path="", calibrator=None, save_engine=False):
    with trt.Builder(TRT_LOGGER) as builder, \
            builder.create_builder_config() as config, \
            builder.create_network(1) as network,\
            trt.Runtime(TRT_LOGGER) as runtime, \
            trt.OnnxParser(network, TRT_LOGGER) as parser:

        # parse onnx model file
        if not os.path.exists(onnx_file_path):
            quit('ONNX file {} not found'.format(onnx_file_path))
        print('Loading ONNX file from path {}...'.format(onnx_file_path))
        with open(onnx_file_path, 'rb') as model:
            print('Beginning ONNX file parsing')
            parser.parse(model.read())
            assert network.num_layers > 0, 'Failed to parse ONNX model. \
                        Please check if the ONNX model is compatible '
        print('Completed parsing of ONNX file')
        print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))

        # build trt engine
        builder.max_batch_size = 1
        config.max_workspace_size = 1 << 30
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = calibrator
        print('Int8 mode enabled')
        plan = builder.build_serialized_network(network, config)
        if plan is None:
            print('Failed to create the engine')
            return None
        print("Completed creating the engine")
        engine = runtime.deserialize_cuda_engine(plan)
        if save_engine:
            with open(engine_file_path, "wb") as f:
                f.write(engine.serialize())
        return engine


def run_int8_quantization():
    print('*** onnx to tensorrt int8 engine ***')
    calibration_table = 'int8_calibration.cache'
    # from unet_calibration import UNetEntropyCalibrator
    # calib = UNetEntropyCalibrator(calib_images, (320, 480), "mycalib.cache")
    calib = YOLOEntropyCalibrator(calib_images, (640, 640), "mycalib.cache")
    engine_model_path = "dm_best_int8.engine"
    # try to generate tensor RT quantization engine file
    runtime_engine = get_engine(onnx_model_path, engine_model_path, calib, save_engine=True)
    assert runtime_engine, 'failed engine generation...'
    print('*** success to generate INT8 engine file ***\n')


if __name__ == '__main__':
    run_int8_quantization()
    # files = os.listdir("D:/opencv-4.5.4/opencv/newbuild/install/x64/vc15/lib")
    # for f in files:
    #     print(f)
    
