import tensorrt as trt
from torchvision import transforms
import torch as t
from collections import OrderedDict, namedtuple
import cv2 as cv
import time
import numpy as np

img_transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((320, 480))
                                    ])


def gpu_trt_demo():
    device = t.device('cuda:0')
    Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
    logger = trt.Logger(trt.Logger.INFO)
    with open("unet_road_16.engine", 'rb') as f, trt.Runtime(logger) as runtime:
        model = runtime.deserialize_cuda_engine(f.read())

    bindings = OrderedDict()
    for index in range(model.num_bindings):
        name = model.get_binding_name(index)
        dtype = trt.nptype(model.get_binding_dtype(index))
        shape = model.get_binding_shape(index)
        data = t.from_numpy(np.empty(shape, dtype=np.dtype(dtype))).to(device)
        bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))
    binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
    context = model.create_execution_context()

    frame = cv.imread("D:/images/kss1.jpg")
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    print(gray.shape)
    fh, fw, fc = frame.shape
    inf_start = time.time()
    x_input = img_transform(gray).view(1, 1, 320, 480).to(device)
    binding_addrs['input.1'] = int(x_input.data_ptr())
    context.execute_v2(list(binding_addrs.values()))

    out_prob = bindings['203'].data.cpu().numpy()
    predic_ = np.argmax(out_prob, axis=1)
    output = np.squeeze(predic_, 0)
    output = cv.resize(np.uint8(output), (fw, fh))
    output[output > 0] = 255
    cv.imshow("output", output)
    contours, h = cv.findContours(output, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    cv.drawContours(frame, contours, -1, (0, 0, 255), -1)
    inf_end = time.time() - inf_start
    cv.putText(frame, "infer time(ms):%.3f" % (inf_end * 1000), (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255),
               2)
    cv.imshow("UNet + TensorRT8.6.x", frame)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    gpu_trt_demo()