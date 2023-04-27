import tensorrt as trt
from torchvision import transforms
import torch as t
from collections import OrderedDict, namedtuple
import cv2 as cv
import time
import numpy as np

img_transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((640, 640))
                                    ])

def load_classes():
    with open("uav_bird.txt", "r") as f:
        class_list = [cname.strip() for cname in f.readlines()]
    return class_list


def format_yolov8(frame):
    row, col, _ = frame.shape
    _max = max(col, row)
    result = np.zeros((_max, _max, 3), np.uint8)
    result[0:row, 0:col] = frame
    result = cv.cvtColor(result, cv.COLOR_BGR2RGB)
    return result

def wrap_detection(input_image, output_data):
    class_ids = []
    confidences = []
    boxes = []
    out_data = output_data.T
    rows = out_data.shape[0]

    image_width, image_height, _ = input_image.shape

    x_factor = image_width / 640.0
    y_factor = image_height / 640.0

    for r in range(rows):
        row = out_data[r]
        classes_scores = row[4:]
        class_id = np.argmax(classes_scores)
        if (classes_scores[class_id] > .25):
            class_ids.append(class_id)
            confidences.append(classes_scores[class_id])
            x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item()
            left = int((x - 0.5 * w) * x_factor)
            top = int((y - 0.5 * h) * y_factor)
            width = int(w * x_factor)
            height = int(h * y_factor)
            box = np.array([left, top, width, height])
            boxes.append(box)

    indexes = cv.dnn.NMSBoxes(boxes, confidences, 0.25, 0.25)

    result_class_ids = []
    result_confidences = []
    result_boxes = []

    for i in indexes:
        result_confidences.append(confidences[i])
        result_class_ids.append(class_ids[i])
        result_boxes.append(boxes[i])

    return result_class_ids, result_confidences, result_boxes

def gpu_trt_demo():
    class_list = load_classes()
    device = t.device('cuda:0')
    Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
    logger = trt.Logger(trt.Logger.INFO)
    with open("drone_best_16.engine", 'rb') as f, trt.Runtime(logger) as runtime:
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

    capture = cv.VideoCapture("D:/bird_test/drone.mp4")
    colors = [(255, 255, 0), (0, 255, 0), (0, 255, 255), (255, 0, 0)]
    while True:
        _, frame = capture.read()
        if frame is None:
            print("End of stream")
            break
        fh, fw, fc = frame.shape
        start = time.time()
        image = format_yolov8(frame)
        x_input = img_transform(image).view(1, 3, 640, 640).to(device)
        binding_addrs['images'] = int(x_input.data_ptr())
        context.execute_v2(list(binding_addrs.values()))
        out_prob = bindings['output0'].data.cpu().numpy()
        end = time.time()

        class_ids, confidences, boxes = wrap_detection(image, np.squeeze(out_prob, 0))
        for (classid, confidence, box) in zip(class_ids, confidences, boxes):
            if box[2] > fw * 0.67:
                continue
            color = colors[int(classid) % len(colors)]
            cv.rectangle(frame, box, color, 2)
            cv.rectangle(frame, (box[0], box[1] - 20), (box[0] + box[2], box[1]), color, -1)
            cv.putText(frame, class_list[classid] + " " + ("%.2f"%confidence), (box[0], box[1] - 10), cv.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0))

        inf_end = end - start
        fps = 1 / inf_end
        fps_label = "FPS: %.2f" % fps
        cv.putText(frame, fps_label, (10, 25), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv.imshow("YOLOv8 + TensorRT8.6.x Object Detection", frame)
        cc = cv.waitKey(1)
        if cc == 27:
            break
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    gpu_trt_demo()