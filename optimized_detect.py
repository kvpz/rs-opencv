import argparse
import os
import sys
from pathlib import Path
import json
import torch
import signal
import time

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadStreams
from utils.general import (LOGGER, Profile, check_img_size, check_requirements, cv2,
                           non_max_suppression, print_args, scale_boxes)
from utils.torch_utils import select_device, smart_inference_mode
import posix_ipc
import numpy as np

mq_object_name = "/object_detection_queue"
mq_color_name = "/attraction_color_queue"

signal_handler_done=False

max_messages = 20
max_message_size = 40000
mode = 0o666  
flag = posix_ipc.O_CREAT | posix_ipc.O_NONBLOCK | posix_ipc.O_EXCL
# mq = posix_ipc.MessageQueue(mq_name,    flags=os.O_CREAT | os.O_NONBLOCK)

try:
  posix_ipc.unlink_message_queue(mq_object_name)
except Exception as e:
  print(mq_object_name + ": Message Queue doesn't exist: "+str(e))

try:
  posix_ipc.unlink_message_queue(mq_color_name)
except Exception as e:
  print(mq_color_name + ": Message Queue doesn't exist: "+str(e))

try:
    object_mq = posix_ipc.MessageQueue(mq_object_name, flag, mode)
except posix_ipc.ExistentialError:
    object_mq = posix_ipc.MessageQueue(mq_object_name)

try:
    color_mq = posix_ipc.MessageQueue(  mq_color_name, flag, mode)
except posix_ipc.ExistentialError:
    color_mq = posix_ipc.MessageQueue(mq_color_name)

def get_obj_distance(depth_frame, xyxy, obj_name,img):

    x1, y1, x2, y2 = [int(x) for x in xyxy]
    obj_size = 5 # this size is for all cylinders
    color_range2 = None
    if obj_name == 'duck':
        color_range = [[20, 100, 100], [30, 255, 255]] # yellow range
        obj_size = 8
    elif obj_name == 'green cylinder':
        color_range = [[40, 50, 50], [80, 255, 255]] # green range
    elif obj_name == 'red cylinder':
        color_range = [[0, 50, 50], [10, 255, 255]] # red range
        color_range2 = [[170, 50, 50], [180, 255, 255]] # red range
    elif obj_name == 'white cylinder':
        color_range = [[0, 0, 200], [255, 30, 255]] # white range
    elif obj_name == 'pink duck':
        color_range = [[150, 50, 50], [180, 255, 255]] # pink range
        obj_size = 8
    else:
        input('unknown object name')

    h, w = img.shape[:2]

    # Testing print some data
    # print(str(w) + ' ' + str(h))
    # print('obj_name: ' + obj_name + ' ' + str(color_range))

    # Convert the image to the HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define a range of yellow color values in the HSV color space
    lower = np.array(color_range[0])
    upper = np.array(color_range[1])

    if color_range2:
        lower2 = np.array(color_range2[0])
        upper2 = np.array(color_range2[1])
        mask1 = cv2.inRange(hsv, lower, upper)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        mask_color = cv2.bitwise_or(mask1, mask2)
    else:
        mask_color = cv2.inRange(hsv, lower, upper)

    # Create a mask based on the defined color range
    color_filtered = cv2.bitwise_and(img, img, mask=mask_color)

    mask_range = np.zeros((h,w), np.uint8)
    mask_range[y1:y2:, x1:x2] = 1

    filtered = cv2.bitwise_and(color_filtered, color_filtered, mask=mask_range)

    # Convert the filtered image to grayscale
    gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)

    # Show Image for testing
    # cv2.imshow('Image', hsv)
    # # Wait for a key press and then close the window
    # cv2.waitKey(1)

    # Find the indices of the non-zero elements in the grayscale image
    y_indices, x_indices = np.where(gray > 0) 
    
    coordinates = np.column_stack((x_indices, y_indices))

    # Get the distances for all pixels in the innermost rectangle
    # * 100 to go from mm to cm
    distances = [depth_frame.get_distance(x, y)*100 for (x, y) in coordinates]

    # Filter out any NaN values
    distances = [d for d in distances if not np.isnan(d) and d > 0]
    if not distances:
        return 0
    
    min_distance = min(distances)

    distances = [d for d in distances if d < min_distance + obj_size]

    # mean = sum(distances) / len(distances)
    # variance = sum((d - mean) ** 2 for d in distances) / (len(distances) - 1)
    # std_dev = variance ** 0.5

    # num=1.5

    # avg_distance = [d for d in distances if mean - num * std_dev <= d <= mean + num * std_dev]
    avg_distance = distances

    # Calculate the average distance
    avg_distance = sum(avg_distance) / len(avg_distance)

    
    return avg_distance

def color_percent(color_info):
    total_red_det=0
    total_green_det=0
    total_pixels=0
    
    for row in range(color_info.shape[0]): # 480
        if row<=color_info.shape[0]/2:
                continue
        for col in range(color_info.shape[1]):
            # Get the RGB values of the current pixel
            #r, g, b = color[row, col]
            #Verified we are checking the bottom half
            b, g, r = color_info[row, col]
            
            total_pixels+=1
            if r>=100 and g<100 and b<100:
                total_red_det+=1
            if r < 100 and g >= 100 and b < 100:
                total_green_det += 1

    red_num = round(total_red_det/total_pixels*100,2)
    green_num = round(total_green_det/total_pixels*100,2)

    color = ''
    print(red_num,green_num)
    if  (red_num >  green_num + 10):
        color = "R".encode()
        print("R")
    elif (green_num > red_num + 10):
        color = "G".encode()
        print("G")

    if color:
        try:
            color_mq.send(color)
        except posix_ipc.BusyError:
            # The queue is full, retrieve the first message
            m = color_mq.receive()
            # Add the new message
            color_mq.send(color)


@smart_inference_mode()
def run(
        weights=ROOT / 'best_small.onnx',  # model path or triton URL
        source=ROOT / 'rs',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(848, 480),  # inference size (height, width)
        conf_thres=0.40,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=30,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    global signal_handler_done
    source = str(source)

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)

    if not signal_handler_done:
        signal_handler_done=True
        def signal_handler(sig, frame):

            # Catch Ctrl+C and stop the pipeline
            dataset.pipeline.stop()
            print('Pipeline stopped')
            exit(0)

        signal.signal(signal.SIGINT, signal_handler)


    bs = len(dataset)
    
    
    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup

    _, _, dt = 0, [], (Profile(), Profile(), Profile())
    for _, im, im0s, depth, _, _ in dataset:
        # print(str(type(im0s[0])) + ' : ' + str(type(depth[0])))
        # print(str(len(im0s[0])) + ' : ' + str(len(depth[0])))
        # input()
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            pred = model(im, augment=augment, visualize=False)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Process predictions

        frames = dataset.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        # # Convert images to numpy arrays
        # color_image = np.asanyarray(color_frame.get_data())

        # color_percent(color_image)

        json_data = {}
        detected_objects = []
        for i, det in enumerate(pred):  # per image
            im0 = im0s[i].copy()         

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Write results
                detected_objects = [{
                "object": names[int(cls)],
                "x1": int(xyxy[0]),
                "y1": int(xyxy[1]),
                "x2": int(xyxy[2]),
                "y2": int(xyxy[3]),
                "inference_time": "{}ms".format(round(float(dt[1].dt) * float(10**3),3)),
                "distance": round(get_obj_distance(depth[i], xyxy,names[int(cls)],im0s[0].copy()), 2),
                } for *xyxy, _, cls in reversed(det) if xyxy[1] > 24]

                #this for the distance: round(get_obj_distance(depth[i], xyxy,25), 2)
                
                json_data = json.dumps(detected_objects)
                print(json_data)
                
            if json_data:
                try:
                    object_mq.send(json_data)
                except posix_ipc.BusyError:
                    # The queue is full, retrieve the first message
                    m = object_mq.receive()
                    # Add the new message
                    object_mq.send(json_data)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'data/best_small.onnx', help='model path or triton URL')
    parser.add_argument('--source', type=str, default='rs', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.75, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
