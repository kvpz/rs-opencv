# YOLOv5 🚀 by Ultralytics, GPL-3.0 license

import argparse
import os
import platform
import sys
from pathlib import Path
import numpy
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
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode
import posix_ipc
import numpy as np

mq_name = "/object_detection_queue"

signal_handler_done=False

mq = posix_ipc.MessageQueue(mq_name, flags=os.O_CREAT | os.O_NONBLOCK)

once=0

def get_obj_distance(depth_frame, xyxy, innermost_pixels,obj_name,img):
    
    global once
    once+=1
    x1, y1, x2, y2 = xyxy
    # w = (x1+x2)//2
    # h = (y1+y2)//2

    # # # Define the top-left corner of the innermost rectangle
    # x_start = ((x1 + w) // 2) - (innermost_pixels // 2)
    # y_start = ((y1 + h) // 2) - (innermost_pixels // 2)


    # Convert the image to the HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define a range of yellow color values in the HSV color space
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Create a mask based on the defined color range
    filtered = cv2.bitwise_and(img, img, mask=mask)

    # Convert the filtered image to grayscale
    gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)

    # Find the indices of the non-zero elements in the grayscale image
    y_indices, x_indices = np.where(gray > 0)

    # Convert the (x, y) coordinates to pixel coordinates in the depth frame
    # depth_scale = depth_frame.get_depth_scale()
    coordinates = np.column_stack((x_indices, y_indices))

    # Get the distances for all pixels in the innermost rectangle
    distances = [depth_frame.get_distance(x, y) for (x, y) in coordinates]
    # for (x,y) in coordinates:
    #     print(x,y)

    # print(distances[0:5])

    # Filter out any NaN values
    distances = [d for d in distances if not np.isnan(d)]
    # print(len(distances))

    if not distances:
        return 0

    mean = sum(distances) / len(distances)
    variance = sum((d - mean) ** 2 for d in distances) / (len(distances) - 1)
    std_dev = variance ** 0.5

    num=1.5

    # avg_distance = [d for d in distances if mean - num * std_dev <= d <= mean + num * std_dev]
    avg_distance = distances

    # print(len(avg_distance))

    # print("here")

    # Calculate the average distance
    avg_distance = sum(avg_distance) / len(avg_distance)

    # try:
    # if obj_name == "duck":
    #     num = 0.2
    # else:
    #     num = 0.079
    # avg_distance = depth_frame.get_distance(x1 + (x2-x1) // 2, y1 + (y2 - y1) // 2)
    # avg_distance = avg_distance**2-num**2
    # if avg_distance<0:
    #     return 0

    # avg_distance = avg_distance**0.5

    # print(color_frame[x1 + (x2-x1) // 2][y1 + (y2 - y1) // 2])
    # if avg_distance != 0:
    #     avg_distance = avg_distance**2-0.11**2
    #     avg_distance = avg_distance**0.5
    # except:
    #     avg_distance = 0

    return avg_distance

@smart_inference_mode()
def run(
        weights=ROOT / 'best_small.onnx',  # model path or triton URL
        source=ROOT / 'rs',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.40,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=30,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    global signal_handler_done
    source = str(source)

    print("\nup here mate\n")

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
                "distance": round(get_obj_distance(depth[i], xyxy,25,names[int(cls)],im0s[0].copy()) * 100 , 2),
                } if xyxy[1] > 24 else None for *xyxy, _, cls in reversed(det)]

                #this for the distance: round(get_obj_distance(depth[i], xyxy,25), 2)
                
                json_data = json.dumps(detected_objects)
                        
            print(json_data)
            time.sleep(3)
            # if detected_objects:
            #     print(len(detected_objects))
            # else:
            #     print(0)
            # print("\n\n")
            # num_messages = mq.current_messages
            if json_data: #and num_messages<10: # send message to MQ if message string is not empty
                try:
                    mq.send(json_data)
                except posix_ipc.BusyError:
                    # The queue is full, retrieve the first message
                    m = mq.receive()
                    # Add the new message
                    # mq.send(json_data)
                    # print (m)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'best_small.onnx', help='model path or triton URL')
    parser.add_argument('--source', type=str, default='rs', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.75, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
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
