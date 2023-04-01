from os import listdir
from os.path import isfile, join
import pyrealsense2 as rs
import numpy as np
import cv2
import os,time
import posix_ipc

mq_name = "/attraction_color_mq"
mq_max_size = 10000
mq_msg_size = 102400
mq = posix_ipc.MessageQueue(mq_name, flags=os.O_CREAT | os.O_NONBLOCK)


my_path = "New_Images/"
onlyfiles = [f for f in listdir(my_path) if isfile(join(my_path, f))]

#Get the image number from the list of files

def get_img_num():
    arr=[]
    for i in os.listdir(my_path):
        num=i.split("image")[1].split(".")[0]
        arr.append(int(num))

    if not arr:
        return None
    return max(arr)

def take_picture():
    global pipeline

    # Get frameset of color and depth
    #for i in range(75):
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    # Convert images to numpy arrays
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())    

    return color_image

def draw_border(image,x1,y1,x2,y2):
    image_opened=cv2.imread(image)
    drawn=cv2.rectangle(image_opened, (x1, y1), (x2, y2), (0, 255, 0), 5)
    num=get_img_num()
    if not num:
        num=0
    cv2.imwrite("{}image_drawn{}.png".format(my_path,num+1),drawn)

def save_to_remote_pc():
    os.system("scp -r {}* blasha@192.168.0.108:/home/blasha/Desktop/Remote_Images/".format(my_path))

def compare_arrays(base_arr,comp_arr):
    if comp_arr[0]>=base_arr[0] and comp_arr[1]>=base_arr[1] and comp_arr[2]>=base_arr[2]:
        return True
    return False

def recognize_square(color):
    lower_range=[120,35,35]
    upper_range=[220,60,70]

    print(color.shape)
    # as_list=color.tolist()
    in_a_row=0
    highest_in_row=0

    starting_x,starting_y,ending_x,ending_y=0,0,0,0

    total_red_det=0
    total_green_det=0
    total_pixels=0
    all_found={}
    print("color . shape [ 0 ] ", color.shape[0])
    
    for row in range(color.shape[0]): # 480
        if row<=color.shape[0]/2:
                continue
        for col in range(color.shape[1]):
            # Get the RGB values of the current pixel
            #r, g, b = color[row, col]
            #Verified we are checking the bottom half
            b, g, r = color[row, col]
            # if lower_range[0]<=r<=upper_range[0] and lower_range[1]<=g<=upper_range[1] and lower_range[2]<=b<=upper_range[2]:
            total_pixels+=1
            if r>=100 and g<100 and b<100:
                total_red_det+=1
            if r < 100 and g >= 100 and b < 100:
                total_green_det += 1

    print("(RED) {}/{} = {}%".format(total_red_det,total_pixels,round(total_red_det/total_pixels*100,2)))
    print("(GREEN) {}/{} = {}%".format(total_green_det,total_pixels,round(total_green_det/total_pixels*100,2)))

    try:
        if round(total_red_det/total_pixels*100,2) > (round(total_green_det/total_pixels*100,2) + 10):
            mq.send("R")
        elif round(total_green_det/total_pixels*100,2) > (round(total_red_det/total_pixels*100,2) + 10):
            mq.send("G")

    except Exception as e:
        print(f"Error sending message: {e}")

if __name__=="__main__":
    # Create a pipeline
    global pipeline
    pipeline = rs.pipeline()

    # Create a config and configure the pipeline to stream
    # color and depth streams
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    # Start streaming
    pipeline.start(config)

    for i in range(30):
        color=take_picture()
        recognize_square(color)
        #time.sleep(1)

    # Stop streaming
    pipeline.stop()

