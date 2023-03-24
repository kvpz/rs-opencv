import pyrealsense2 as rs
import numpy as np
import cv2
import json
import time

jsonObj = json.load(open("/home/ieeefiu/distancedata.json"))
json_string= str(jsonObj).replace("'", '\"')

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

freq=int(jsonObj["viewer"]['stream-fps'])
width,height = int(jsonObj["viewer"]['stream-width']), int(jsonObj["viewer"]['stream-height'])
print("W:", width)
print("H:", height)
print("FPS:", freq)
config.enable_stream(rs.stream.depth, width, height, rs.format.z16, freq)
config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, freq)
cfg = pipeline.start(config)
dev = cfg.get_device()
advnc_mode = rs.rs400_advanced_mode(dev)
advnc_mode.load_json(json_string)

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        num=75
        frames =[]
        for i in range(num):
          frameset = pipeline.wait_for_frames()
          frames.append(frameset.get_depth_frame())
          # depth_frame = frames.get_depth_frame()
          # color_frame = frames.get_color_frame()

        pipeline.stop()
        print("Frames Captured")

        colorizer = rs.colorizer()
        decimation = rs.decimation_filter()
        spatial = rs.spatial_filter()
        temporal = rs.temporal_filter()
        hole_filling = rs.hole_filling_filter()


        depth_to_disparity = rs.disparity_transform(True)
        disparity_to_depth = rs.disparity_transform(False)

        print("\n")
        for x in range(num):
            if x==num-1:
              frame = frames[x] 

              print(type(frame))

              print(frame.get_distance(424,240))

              frame = decimation.process(frame)

              print(frame.get_distance(424,240))

              frame = depth_to_disparity.process(frame)

              print(frame.get_distance(424,240))

              frame = spatial.process(frame)

              print(frame.get_distance(424,240))

              frame = temporal.process(frame)

              print(frame.get_distance(424,240))

              frame = disparity_to_depth.process(frame)

              print(frame.get_distance(424,240))

              frame = hole_filling.process(frame)

              print(frame.get_distance(424,240))


        colorized_depth = np.asanyarray(colorizer.colorize(frame).get_data())

        
        # color_image = np.asanyarray(color_frame.get_data())


        cv2.imwrite("stacked_images.jpg", colorized_depth)


        break
        

finally:

    # Stop streaming
    try:
      pipeline.stop()
    except Exception as e:
       pass