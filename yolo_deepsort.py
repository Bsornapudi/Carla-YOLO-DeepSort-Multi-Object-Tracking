import queue
from typing import Any
import torch
import numpy as np
import cv2
from time import perf_counter
from ultralytics import YOLO
#import yaml
from pathlib import Path
from types import SimpleNamespace
import carla
import random
#import os
from deep_sort.deep_sort import DeepSort
from deep_sort.utils.parser import get_config
#from PIL import Image, ImageDraw, ImageFont

YOLO_PATH = 'weights/yolov8n.pt'
CLASS_IDS = [ 2, 5, 7]
CLASS_NAMES = { 2: 'car', 5: 'bus' ,7: 'truck'}

IM_WIDTH = 256*4
IM_HEIGHT = 256*3
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
output_path = "output.mp4"


class main:
    def __init__(self):
        
        self.model = self.load_model()
        print('after load model')
        self.save_vid = True
        self.output_path = "output.mp4"
        
        self.cfg = get_config()
        self.cfg.merge_from_file('deep_sort/configs/deep_sort.yaml')
        self.deepsort_weights = "deep_sort/deep/checkpoint/ckpt.t7"

        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.deepsort = DeepSort(
            self.deepsort_weights,
            max_age=70
        )
        
    def load_model(self):
        
        model = YOLO(YOLO_PATH)
        return model
    
    def __call__(self):
        # The local Host for carla simulator is 2000
        client = carla.Client('localhost', 2000)
        client.set_timeout(20.0)
        world = client.get_world() 
        
        print('in call')
        # blueprint will access to all blueprints to create objects (vehicles, people, etc.)
        #bp_lib = world.get_blueprint_library()
        spawn_points = world.get_map().get_spawn_points()

        vehicle_bp = world.get_blueprint_library().find('vehicle.tesla.model3')
        vehicle_bp.set_attribute('role_name', 'ego')
        vehicle = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))

        spectator = world.get_spectator()
        #transform = carla.Transform(vehicle.get_transform().transform(carla.Location(x=-4, z=2.5)), vehicle.get_transform().rotation)
        transform = carla.Transform(vehicle.get_transform().transform(carla.Location(x=-4, z=2.5)), carla.Rotation(yaw=-180, pitch=-90))
        
        spectator.set_transform(transform)

        spawn_num = 50

        for i in range(spawn_num):
            vehicle_bp = random.choice(world.get_blueprint_library().filter('vehicle'))
            npc = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))
            
        for v in world.get_actors().filter('*vehicle*'):
            v.set_autopilot(True)
                   
        camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', f'{IM_WIDTH}')
        camera_bp.set_attribute('image_size_y', f'{IM_HEIGHT}')
        camera_bp.set_attribute('fov', '110')
        
        camera_location = carla.Location(2,0,1)
        camera_rotation = carla.Rotation(0,180,0)
         

        camera_init_trans = carla.Transform(camera_location,camera_rotation)
        camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=vehicle , attachment_type=carla.AttachmentType.SpringArm)
        
        def camera_callback(image, data_dict):
            image_data = np.array(image.raw_data)
            image_rgb = image_data.reshape((image.height, image.width, 4))[:, :, :3]
            data_dict['image'] = image_rgb

        image_w = camera_bp.get_attribute('image_size_x').as_int()
        image_h = camera_bp.get_attribute('image_size_y').as_int()

        camera_data = {'image': np.zeros((image_h, image_w, 4))}
        camera.listen(lambda image: camera_callback(image, camera_data))

        vehicle.set_autopilot(True)
        
        fps = 0
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(self.output_path, fourcc, 13.0, (IM_WIDTH, IM_HEIGHT))
        
        vehicle.set_autopilot(True)
        

        while True:
            print('in while loop')
            frame = camera_data['image']
            results = self.model(frame)
        
            bbox_xyxy = []
            conf_score = []
            cls_id = []
            outputs = []
            
            for box in results:  
                for row in box.boxes.data.tolist():
                    x1, y1, x2, y2, conf, id = row
                    
                    if int(id) in CLASS_IDS and id != 0 :
                        bbox_xyxy.append([int(x1), int(y1), int(x2), int(y2)])
                        conf_score.append(conf)
                        cls_id.append(int(id))
                    else:
                            continue        
                outputs = self.deepsort.update(bbox_xyxy, conf_score, frame)
                print('deepsort output' , outputs)
                

            frame = np.array(frame)
            if len(outputs) > 0:
                for j, (output, conf) in enumerate(zip(outputs, conf_score)):
                        frame = main.annotation(self, frame, output, conf, cls_id[j])
  
            frame = cv2.UMat(frame)
            cv2.imshow('deepSORT', frame)
            
            if self.save_vid:
                video_writer.write(frame)
                
            if cv2.waitKey(1) == ord('q'):
                break
            
        cv2.destroyAllWindows()
        camera.stop()
        camera.destroy()
        vehicle.destroy()
        #clear()

    def compute_color_for_labels(self , label):

        color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
        return tuple(color)    
    
    def annotation(self, frame, output, conf, cls_id):
        x1, y1, x2, y2 = map(int,output[0:4])
        id = int(output[4])
        
        label = ''
        if cls_id in CLASS_NAMES:
            label = CLASS_NAMES[cls_id] 
        
        # Convert the frame to a NumPy array (if it's not already)
        frame = frame if isinstance(frame, np.ndarray) else np.array(frame)

        color = self.compute_color_for_labels(id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
        c_id = f'{label} {id}'
        cv2.rectangle(frame, (x1, y1),(x2,y2), color, 1)
        cv2.rectangle(frame, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(frame, c_id, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [255, 255, 255], 2)

        return frame
    
if __name__ == '__main__':
    run = main()