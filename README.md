# ReSORT: an ID-recovery multi-face tracking method for surveillance cameras

### 01. Setup 
```
sudo chmod u+x detector/install.sh
sudo chmod u+x tracker/extractor/install.sh
bash install.sh
```

### 02. Run demo and visualize results
```
python demo.py --video <PATH OF VIDEO> \
               --tracker <TRACKER> \
               --network "resnet50" \
               --cpu False \
               --confidence_threshold 0.02 \
               --top_k 5000 \ 
               --nms_threshold 0.4 \
               --keep_top_k 750 \
               --resize 1 \
               --scale 1 \
               --save_video True \
               --vis_thres 0.8
```
--video : Path to the video. Default: "data/multi_cam.mp4" \n
--tracker: Select "sort", "resort" or "deepsort". Default: "sort" \n
--network: Backbone network "mobile0.25" or "resnet50". Default: "resnet50" \n
--scale: Scaling of the FPS counter. Default: 1 \n
E.g: new_fps = old_fps / scale \n
--save_video: Return output which is a video named "output.avi". Default: True \n
