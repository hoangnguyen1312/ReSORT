import os
import argparse
import numpy as np
import cv2
from tracker.resort.resort import ReSort
from tracker.sort.sort import Sort
from tracker.deepsort.deepsort import DeepSort
from detector.detector import RetinafaceDetector
from tracker.visualization import *
import glob2
from tqdm import tqdm
import os
import json
import timeit
from utils.lib_chokepoint import load_images_chokepoint
parser = argparse.ArgumentParser(description='Retinaface')

parser.add_argument('--tracker', default='sort', help='Tracker: sort, deepsort, arcsort, opticalflow')
parser.add_argument('--network', default='resnet50', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('--resize', default=1, type=int, help='resize')
parser.add_argument('-s', '--save_image', action="store_true", default=True, help='show detection results')
parser.add_argument('--vis_thres', default=0.6, type=float, help='visualization_threshold')
parser.add_argument('--dataset_name', default="Chokepoint", type=str, help='name dataset')
parser.add_argument('--w_new', type=int, help='width')
parser.add_argument('--h_new', type=int, help='height')
parser.add_argument('--scale', default=1)
parser.add_argument('--threshold', default=0.44)
args = parser.parse_args()

if args.dataset_name == "Chokepoint":
    lst_subfolders = sorted(glob2.glob("dataset/{}/*/".format(args.dataset_name)))
path = "dataset/Chokepoint/chokepoint"

lst_subfolders = load_images_chokepoint(path)
def load_tracker(tracker, max_age=1, min_hits=3, iou_threshold=0.3):
    mot_tracker = None
    if tracker == 'sort':
        mot_tracker = Sort(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold)
    elif tracker == 'resort':
        mot_tracker = ReSort(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold, sim_threshold=float(args.threshold), w=args.w_new, h=args.h_new, scale=args.scale)
    elif tracker == 'deepsort':
        mot_tracker = DeepSort()
    return mot_tracker

if __name__ == '__main__':
    w_thresh = 20
    h_thresh = 20
    max_age = 1
    min_hits = 3
    iou_threshold = 0.3
    detector = RetinafaceDetector(args.network, args.cpu)
    mot_tracker = load_tracker(args.tracker, max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold)
    for idx__, subfolder in enumerate(lst_subfolders[1:4]):
        print("========================================")
        print("Access all frames in folder {}".format("/".join(subfolder[0].split("/")[:-1])))
        print("========================================")

        result_dict = {}
        result_dict["Information"] = {}
        total_time = 0
        time_tracker_inference = 0
        frame_rate = 1
        for i in tqdm(range(len(subfolder))):
            frame = subfolder[i]
            count_frame = int(frame.split("/")[-1][:-4])
            frame_rate = count_frame
            converted_frame_id = -1
            is_visualize = False
            fps_scale = float(args.scale) 
            args.scale = float(args.scale) 
            if  (frame_rate + 1) % fps_scale == 0:
                converted_frame_id = int((frame_rate + 1)/ fps_scale) - 1
                is_visualize = True
            else:
                if not fps_scale.is_integer():
                    mean = (frame_rate + 1) / fps_scale
                    if int(fps_scale*int(mean)) == (frame_rate + 1):
                        converted_frame_id = int(mean) - 1
                        is_visualize = True
                    elif int(fps_scale * int(mean + 1)) == (frame_rate + 1):
                        converted_frame_id = int(mean)
                        is_visualize = True
            img_raw = cv2.imread(frame)
#             ret, img_raw = cap.read()
            if is_visualize == True:
                count_frame = converted_frame_id 
                result_dict[count_frame] = []
#                 if ret == True:
                if args.w_new != None and args.h_new != None:
                    img_raw = cv2.resize(img_raw, (args.w_new, args.h_new))
                start = timeit.default_timer()
                dets = detector.detect_faces(img_raw, confidence_threshold=args.confidence_threshold, top_k=args.top_k, 
                                    nms_threshold=args.nms_threshold, keep_top_k=args.keep_top_k, resize=args.resize) 
                if args.tracker == 'sort':
                    detections, facial_landmarks = convert_detection_to_tracker(dets, args.vis_thres, w_thresh, h_thresh)
                    start_tracker = timeit.default_timer()
                    trackers = mot_tracker.update(np.array(detections))
                    # draw trackers
                    # image_raw = draw_box_and_landmarks_sort_trackers(img_raw, trackers)
                    stop = timeit.default_timer()
                    execution_time = stop - start
                    exec_inference_time = stop - start_tracker
                    time_tracker_inference += exec_inference_time
                    total_time += execution_time
                    for t in trackers:
                        result_dict[count_frame].append({
                            "id": t[4],
                            "xtl": t[0],
                            "ytl": t[1],
                            "xbr": t[2],
                            "ybr": t[3]
                        })
                elif args.tracker == 'resort':
                    detections, facial_landmarks = convert_detection_to_tracker(dets, args.vis_thres, w_thresh, h_thresh)
                    start_tracker = timeit.default_timer()
                    # print(detections)
                    trackers, change_id_dict = mot_tracker.update(np.asarray(detections), facial_landmarks, img_raw)
                    stop = timeit.default_timer()
                    execution_time = stop - start
                    exec_inference_time = stop - start_tracker
                    time_tracker_inference += exec_inference_time
                    total_time += execution_time
                    for t in trackers:
                        result_dict[count_frame].append({
                            "id": t[4],
                            "xtl": t[0],
                            "ytl": t[1],
                            "xbr": t[2],
                            "ybr": t[3]
                        })
                    result_dict[count_frame].append({"is_recover": change_id_dict})
                elif args.tracker == 'deepsort':
                    detections, facial_landmarks = convert_detection_to_tracker(dets, args.vis_thres, w_thresh, h_thresh)
                    if len(detections) > 0:
                        detections = np.asarray(detections)
                        detections, scores = detections[:, :-1], detections[:, -1]
                        out_scores = scores.reshape(-1, 1)
                        detections = np.array(detections)
                        out_scores = np.array(out_scores) 

                        start_tracker = timeit.default_timer()
                        trackers, detections_class = mot_tracker.run_deep_sort(img_raw, out_scores, detections, facial_landmarks)

                        stop = timeit.default_timer()
                        execution_time = stop - start
                        exec_inference_time = stop - start_tracker
                        time_tracker_inference += exec_inference_time
                        total_time += execution_time
                        if trackers is None or detections_class is None:
                            continue
                        for track in trackers.tracks:
                            if not track.is_confirmed() or track.time_since_update > 1:
                                continue
                            bbox = track.to_tlbr() 
                            id_num = str(track.track_id) 
                            features = track.features 
                            result_dict[count_frame].append({
                                "id": id_num,
                                "xtl": bbox[0],
                                "ytl": bbox[1],
                                "xbr": bbox[2],
                                "ybr": bbox[3]
                            })      

        import os
        if args.w_new != None and args.h_new != None: 
            root = '{}/{}_{}x{}_{}_{}'.format(args.dataset_name, args.tracker, args.w_new, args.h_new, args.scale, args.threshold)
        else:
            root = '{}/{}_original_resolution_{}_{}'.format(args.dataset_name, args.tracker, args.scale, args.threshold)
        if not os.path.exists(root):
            print("[+] Creating new output folder: ", root)
            os.makedirs(root)
        result_dict["Information"]["total_frame"] = int(len(subfolder)/ args.scale)
        result_dict["Information"]["inference_time"] = time_tracker_inference
        result_dict["Information"]["execution_time"] = total_time
        with open(root + '/{}.json'.format(subfolder[0].split("/")[-2]), 'w') as outfile:
                json.dump(result_dict, outfile)
    print()
