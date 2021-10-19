# using motmetrics 
import motmetrics as mm
import numpy as np
class MotMetric(object):
    def __init__(self, gt_mat,
                       res_mat,
                       data_name, 
                       mot_metric=mm.MOTAccumulator(auto_id = True), 
                       iou_thresh=0.5
                ):
        self.mot_metric = mot_metric
        self.iou_thresh = iou_thresh
        self.data_name = data_name
        self.gt_mat = gt_mat
        self.res_mat = res_mat


    def __extract_bbox_coors(self, cur_gt_frame, cur_res_frame):
        objects_gt = cur_gt_frame[:, -1].reshape(-1, ).tolist()
        objects_res = cur_res_frame[:, -1].reshape(-1, ).tolist()
        bbox_coors_gt_frame = cur_gt_frame[:, 0:-1]
        bbox_coors_gt_frame[:, 2] -= bbox_coors_gt_frame[:, 0]
        bbox_coors_gt_frame[:, 3] -= bbox_coors_gt_frame[:, 1]
        bbox_coors_res_frame = cur_res_frame[:, 0:-1]
        bbox_coors_res_frame[:, 2] -= bbox_coors_res_frame[:, 0]
        bbox_coors_res_frame[:, 3] -= bbox_coors_res_frame[:, 1]
        return objects_gt, bbox_coors_gt_frame, objects_res, bbox_coors_res_frame 

    def __get__(self, index_frame):
        
        cur_gt_frame = self.gt_mat[self.gt_mat[:, 0] == index_frame, 1:] if self.gt_mat.shape[0] > 0 else np.random.rand(0, 5)
        cur_res_frame = self.res_mat[self.res_mat[:, 0] == index_frame, 1:] if self.res_mat.shape[0] > 0 else np.random.rand(0, 5)
        
        o1, coor1, o2, coor2 = self.__extract_bbox_coors(cur_gt_frame, cur_res_frame)
        iou_cost = mm.distances.iou_matrix(coor1, coor2, max_iou = self.iou_thresh)

        self.mot_metric.update(
            o1,                     
            o2,                 
            iou_cost
        )
    def update(self):
        mh = mm.metrics.create()
        summary = mh.compute(self.mot_metric, 
                             metrics = mm.metrics.motchallenge_metrics, 
                             name = self.data_name
                             )
        summary = summary.to_dict()
        for key, value in summary.items():
            summary[key] = value[self.data_name]
        return summary

    def mot_events(self):
        return self.mot_metric.events
        
    def mot_switching_raw(self):
        df = self.mot_metric.mot_events
        switch_events = df[df['Type'] == "SWITCH"]
        frame_event = list(switch_events.index)
        mapping_id = switch_events[["OId", "HId"]]
        return mapping_id, frame_event
    
    def mot_matching_first_raw(self):
        df = self.mot_metric.mot_events
        matching_events = df[df['Type'] == "MATCH"]
        mapping_id = matching_events[["OId", "HId"]]
        return mapping_id
