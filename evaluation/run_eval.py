from jsonloader import JsonLoader
import pandas as pd
from tqdm import tqdm
import numpy as np

from reid_metric import MappingId
from pymot_metric import MotMetric
from collections import OrderedDict
from utils import get_unique_values_with_dict, get_number_not_ascend, save_result
import argparse
class Eval(object):
    def __init__(self, name_data, 
                       name_tracker, 
                       scale,
                       width,
                       height,
                       threshold,
                       pymot_metric=MotMetric,
                       reid_metric=MappingId
                       ):
        self.name_data = name_data
        self.name_tracker = name_tracker
        self.scale = scale
        self.width = width
        self.height = height
        self.threshold = threshold
        self.pymot_metric = pymot_metric
        self.reid_metric=MappingId
        self.jsonloader = JsonLoader(name_data=self.name_data, 
                                     name_tracker=self.name_tracker, 
                                     scale=self.scale,
                                     width=self.width,
                                     height=self.height,
                                     threshold=self.threshold)
        self.total_gt=0
        self.total_inferentime=0 
        self.total_exec_time=0 
        self.total_frame=0
        self.total_frame_gt = []
        self.total_frame_res = []
        self.change_recover_id = {}
        self.total_state_tracking = [] # useless
        self.official_metrics = {}
    
    def __proc_json(self):
        '''
        process dataset from json files
        '''
        len_frame_history = 0
        for json_index_file in range(self.jsonloader.__len__()):
            # print(json_index_file)
            gt_mat, res_mat, info, dict_change_id, state_tracking = self.jsonloader.__get__(json_index_file)    
            # print(gt_mat.shape, res_mat.shape)
            if gt_mat.shape[0] == 0:
                continue
            lst_frame_gt = sorted(list(set(gt_mat[:, 0].tolist())))  
            gt_mat[:, 0] += len_frame_history
            # print(gt_mat[:, 0])
            self.total_frame_gt.extend(gt_mat.tolist())
            
            lst_frame_res = sorted(list(set(res_mat[:, 0].tolist()))) if res_mat.shape[0] > 0 else []
            res_mat[:, 0] += len_frame_history
            self.total_frame_res.extend(res_mat.tolist())
            
            # useless
            # if state_tracking is not None:
            #     state_tracking[:, 0] += len_frame_history
            #     self.total_state_tracking.extend(state_tracking.tolist())
            
            self.total_frame += int(info["total_frame"])
            self.total_inferentime += info["inference_time"]
            self.total_exec_time += info["execution_time"]
            if bool(dict_change_id):
                for key, value in dict_change_id.items():
                    cur_frame = int(key)
                    cur_frame += len_frame_history
                    self.change_recover_id[cur_frame] = value["is_recover"]
            len_frame_history += max(len(lst_frame_gt), len(lst_frame_res)) 
            # if json_index_file == 1:
            #     break
    def __proc_change_id_dict(self):
        '''
        apply change_recover_id into dataset
        '''
        self.res_mat = np.array(self.total_frame_res)
        for cur_frame, change_id_dict in self.change_recover_id.items():
            for new_id, recover_id in change_id_dict.items():
                new_id = int(new_id)
                recover_id = int(recover_id)
                row_mask_frame = self.res_mat[:, 0] < cur_frame
                row_mask_id = self.res_mat[:, -1] == new_id
                self.res_mat[np.logical_and(row_mask_frame, row_mask_id), -1:] = recover_id
        self.gt_mat = np.array(self.total_frame_gt)
        # no process for gt_mat

    def __proc_pymot_metric(self):
        '''
        calculate pymotmetric
        '''
        mot_metric = self.pymot_metric(gt_mat=self.gt_mat,
                                       res_mat=self.res_mat,
                                       data_name=self.name_data
                                      )
        # print(self.res_mat.shape)
        for frame_id in tqdm(range(self.total_frame)):
            mot_metric.__get__(index_frame=frame_id)

        self.events_tracking = mot_metric.mot_events()
        mot_switching_raw, frame_event = mot_metric.mot_switching_raw()

        mot_matching_raw = mot_metric.mot_matching_first_raw()
        list_map = sorted(mot_matching_raw.to_numpy().tolist(), key=lambda x: x[0])
        first_gt_dict = {}
        for each_tuple in list_map:
            if each_tuple[0] in first_gt_dict:
                continue
            else:
                first_gt_dict[each_tuple[0]] = each_tuple[1]
        # print(first_gt_dict)
        switch_map = mot_switching_raw.to_numpy()
        frame_id = np.asarray(list(map(lambda x: [x[0]], frame_event)))
        switch_map = np.append(switch_map, frame_id, axis=1)
        switch_map = sorted(switch_map.tolist(), key=lambda x:x[0])
        dict_id = OrderedDict()
        for key, *value in switch_map:
            dict_id.setdefault(key, []).append(value)
        total_id = list(dict_id.keys())
        new_num_switches = 0
        # useless
        new_num_miss_recovered_id = 0
        for gt_id in total_id:
            list_switched_ids = list(map(lambda x: x[0], dict_id[gt_id]))

            first_gt_value = first_gt_dict[gt_id]
            list_frame = list(map(lambda x: x[1], dict_id[gt_id]))
            dict_switched_ids = list(get_unique_values_with_dict(list_switched_ids).keys())
            # bug 
            # bool_switched_ids = get_number_not_ascend(dict_switched_ids)
            # print(first_gt_value, dict_switched_ids)
            if first_gt_value in dict_switched_ids:
                # new_num_switch_id = len(dict_switched_ids) - int(bool_switched_ids)
                new_num_switch_id = len(dict_switched_ids) - 1
            else:
                new_num_switch_id = len(dict_switched_ids)

            new_num_switches += new_num_switch_id
        
        pymot_metric_dict = mot_metric.update()
        pymot_metric_dict["num_groundtruth"] = self.gt_mat.shape[0]
        pymot_metric_dict["inference_time"] = self.total_inferentime
        pymot_metric_dict["execution_time"] = self.total_exec_time
        pymot_metric_dict["num_switches"] = new_num_switches + pymot_metric_dict["num_transfer"]
        
        pymot_metric_dict["mota"] = 1 - (pymot_metric_dict["num_false_positives"]+pymot_metric_dict["num_misses"]+pymot_metric_dict["num_switches"])/(pymot_metric_dict["num_groundtruth"])
        print("=====================================")
        print(self.total_frame/pymot_metric_dict["execution_time"])
        print("=====================================")
        self.official_metrics.update(pymot_metric_dict)
    
    def __proc_reid_metric(self):
        # self.events_tracking.to_csv("resort_chokepoint.csv")
        id_metric = self.reid_metric(self.events_tracking, self.change_recover_id)
        id_metric_dict = id_metric.update()
        self.official_metrics.update(id_metric_dict)
    
    def __call__(self):
        self.__proc_json()
        self.__proc_change_id_dict()
        self.__proc_pymot_metric()
        self.__proc_reid_metric()
        name_metrics=[
            "recall",
            "precision",
            "mota",
            "motp",
            "mostly_tracked",
            "partially_tracked",
            "mostly_lost",
            "num_switches",
            "num_fragmentations",
            "num_false_positives",
            "num_misses",
            "num_ascend",
            "num_transfer",
            "num_migrate",
            "inference_time",
            "execution_time",
            "num_groundtruth",
            "fi",
            "ti",
            "num_mostly_true",
            "num_partially_true",
            "num_mostly_false",
            "num_migrate_tracked_id",
            "num_true_recovery",
            "num_false_recovery"
        ]
        df_official_metrics = pd.DataFrame({})
        index_column = 0
        for each_metric_name in name_metrics:    
            df_official_metrics.insert(index_column, each_metric_name, [self.official_metrics[each_metric_name]])
            index_column += 1
        print(self.total_frame)
        print(self.total_exec_time)
        print(self.total_inferentime)
        print(df_official_metrics.head())
        '''
        Save result into csv
        '''
        save_result(name_data=self.name_data, 
                    name_tracker=self.name_tracker, 
                    scale=self.scale,
                    width=self.width,
                    height=self.height, 
                    threshold=self.threshold,
                    final_metric=df_official_metrics)
               
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Conduct to evaluate')
    parser.add_argument('--data_name', default='Chokepoint', type=str, help='Access dataset folder')
    parser.add_argument('--tracker', default="sort", type=str, help='Access dataset folder')
    parser.add_argument('--scale', default=1, type=float, help='Access dataset folder')
    parser.add_argument('--width', type=int, help='Access dataset folder')
    parser.add_argument('--height', type=int, help='Access dataset folder')
    parser.add_argument('--threshold', default=1, type=int, help='Access dataset folder')
    args = parser.parse_args()
    name_data = args.data_name
    name_tracker = args.tracker
    scale = args.scale
    width = args.width
    height = args.height
    threshold = args.threshold

    eval_metrics = Eval(name_data, name_tracker, scale, width, height, threshold)
    eval_metrics()