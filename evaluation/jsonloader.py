# load file jsons and extract information (id, bbox)
import json
from dataloader import DataLoader
import numpy as np
import math
class JsonLoader(object):
    def __init__(self, name_data, name_tracker, scale, width, height, threshold):
        self.dataloader = DataLoader(name_data=name_data, 
                                     name_tracker=name_tracker, 
                                     scale=scale,
                                     width=width,
                                     height=height,
                                     threshold=threshold)
        gt_json_files, res_json_files = self.dataloader()
        self.name_data = name_data
        self.tracker = name_tracker
        self.gt_json_files = gt_json_files
        self.res_json_files = res_json_files
        assert len(self.gt_json_files) == len(self.res_json_files)
    
    def __gt(self, json_dict):
        list_active_frame = []
        for i in json_dict["objects"]:
            list_active_frame.extend(json_dict["objects"][i])
        list_active_frame_matrix = []
        for active_frame in list_active_frame:
            frame_values = list(map(lambda x: int(math.ceil(float(x))), active_frame.values()))
            list_active_frame_matrix.append(frame_values)
        list_active_frame_matrix.sort(key=lambda x: x[0])
        list_active_frame_matrix = np.array(list_active_frame_matrix)

        return list_active_frame_matrix
    
    def __res(self, json_dict):
        info = json_dict["Information"]
        list_active_frame = []
        count_frame = 0
        change_id_dict = {}
        for key, values in list(json_dict.items())[1:]:
            if len(values) > 0:
                if self.tracker.find("arc") != -1:
                    recover_id_dict = values[-1]
                    if bool(recover_id_dict["is_recover"]):
                        change_id_dict[key] = recover_id_dict   
                    values = values[:-1]  

                for v in values:
                    if self.name_data == "MSU_AVIS" and self.tracker.find("deep") != -1:
                        if len(list(v.keys())) == 7:
                            v_values = list(map(lambda x: int(math.ceil(float(x))), list(v.values())[:-2]))  
                        else:
                            v_values = list(map(lambda x: int(math.ceil(float(x))), list(v.values())))     
                    else:
                        v_values = list(map(lambda x: int(math.ceil(float(x))), list(v.values())))
                        
                    list_active_frame.append([int(key)] + v_values)
        list_state_time_since_update = None
        if len(list_active_frame) == 0:
            list_active_frame_matrix = np.full([0, 6], np.nan)
        else:
            list_active_frame_matrix = np.array(list_active_frame)
            if self.tracker.find("arc") != -1 and list_active_frame_matrix.shape[1] == 7:
                list_state_time_since_update = list_active_frame_matrix[:, [2, 3, 4, 5, 0, 1, 6]][:, -3:]
                list_active_frame_matrix = list_active_frame_matrix[:, :-1]
        list_active_frame_matrix = list_active_frame_matrix[:, [0, 2, 3, 4, 5, 1]]
        return list_active_frame_matrix, info, change_id_dict, list_state_time_since_update

    def __get__(self, index):
        assert len(self.gt_json_files) == len(self.res_json_files)
        def proc_json_gt(gt_json):
            
            file_json = open(gt_json, )
            json_dict = json.load(file_json)
            file_json.close()
            return self.__gt(json_dict=json_dict)

        current_gt_json = self.gt_json_files[index]
        gt = proc_json_gt(gt_json=current_gt_json)

        def proc_json_res(res_json):
            file_json = open(res_json, )
            json_dict = json.load(file_json)
            file_json.close()
            return self.__res(json_dict=json_dict)
        current_res_json = self.res_json_files[index]
        res, information_track, recover_id_dict, state_tracking = proc_json_res(res_json=current_res_json)
        return gt, res, information_track, recover_id_dict, state_tracking  
    
    def __len__(self):
        return len(self.gt_json_files)
        
      



