# Get mapping from events dataframe generated from motmetrics
from collections import OrderedDict
from utils import get_unique_values_with_dict
import pandas as pd
class MappingId(object):
    '''
        df : acc.events
    '''
    def __init__(self, df, change_id_dict):
        self.raw_mapping = df[df['Type'] == "RAW"]
        self.mapping_id = self.raw_mapping[["OId", "HId"]]
        self.mapping_id.dropna(subset = ["OId", "HId"], inplace=True)
        self.mapping_id_list = sorted(self.mapping_id.to_numpy().tolist(), key=lambda x: x[0])
        self.change_id_dict = change_id_dict
    
    def __percent_re_identification(self, dict_id, key):
        object_id = list(map(lambda x: x[0], dict_id[key]))
        object_id_dict = get_unique_values_with_dict(object_id)
        most_frequently_value = max(object_id_dict.values())  
        max_keys = [k for k, v in object_id_dict.items() if v == most_frequently_value]
        first_mapping_gt = max_keys[0]
        true_identification = 0
        false_indentification = 0
        total_identification = len(object_id)
        for next_id in object_id[1:]:
            if next_id == first_mapping_gt:
                true_identification += 1
            else:
                false_indentification += 1
        return first_mapping_gt, false_indentification, true_identification, total_identification, list(object_id_dict.keys())[0]
    
    def __percent_true_recovery(self, true_id):
        true_recovery = 0
        false_recovery = 0
        for key, value in self.change_id_dict.items():
            list_new_id = list(value.values())
            for each_id in list_new_id:
                if each_id in true_id:
                    true_recovery += 1
                else:
                    false_recovery += 1
        return true_recovery, false_recovery

    def update(self):
        dict_id = OrderedDict()
        for key, *value in self.mapping_id_list:
            dict_id.setdefault(key, []).append(value)
        total_id = list(dict_id.keys())
        list_mapping_gt = []
        id_history = []
        trueid_history = []
        falseid_history = []
        count_migrate = 0
        num_mostly_true = 0
        num_partially_true = 0
        num_mostly_false = 0
        for _id in total_id:
            gt_id, fi, ti, total, _ = self.__percent_re_identification(dict_id, _id)
            # temp_lst.append(temp_)
            if gt_id in list_mapping_gt:
                count_migrate += 1
            else:
                list_mapping_gt.append(gt_id)
            percent_ti = ti / total
            if percent_ti > 0.5:
                num_mostly_true += 1
            elif percent_ti <= 0.5 and percent_ti > 0.2:
                num_partially_true += 1
            else:
                num_mostly_false += 1
            falseid_history.append(fi)
            trueid_history.append(ti)
            id_history.append(total)
        true_recovery, false_recovery = self.__percent_true_recovery(list_mapping_gt)
        summary_result = {
            "fi": sum(falseid_history)/(sum(id_history) + 1e-13),
            "ti": sum(trueid_history)/(sum(id_history) + 1e-13),
            "num_mostly_true": num_mostly_true,
            "num_partially_true": num_partially_true,
            "num_mostly_false": num_mostly_false,
            "num_migrate_tracked_id": count_migrate,
            "num_true_recovery": true_recovery,
            "num_false_recovery": false_recovery
        }
        return summary_result
        


        

        


