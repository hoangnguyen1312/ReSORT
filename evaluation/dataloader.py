import json
import glob2

class DataLoader(object):
    def __init__(self, name_data, name_tracker, scale, width, height, threshold):
        self.name_data = name_data
        self.name_tracker = name_tracker
        self.scale = scale
        self.width = width
        self.height = height
        self.threshold=threshold
        self.root = "dataset/{}/".format(name_data)
        self.gt_paths = self.root + "gt/"
        self.res_paths = self.root + "prediction/{}x{}_{}/{}/".format(width, height, scale, name_tracker)
    def __call__(self):
        def read_gt_jsons():
            gt_jsons = glob2.glob(self.gt_paths + "JSON_divide_{}_{}x{}/**/*.json".format(int(self.scale), self.width, self.height))
            return sorted(gt_jsons, key=lambda x: x.split("/")[-1])
        def read_res_jsons():
            res_jsons = glob2.glob(self.res_paths + "**/*.json")
            res_jsons = list(filter(lambda x: x.find("ipynb_checkpoints") == -1, res_jsons))
            return sorted(res_jsons)
        gt_jsons = read_gt_jsons()
        res_jsons = read_res_jsons()
        assert len(gt_jsons) > 0 and len(gt_jsons) == len(res_jsons)
        return gt_jsons, res_jsons
