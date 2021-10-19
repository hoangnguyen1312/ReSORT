import os 

def get_unique_values_with_dict(values):
    new_values = {}
    for value in values:
        if value in new_values.keys():
            new_values[value] += 1
        else:
            new_values[value] = 1
    return new_values

def get_number_not_ascend(values):
    for val in values:
        if val < values[0]:
            return True
    return False

def save_result(name_data, 
                name_tracker,
                scale,
                width,
                height, 
                threshold,
                final_metric
               ):
    root = "csv_result/"
    directory = root+"{}/{}/{}_{}x{}_{}/".format(name_data, name_tracker, scale, width, height, threshold)
    if not os.path.exists(directory):
        os.makedirs(directory)
    final_metric.to_csv(directory+"{}_{}_metric(updated).csv".format(name_data, name_tracker))