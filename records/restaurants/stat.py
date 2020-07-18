import glob
import json
from pprint import pprint

def best_epoch(epoch_list):
    best_f1 = 0.0
    best_epoch = None
    for epoch in epoch_list:
        if epoch["f1"] > best_f1:
            best_f1 = epoch["f1"]
            best_epoch = epoch
    return best_f1, best_epoch

def calculate_tp(counts, rec):
    pos, neg = counts
    tp = rec * pos
    return tp

def calculate_fp(tp, pre):
    fp = (tp/pre) - tp
    return fp

def calculate_micro_f1(perf_list):
    total_tp = sum([target[1] for target in perf_list])
    total_fp = sum([target[2] for target in perf_list])
    total_p = sum([target[0] for target in perf_list])
    precision = total_tp/(total_tp+total_fp)
    recall = total_tp/total_p
    f1 = 2 * ((precision*recall)/(precision + recall))
    return f1

def calculate_macro_f1(perf_list):
    return 1/len(perf_list) * sum([target[3] for target in perf_list])

multiclass = True

class_counts = {
    "entity": {
        "RESTAURANT": (560, 1435),
        "SERVICE": (419, 1576),
        "FOOD": (757, 1238),
        "DRINKS": (79, 1916),
        "AMBIENCE": (226, 1769),
        "NaN": (292, 1703),
        "LOCATION": (28, 1967)
    }, 
    "attribute": {
        "GENERAL": (994, 1001),
        "QUALITY": (716, 1279),
        "STYLE_OPTIONS": (156, 1839),
        "PRICES": (177, 1818),
        "MISCELLANEOUS": (97, 1898),
        "NaN": (292, 1703)
    }
}

type = "attribute"

folder = "_multiple_runs"

file_list = glob.glob("records/restaurants/"+type+folder+"/training*")
file_list = glob.glob("records/restaurants/"+type+"_multi/training*")
file_performance = []
for file in file_list:
    index = 0
    parameter_dict = {}
    train_list = []
    eval_list = []
    with open(file, "r") as f:
        lines = f.readlines()
        for line in lines:
            if index == 0:
                parameter_dict = json.loads(line)
            else:
                dictionary = json.loads(line)
                if "step" in dictionary.keys():
                    if dictionary["step"] == "train":
                        train_list.append(dictionary)
                    else:
                        eval_list.append(dictionary)
            index += 1
    file_performance.append({"file": file, "param": parameter_dict, "train": train_list, "val": eval_list})

# For multiclass
if multiclass:
    for file in file_performance:
        print(file["file"])
        print(file["param"]["embedding"], ":", "KCL" if file["param"]["use_kcl"] else "MCL")
        best_train, epoch = best_epoch(file["train"])
        best_val, epoch = best_epoch(file["val"])
        print("Train F1:", best_train)
        print("Val F1:", best_val)

else:
    mcl = []
    kcl = []
    for file in file_performance:
        if ("use_linmodel" in file["param"].keys() and not file["param"]["use_linmodel"]) or "use_linmodel" not in file["param"].keys():
            continue
        best_f1, epoch = best_epoch(file["val"])
        if epoch is None:
            continue
        if epoch["precision"] <= 0.001:
            print("Cutting one training...", file["file"])
            continue
        if file["param"]["binary_target_class"] not in class_counts[type].keys():
            print("Not contained:", file["file"])
            continue
        print(file["file"])
        counts = class_counts[type][file["param"]["binary_target_class"]]
        p = counts[0]
        tp = calculate_tp(counts, epoch["recall"])
        fp = calculate_fp(tp, epoch["precision"])
        best_val = (p, tp, fp, best_f1)
        if file["param"]["use_kcl"]:
            kcl.append(best_val)
        else:
            mcl.append(best_val)
    print(type)
    if len(mcl) > 0:
        macro_mcl = calculate_macro_f1(mcl)
        print("MCL macro F1:", macro_mcl)
        micro_mcl = calculate_micro_f1(mcl)
        print("MCL micro F1:", micro_mcl)
    if len(kcl) > 0:
        macro_kcl = calculate_macro_f1(kcl)
        print("KCL macro F1:", macro_kcl)
        micro_kcl = calculate_micro_f1(kcl)
        print("KCL micro F1:", micro_kcl)