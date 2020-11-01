
import pandas as pd

import pandas as pd
import math
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import lightgbm
from shapely.geometry import Polygon
import pickle as plk

import csv
import os
import math
import numpy as np
def get_polygon(point):

    if point["shape"] == "circle":
        return Polygon([(point.xcenter+point.rhorizontal,point.ycenter),
            (point.xcenter+0.5*point.rhorizontal,point.ycenter+0.5*point.rvertical),
             (point.xcenter+0*point.rhorizontal,point.ycenter+1*point.rvertical),
             (point.xcenter-0.5*point.rhorizontal,point.ycenter+0.5*point.rvertical),
             (point.xcenter-1*point.rhorizontal,point.ycenter+0*point.rvertical),
             (point.xcenter-0.5*point.rhorizontal,point.ycenter-0.5*point.rvertical),
             (point.xcenter+0*point.rhorizontal,point.ycenter-1*point.rvertical),
             (point.xcenter+0.5*point.rhorizontal,point.ycenter-0.5*point.rvertical)])
                
    elif point["shape"] == "rectangle":
        return Polygon([(point.xcenter+point.rhorizontal,point.ycenter),
            (point.xcenter+1*point.rhorizontal,point.ycenter+1*point.rvertical),
             (point.xcenter-1*point.rhorizontal,point.ycenter+1*point.rvertical),
             (point.xcenter-1*point.rhorizontal,point.ycenter-1*point.rvertical),
             (point.xcenter+1*point.rhorizontal,point.ycenter-1*point.rvertical)
           ])
    print("problem get poly", point)


    
def get_difference_area(point,points):
    inter_area = get_polygon(point)
    for p in points.iterrows():
   #     print("inter ",inter_area," poly ", get_polygon(p[1]))
   #     print("inter area ",inter_area.area," poly ", get_polygon(p[1]).area)
        if inter_area.area < 0.00001:
            return inter_area
        inter_area = inter_area.difference(get_polygon(p[1]))
    return inter_area

def get_difference_areas(points1,points2,verbose=False):
    if points1.shape[0]==0:
        return [0]
    res = []
    return points1.apply(lambda x: get_difference_area(x,points2).area/(get_polygon(x).area if get_polygon(x).area>0 else 0.0000001),axis=1).tolist()
    

def get_dist_to_point(point1,point2,verbose=True, metric="euclead"):
    if verbose:
        print("get dist",point2)
    if metric=="euclead":
        return math.sqrt(
            (point1.xcenter - point2.xcenter)**2+(point1.ycenter - point2.ycenter)**2
        )
    if metric=="manhattan":
        return  abs(point1.xcenter - point2.xcenter),abs(point1.ycenter - point2.ycenter)
    return None
    

def get_closest_point(raw_point, raw_points, verbose=True,options={"metric":"euclead"}):
    if verbose:
        print("get closest",raw_points)
    dists = raw_points.reset_index().apply(lambda x:
                                           get_dist_to_point(
                                               raw_point,x,verbose=verbose,metric=options["metric"]),
                                           axis=1)
    if verbose:
        print(raw_point.xcenter, raw_point.ycenter,"dist", dists.min(),raw_points.shape)
    return raw_points.reset_index().iloc[dists.idxmin()]

def is_point_good(point1, point2,trashold=0.1,verbose=True):
    if verbose:
        print("isgood",point1,point2)
    x_shapes = point1.rhorizontal + point2.rhorizontal
    y_shapes = point1.rvertical + point2.rvertical
    x_dist,y_dist=get_dist_to_point(point1,point2,verbose=verbose, metric="manhattan")
    if x_dist>trashold*x_shapes or y_dist>trashold*y_shapes:
        return False
    return True
# def is_closes_point_good(raw_point, closests, verbose=True,options={"metric":"euclead"}):
# #    closest=get_closest_point(raw_point, raw_points, verbose=verbose,options=options)
#     if verbose:
#         print("is closes good",raw_point,closests)
#     return (raw_point, closests, verbose=verbose)

def get_closest_points(points1,points2,verbose=False,options={}):
    if points1.shape[0]==0:
        return []
    if points2.shape[0]==0:
        return []
    closests = [get_closest_point(p1[1],points2,verbose=verbose)["index"] for p1 in points1.iterrows()]
    if verbose:
        print("get closess2 ",type(closests))
    return closests

def get_bad_num(points1,closests,verbose=False,trashold=0.5,options={}):
    if points1.shape[0]==0:
        return 0
    if len(closests)==0:
        return points1.shape[0]
    if points1.shape[0] != len(closests):
        print("get bad problem p1",points1.shape,closests.shape,points1,"type p1",type(points1),"p2",closests,
              "type p2",type(closests))
        return -1
    badlist = [not is_point_good(points1.iloc[i],closests.iloc[i],trashold=trashold,verbose=verbose)
               for i in range(points1.shape[0])] 
    if verbose:
        print(badlist)
    return int(sum(badlist))

def get_df_train_base(df=None,df_exp=None,options={}):
    df_train =pd.DataFrame()

    for iter,file_name in  enumerate(df.file_name.unique()):
        df_file = df[df["file_name"] == file_name] 
        if iter%2000==0:
            print(file_name, iter)
        for ind in [1,2,3]:
            sample="sample_"+str(ind)

            target = df_exp.loc[file_name+".png"]["Sample "+str(ind)
                                                 ] if file_name+".png" in df_exp.index else None
         #   print(sample)
            closests_to_exp=get_closest_points(df_file[df_file["user_name"]=="Expert"],
                                               df_file[df_file["user_name"]==sample],
                           verbose=False)
            
            closests_to_neuro=get_closest_points(df_file[df_file["user_name"]==sample],df_file[df_file["user_name"]=="Expert"],
                           verbose=False)
            #print("type df_raw",closests_to_exp,type(df_raw.iloc[closests_to_exp]))
 
            diff_ares_neuro = get_difference_areas(df_file[df_file["user_name"]==sample],
                                                    df_file[df_file["user_name"]=="Expert"],
                           verbose=False)
            diff_ares_exp = get_difference_areas(df_file[df_file["user_name"]=="Expert"],
                                                    df_file[df_file["user_name"]==sample],
                                                           verbose=False)

            df_train=df_train.append(pd.Series({"file_name":file_name,"sample":sample,
                                                "expert_points":df_file[df_file["user_name"]=="Expert"].reset_index()["index"].tolist(),
                                                "sample_points":df_file[df_file["user_name"]==sample].reset_index()["index"].tolist(),
                                                "diff_ares_neuro":diff_ares_neuro,
                                                "diff_ares_exp":diff_ares_exp,
                                                "closests_to_exp":closests_to_exp,
                                                "closests_to_neuro":closests_to_neuro,
                                               "target":target}),
                                      ignore_index=True)

    return df_train
def get_df_train(df_train=None,df_raw=None,exp_diff_tr=0.5,neuro_diff=0.5,trashold_exp=0,
                trashold_neuro=0.5):
    df_train["big_exp_diff"] = df_train.diff_ares_exp.apply(lambda x: sum([v>exp_diff_tr for v in x]))
    df_train["big_neuro_diff"] = df_train.diff_ares_neuro.apply(lambda x: sum([v>neuro_diff for v in x]))
    df_train["badcount_expert"]=df_train.apply(lambda x: get_bad_num(df_raw.loc[x.expert_points]
                                                                  ,df_raw.loc[x.closests_to_exp],
                           verbose=False,trashold=trashold_exp), axis=1)
    df_train["badcount_neuro"]=df_train.apply(lambda x: get_bad_num(df_raw.loc[x.sample_points]
                                                                  ,df_raw.loc[x.closests_to_neuro],
                           verbose=False, trashold=trashold_neuro), axis=1)
    return df_train




def filelist(directory):
    ret_list = []
    for folder, subs, files in os.walk(directory):
        for filename in files:
            # if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
            ret_list.append(os.path.join(folder, filename))
    return ret_list


def bb_intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    iou = interArea / float(boxAArea + boxBArea - interArea)
    iou = math.sqrt(iou)
    return iou


def custom_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    xcA = (boxA[0] + boxA[2]) / 2
    ycA = (boxA[1] + boxA[3]) / 2
    xcB = (boxB[0] + boxB[2]) / 2
    ycB = (boxB[1] + boxB[3]) / 2
    rxA = boxA[2] - xcA
    ryA = boxA[3] - ycA
    rxB = boxB[2] - xcB
    ryB = boxB[3] - ycB
    dist = math.sqrt((xcB - xcA)**2 + (ycB - ycA)**2)

    g_ell_center = (xcA / 1024, ycA/ 1024)
    g_ell_width = rxA * 2 / 1024
    g_ell_height = ryA * 2 / 1024
    angle = 0.

    cos_angle = np.cos(np.radians(180. - angle))
    sin_angle = np.sin(np.radians(180. - angle))

    xc = (xcB / 1024) - g_ell_center[0]
    yc = (ycB / 1024) - g_ell_center[1]

    xct = xc * cos_angle - yc * sin_angle
    yct = xc * sin_angle + yc * cos_angle

    rad_cc = (xct ** 2 / (g_ell_width / 2.) ** 2) + (yct ** 2 / (g_ell_height / 2.) ** 2)

    colors_array = []

    #for r in rad_cc:





    if dist < 0.9 * min(rxA, rxB, ryA, ryB):
        interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
        boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
        boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
        metric = interArea / float(min(boxAArea, boxBArea))
        return metric
    else:
        return 0


def generate_additional_features(csv_file, src_file_dir, marks_csv_file, out_csv_name="custom_metric.csv"):
    marks_csvfile = open(marks_csv_file, encoding='utf8', mode='r')
    marks_csvreader = csv.DictReader(marks_csvfile, delimiter=',', quotechar='"')
    marks = {}
    for row in marks_csvreader:
        filename = row['Case'].rsplit(".", 1)[0]
        #if filename not in marks.keys():
        marks[filename] = {}
        marks[filename]['sample_1'] = row['Sample 1']
        marks[filename]['sample_2'] = row['Sample 2']
        marks[filename]['sample_3'] = row['Sample 3']
        #print(row)
        #exit(1)


    csvfile = open(csv_file, encoding='utf8', mode='r')
    csvreader = csv.DictReader(csvfile, delimiter=',', quotechar='"')
    files = filelist(src_file_dir)
    iou_thresh = 0.7

    labels = {
        'sample_2' : {},
        'Expert' : {},
        'sample_1' : {},
        'sample_3' : {},
    }

    for row in csvreader:
        user_name = row[' user_name']
        if row['file_name'] in labels[user_name].keys():
            labels[user_name][row['file_name']].append(row)
        else:
            labels[user_name][row['file_name']] = [row]

    csv_f = open(out_csv_name,'w')
    w = csv.writer(csv_f)
    dict_row = {
            'file_name' : '',
            'user_name' : '',
            'TP' : '',
            'FP' : '',
            'FN' : '',
            'precision' : '',
            'recall' : '',
            'mark' : '',
        }
    w.writerow(dict_row.keys())

    for f in files:
        ele = os.path.basename(f).rsplit("_", 1)[0]
        if ele == "00003894_005":
            print(1)
        if ele in labels['Expert'].keys():
            gt_rows = labels['Expert'][ele].copy()
        else:
            gt_rows = []
        if ele in labels['sample_1'].keys():
            sample1_rows = labels['sample_1'][ele].copy()
        else:
            sample1_rows = []
        if ele in labels['sample_2'].keys():
            sample2_rows = labels['sample_2'][ele].copy()
        else:
            sample2_rows = []
        if ele in labels['sample_3'].keys():
            sample3_rows = labels['sample_3'][ele].copy()
        else:
            sample3_rows = []

        #calc iou, tp, fn, fp
        total_tp_count = 0
        total_fp_count = 0
        total_fn_count = 0

        #for label in labels['sample_1']:
        for deti in range(len(sample1_rows)):
            max_iou = float(0)
            max_iou_gt_index = 0
            det_xcenter = int(float(sample1_rows[deti][' xcenter']))
            det_ycenter = int(float(sample1_rows[deti][' ycenter']))
            det_rhorizontal = int(float(sample1_rows[deti][' rhorizontal']))
            det_rvertical = int(float(sample1_rows[deti][' rvertical']))
            det_xmin = det_xcenter - det_rhorizontal
            det_ymin = det_ycenter - det_rvertical
            det_xmax = det_xcenter + det_rhorizontal
            det_ymax = det_ycenter + det_rvertical
            if len(gt_rows) > 0:
                for gti in range(len(gt_rows)):
                    gt_xcenter = int(float(gt_rows[gti][' xcenter']))
                    gt_ycenter = int(float(gt_rows[gti][' ycenter']))
                    gt_rhorizontal = int(float(gt_rows[gti][' rhorizontal']))
                    gt_rvertical = int(float(gt_rows[gti][' rvertical']))
                    gt_xmin = gt_xcenter - gt_rhorizontal
                    gt_ymin = gt_ycenter - gt_rvertical
                    gt_xmax = gt_xcenter + gt_rhorizontal
                    gt_ymax = gt_ycenter + gt_rvertical
                    box_gt = (gt_xmin, gt_ymin, gt_xmax, gt_ymax)
                    box_det = (det_xmin, det_ymin, det_xmax, det_ymax)

                    iou = abs(custom_iou(box_gt, box_det))

                    if iou > max_iou:
                        max_iou = iou
                        max_iou_gt_index = gti

                if max_iou > iou_thresh:
                    total_tp_count += 1
                    del gt_rows[max_iou_gt_index]
                else:
                    total_fp_count += 1
            else:
                total_fp_count += 1

        if len(gt_rows) > 0:
            total_fn_count += len(gt_rows)

        precision = 0
        if float(total_fp_count + total_tp_count) != 0:
            precision = total_tp_count / float(total_fp_count + total_tp_count)

        recall = 0
        if float(total_fn_count + total_tp_count) != 0:
            recall = total_tp_count / float(total_fn_count + total_tp_count)

        if ele in marks.keys():
            mark = marks[ele]['sample_1']
        else:
            mark = "No Mark"

        dict_row = {
            'file_name' : ele,
            'user_name' : 'sample_1',
            'TP' : total_tp_count,
            'FP' : total_fp_count,
            'FN' : total_fn_count,
            'precision' : precision,
            'recall' : recall,
            'mark' : mark,
        }
        w.writerow(dict_row.values())
       # print("{}:{},\ttp:{},\tfp:{},\tfn:{},presision:{:5.2f},\trecall:{:5.2f}\tmark:{}".format(ele, "sample_1", total_tp_count,
       #                                                                                total_fp_count, total_fn_count,
       #                                                                                precision, recall, mark))

        if ele in labels['Expert'].keys():
            gt_rows = labels['Expert'][ele].copy()
        else:
            gt_rows = []
        # calc iou, tp, fn, fp
        total_tp_count = 0
        total_fp_count = 0
        total_fn_count = 0

        # for label in labels['sample_2']:
        for deti in range(len(sample2_rows)):
            max_iou = float(0)
            max_iou_gt_index = 0
            det_xcenter = int(float(sample2_rows[deti][' xcenter']))
            det_ycenter = int(float(sample2_rows[deti][' ycenter']))
            det_rhorizontal = int(float(sample2_rows[deti][' rhorizontal']))
            det_rvertical = int(float(sample2_rows[deti][' rvertical']))
            det_xmin = det_xcenter - det_rhorizontal
            det_ymin = det_ycenter - det_rvertical
            det_xmax = det_xcenter + det_rhorizontal
            det_ymax = det_ycenter + det_rvertical
            if len(gt_rows) > 0:
                for gti in range(len(gt_rows)):
                    gt_xcenter = int(float(gt_rows[gti][' xcenter']))
                    gt_ycenter = int(float(gt_rows[gti][' ycenter']))
                    gt_rhorizontal = int(float(gt_rows[gti][' rhorizontal']))
                    gt_rvertical = int(float(gt_rows[gti][' rvertical']))
                    gt_xmin = gt_xcenter - gt_rhorizontal
                    gt_ymin = gt_ycenter - gt_rvertical
                    gt_xmax = gt_xcenter + gt_rhorizontal
                    gt_ymax = gt_ycenter + gt_rvertical
                    box_gt = (gt_xmin, gt_ymin, gt_xmax, gt_ymax)
                    box_det = (det_xmin, det_ymin, det_xmax, det_ymax)

                    iou = abs(custom_iou(box_gt, box_det))

                    if iou > max_iou:
                        max_iou = iou
                        max_iou_gt_index = gti

                if max_iou > iou_thresh:
                    total_tp_count += 1
                    del gt_rows[max_iou_gt_index]
                else:
                    total_fp_count += 1
            else:
                total_fp_count += 1

        if len(gt_rows) > 0:
            total_fn_count += len(gt_rows)

        precision = 0
        if float(total_fp_count + total_tp_count) != 0:
            precision = total_tp_count / float(total_fp_count + total_tp_count)

        recall = 0
        if float(total_fn_count + total_tp_count) != 0:
            recall = total_tp_count / float(total_fn_count + total_tp_count)

        if ele in marks.keys():
            mark = marks[ele]['sample_2']
        else:
            mark = "No Mark"

        dict_row = {
            'file_name': ele,
            'user_name': 'sample_2',
            'TP': total_tp_count,
            'FP': total_fp_count,
            'FN': total_fn_count,
            'precision': precision,
            'recall': recall,
            'mark': mark,
        }
        w.writerow(dict_row.values())
       # print("{}:{},\ttp:{},\tfp:{},\tfn:{},presision:{:5.2f},\trecall:{:5.2f}\tmark:{}".format(ele, "sample_2", total_tp_count,
       #                                                                                total_fp_count, total_fn_count,
       #                                                                                precision, recall, mark))

        if ele in labels['Expert'].keys():
            gt_rows = labels['Expert'][ele].copy()
        else:
            gt_rows = []
        # calc iou, tp, fn, fp
        total_tp_count = 0
        total_fp_count = 0
        total_fn_count = 0
        # for label in labels['sample_3']:
        for deti in range(len(sample3_rows)):
            max_iou = float(0)
            max_iou_gt_index = 0
            det_xcenter = int(float(sample3_rows[deti][' xcenter']))
            det_ycenter = int(float(sample3_rows[deti][' ycenter']))
            det_rhorizontal = int(float(sample3_rows[deti][' rhorizontal']))
            det_rvertical = int(float(sample3_rows[deti][' rvertical']))
            det_xmin = det_xcenter - det_rhorizontal
            det_ymin = det_ycenter - det_rvertical
            det_xmax = det_xcenter + det_rhorizontal
            det_ymax = det_ycenter + det_rvertical
            if len(gt_rows) > 0:
                for gti in range(len(gt_rows)):
                    gt_xcenter = int(float(gt_rows[gti][' xcenter']))
                    gt_ycenter = int(float(gt_rows[gti][' ycenter']))
                    gt_rhorizontal = int(float(gt_rows[gti][' rhorizontal']))
                    gt_rvertical = int(float(gt_rows[gti][' rvertical']))
                    gt_xmin = gt_xcenter - gt_rhorizontal
                    gt_ymin = gt_ycenter - gt_rvertical
                    gt_xmax = gt_xcenter + gt_rhorizontal
                    gt_ymax = gt_ycenter + gt_rvertical
                    box_gt = (gt_xmin, gt_ymin, gt_xmax, gt_ymax)
                    box_det = (det_xmin, det_ymin, det_xmax, det_ymax)

                    iou = abs(custom_iou(box_gt, box_det))

                    if iou > max_iou:
                        max_iou = iou
                        max_iou_gt_index = gti

                if max_iou > iou_thresh:
                    total_tp_count += 1
                    del gt_rows[max_iou_gt_index]
                else:
                    total_fp_count += 1
            else:
                total_fp_count += 1

        if len(gt_rows) > 0:
            total_fn_count += len(gt_rows)

        precision = 0
        if float(total_fp_count + total_tp_count) != 0:
            precision = total_tp_count / float(total_fp_count + total_tp_count)

        recall = 0
        if float(total_fn_count + total_tp_count) != 0:
            recall = total_tp_count / float(total_fn_count + total_tp_count)

        if ele in marks.keys():
            mark = marks[ele]['sample_3']
        else:
            mark = "No Mark"
        dict_row = {
            'file_name': ele,
            'user_name': 'sample_3',
            'TP': total_tp_count,
            'FP': total_fp_count,
            'FN': total_fn_count,
            'precision': precision,
            'recall': recall,
            'mark': mark,
        }
        w.writerow(dict_row.values())
       # print("{}:{},\ttp:{},\tfp:{},\tfn:{},presision:{:5.2f},\trecall:{:5.2f}\tmark:{}".format(ele, "sample_3", total_tp_count,
       #                                                                                total_fp_count, total_fn_count,
       #                                                                                precision, recall, mark))
    csv_f.close()
    return os.path.abspath(out_csv_name)



#print(1)

def generate_answer(input_dir="Dataset",model="model.pkl",out_file = "SecretPart_DL.csv"):
    src_file_dir = input_dir+"/Expert"
    csv_file = input_dir+"/DX_TEST_RESULT_FULL.csv"
    marks_csv_file = input_dir+"/OpenPart.csv"
    generate_additional_features(csv_file, src_file_dir, marks_csv_file, out_csv_name="tmp.csv")

    df_exp = pd.read_csv("Dataset/OpenPart.csv")
    df_exp.head()
    df_exp.set_index("Case",inplace=True)

    df_raw = pd.read_csv(input_dir+"/DX_TEST_RESULT_FULL.csv")
    df_raw.rename({v:v[1:]  for v in df_raw.columns if v[0]==" "}, axis=1, inplace=True)
    df_train_base = get_df_train_base(df=df_raw,df_exp=df_exp)
    ivan_feature = pd.read_csv("tmp.csv")
    ivan_feature.rename({"user_name":"sample"}, axis=1, inplace=True)
    d = {'exp_diff_tr': 0.2,
           'neuro_diff': 0.8,
           'trashold_exp': 0.8,
                'trashold_neuro': 0.5}
    df_train = get_df_train(df_train_base,df_raw=df_raw,**d)
    df_train = df_train.set_index(["file_name","sample"]).join(ivan_feature.set_index(["file_name","sample"])).reset_index()
    model_file = "model.pkl"
    with open(model_file,"rb") as f:
        model = plk.load(f)
    predict_list = [c for c in (df_raw.file_name + ".png").unique() if c not in df_exp.index] 

    train_cols = ['badcount_expert',
     'badcount_neuro',
     'big_exp_diff',
     'big_neuro_diff',
     'TP',
     'FP',
     'FN',
     'precision',
     'recall']
    df_train["preds"] = model.predict(df_train[train_cols])
    result = pd.DataFrame()
    #['Case', 'Sample 1', 'Sample 2', 'Sample 3']

    for r in predict_list:
        df_loc = df_train[df_train.file_name+".png" == r]
        result = result.append(pd.Series({"Case":r,
                                         'Sample 1':df_loc[df_loc["sample"]=="sample_1"]["preds"].iloc[0],
                                          'Sample 2':df_loc[df_loc["sample"]=="sample_2"]["preds"].iloc[0],
                                          'Sample 3':df_loc[df_loc["sample"]=="sample_3"]["preds"].iloc[0]

                                         }), ignore_index=True)

    result.to_csv(out_file)

if __name__=="__main__":
    import sys
    print( "run script: ", sys.argv[0])
    print("Number of arguments: ", len(sys.argv))
    input_dir="Dataset"
    model="model.pkl"
    out_file = "SecretPart_DL.csv"
    if len(sys.argv) >1:
        input_dit=sys.argv[1]
    if len(sys.argv) >2:
        input_dit=sys.argv[2]
    if len(sys.argv) >3:
        out_file=sys.argv[3]
    if len(sys.argv) >4:    
        print(" a lot of arguments!")

        
    print("The arguments are: " , str(sys.argv))
    generate_answer(input_dir=input_dir,model=model,out_file=out_file)
