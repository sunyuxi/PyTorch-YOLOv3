# -*- coding:utf8 -*-

import os
import sys
import cv2
import shapely.geometry
import numpy as np
import random
import bbox_visualizer as bbv

def getHBB(left, top, width, heigth, poly_coords):
    xmin, ymin, xmax, ymax = min([one[0] for one in poly_coords]), min([one[1] for one in poly_coords]), max([one[0] for one in poly_coords]), max([one[1] for one in poly_coords])
    xmin, ymin, xmax, ymax = xmin-left, ymin-top, xmax-left, ymax-top
    x = (xmin + xmax)/2
    y = (ymin + ymax)/2
    w = xmax - xmin
    h = ymax - ymin
    return x/width, y/heigth, w/width, h/heigth

def updateHBB(scaled_width, scaled_height, const_patch_wh, local_left, local_top, hbb_list):
    new_hbb_list = list()
    for hbb, cls in hbb_list:
        x, y, w, h = hbb[0]*scaled_width, hbb[1]*scaled_height, hbb[2]*scaled_width, hbb[3]*scaled_height
        x, y = x+local_left, y+local_top
        new_hbb = x/const_patch_wh, y/const_patch_wh, w/const_patch_wh, h/const_patch_wh
        new_hbb_list.append((new_hbb, cls))
    return new_hbb_list

#将图片切成patch
def transformDOTA2YOLOv1(class_filepath, input_images_dir, input_gt_dir, output_images_dir, output_gt_dir):
    class_list = [line.strip() for line in open(class_filepath)]
    for imgname in os.listdir(input_images_dir):
        imgname_without_ext = os.path.splitext(imgname)[0]
        img_fielpath = os.path.join(input_images_dir, imgname)
        gt_filepath = os.path.join(input_gt_dir, imgname_without_ext + '.txt')

        ori_gt_list = list()
        line_items_tmp = [line.strip().split() for line in open(gt_filepath)][2:]
        #bbox_list_debug=list()
        #labels_list_debug=list()
        for items in line_items_tmp:
            assert len(items)>=9
            cls_idx = class_list.index(items[8])
            tmp_list = list(map(int, items[0:8]))
            ori_gt_list.append( ([(tmp_list[one], tmp_list[one+1]) for one in range(0, len(tmp_list), 2)], cls_idx) )
            
            #labels_list_debug.append(items[8])
            #xmin, xmax, ymin, ymax = min([one[0] for one in ori_gt_list[-1][0]]), max([one[0] for one in ori_gt_list[-1][0]]), min([one[1] for one in ori_gt_list[-1][0]]), max([one[1] for one in ori_gt_list[-1][0]])
            #print([xmin, ymin, xmax, ymax])
            #bbox_list_debug.append([xmin, ymin, xmax, ymax])
        
        #debug_img = cv2.imread(img_fielpath)
        #img_with_boxes = bbv.draw_multiple_rectangles(debug_img, bbox_list_debug)
        #img_with_boxes = bbv.add_multiple_labels(img_with_boxes, labels_list_debug, bbox_list_debug)
        #cv2.imwrite(os.path.join('tmp', imgname), img_with_boxes)

        ori_img = cv2.imread(img_fielpath)
        height, width, _ = ori_img.shape

        # 将图片切成1024*1024的patch，并且stride设置为512
        left, top, const_patch_wh, stride, const_min_wh = 0, 0, 1024, 512, 100
        patch_list = list()
        while left<width:
            top=0
            while top<height:
                right, bottom = min(left + const_patch_wh, width), min(top + const_patch_wh, height)
                # 长宽必须满足最小值
                if min(right-left, bottom-top)>const_min_wh:
                    patch_list.append((left, top, right, bottom))
                top = top + stride
            left = left + stride
        
        # 将每个bbox分配给各个patch，并且删除不满足的patch，比如，包含的bbox为0。
        # 同时，对于被分割的bbox和原bbox的面积比小于30%，则视为无效
        for bbox_patch in patch_list:
            poly_patch = shapely.geometry.Polygon([(bbox_patch[0], bbox_patch[1]), (bbox_patch[2], bbox_patch[1]), 
                                                (bbox_patch[2], bbox_patch[3]), (bbox_patch[0], bbox_patch[3])])
            hbb_list = list()
            ori_patch_width, ori_patch_height = bbox_patch[2]-bbox_patch[0], bbox_patch[3]-bbox_patch[1]
            for bbox_ori_gt in ori_gt_list:
                poly_ori_gt = shapely.geometry.Polygon(bbox_ori_gt[0])
                #poly_coords = poly_ori_gt.exterior.coords
                #xmin, ymin, xmax, ymax = min([one[0] for one in poly_coords]), min([one[1] for one in poly_coords]), max([one[0] for one in poly_coords]), max([one[1] for one in poly_coords])
                #print((xmin, ymin, xmax, ymax))
                
                poly_iter = poly_ori_gt.intersection(poly_patch)
                if not isinstance(poly_iter, shapely.geometry.Polygon):
                    #print(type(poly_iter))
                    #print(poly_iter.area)
                    continue
                if (poly_iter.area/poly_ori_gt.area<=0.2) or len(poly_iter.exterior.coords)<3:
                    continue
                #poly_coords = poly_iter.exterior.coords
                #xmin, ymin, xmax, ymax = min([one[0] for one in poly_coords]), min([one[1] for one in poly_coords]), max([one[0] for one in poly_coords]), max([one[1] for one in poly_coords])
                #print((xmin, ymin, xmax, ymax))
                hbb = getHBB(bbox_patch[0], bbox_patch[1], ori_patch_width, ori_patch_height, poly_iter.exterior.coords)
                #hbb = getHBB(bbox_patch[0], bbox_patch[1], width, height, poly_ori_gt.exterior.coords)
                hbb_list.append((hbb, bbox_ori_gt[1]))
                #print('=========')
            if len(hbb_list)<=0:
                continue
            
            # resize patch为长宽相同的图片
            local_left, local_top = 0, 0
            out_patch_img = np.zeros((const_patch_wh, const_patch_wh, 3))
            if ori_patch_width>=ori_patch_height:
                scaled_patch_width, scaled_patch_height = const_patch_wh, int(ori_patch_height*const_patch_wh*1.0/ori_patch_width)
                local_left, local_top = 0, 0 #random.randint(0, const_patch_wh - scaled_patch_height)
            else:
                scaled_patch_width, scaled_patch_height = int(ori_patch_width*const_patch_wh*1.0/ori_patch_height), const_patch_wh
                local_left, local_top = 0, 0 #random.randint(0, const_patch_wh - scaled_patch_width), 0
            try:
                patch_img = cv2.resize(ori_img[bbox_patch[1]:bbox_patch[3], bbox_patch[0]:bbox_patch[2]], (scaled_patch_width, scaled_patch_height))
            except:
                print('resize error:')
                print((bbox_patch[1], bbox_patch[3], bbox_patch[0], bbox_patch[2], scaled_patch_width, scaled_patch_height))
                continue

            out_patch_img[local_top:local_top+scaled_patch_height, local_left:local_left+scaled_patch_width] = patch_img
            # 将原始subpatch的hbb值，调整为适应缩放的图片
            hbb_list = updateHBB(scaled_patch_width, scaled_patch_height, const_patch_wh, local_left, local_top, hbb_list)

            patchname_wo_ext = imgname_without_ext+"_"+str(bbox_patch[0])+"_"+str(bbox_patch[1])+"_"+str(ori_patch_width)+"_"+str(ori_patch_height)+"_"+str(local_left)+"_"+str(local_top)
            #cv2.imwrite(os.path.join(output_images_dir, patchname_wo_ext+".png"), out_patch_img)
            #out_patch_img = np.zeros((height, width, 3))
            #out_patch_img[bbox_patch[1]:bbox_patch[3], bbox_patch[0]:bbox_patch[2], :] = ori_img[bbox_patch[1]:bbox_patch[3], bbox_patch[0]:bbox_patch[2], :]
            cv2.imwrite(os.path.join(output_images_dir, patchname_wo_ext+".png"), out_patch_img)
            with open(os.path.join(output_gt_dir, patchname_wo_ext+".txt"), 'w', encoding='utf8') as f:
                for hbb, cls in hbb_list:
                    hbb_str = '\t'.join([str(i) for i in hbb])
                    f.write(str(cls)+'\t'+hbb_str+'\n')

            showHBB(patchname_wo_ext+".png", class_filepath, output_images_dir, output_gt_dir, 'tmp')

# 将图像转换成1024的倍数
def convertImg1024(img, const_patch_wh, ori_gt_list):
    ori_height, ori_width, _ = img.shape
    max_wh = max(ori_height, ori_width)
    if max_wh % const_patch_wh != 0:
        max_wh = (int(max_wh/const_patch_wh)+1)*const_patch_wh
    print('max_wh:'+str(max_wh))
    
    new_img = np.zeros((max_wh, max_wh, 3))
    left, top, right, bottom = 0, 0, ori_width, ori_height
    while left<max_wh:
        right = min(left+ori_width, max_wh)
        local_right, local_bottom = right-left, bottom
        new_img[0:bottom, left:right, :] = img[0:local_bottom, 0:local_right, :]
        left += ori_width

    top, left, right = bottom, 0, max_wh
    while top<max_wh:
        bottom = min(top+ori_height, max_wh)
        local_right, local_bottom = max_wh, bottom-top
        new_img[top:bottom, left:right, :] = new_img[0:local_bottom, 0:local_right, :]
        top += ori_height

    new_gt_list = list()
    for left in range(0, max_wh, ori_width):
        for top in range(0, max_wh, ori_height):
            for items, cls in ori_gt_list:
                # bbox of gt可能包含一些不合法的，比如，超出图像范围的。这些不需要在此处特殊处理，后续split成patch时，会自动进行过滤
                new_items = [(one[0]+left, one[1]+top) for one in items]
                new_gt_list.append((new_items, cls))
    return new_img, new_gt_list

def transformDOTA2YOLOv2(class_filepath, input_images_dir, input_gt_dir, output_images_dir, output_bigimages_dir, output_gt_dir):
    class_list = [line.strip() for line in open(class_filepath)]
    for imgname in os.listdir(input_images_dir):
        imgname_without_ext = os.path.splitext(imgname)[0]
        img_fielpath = os.path.join(input_images_dir, imgname)
        gt_filepath = os.path.join(input_gt_dir, imgname_without_ext + '.txt')

        ori_gt_list = list()
        line_items_tmp = [line.strip().split() for line in open(gt_filepath)][2:]
        
        for items in line_items_tmp:
            assert len(items)>=9
            cls_idx = class_list.index(items[8])
            tmp_list = list(map(int, items[0:8]))
            ori_gt_list.append( ([(tmp_list[one], tmp_list[one+1]) for one in range(0, len(tmp_list), 2)], cls_idx) )
        
        ori_img = cv2.imread(img_fielpath)
        const_patch_wh, stride, const_min_wh = 1024, 1024, 100
        ori_img, ori_gt_list = convertImg1024(ori_img, const_patch_wh, ori_gt_list)
        height, width, _ = ori_img.shape
        cv2.imwrite(os.path.join(output_bigimages_dir, imgname), ori_img)
        '''# debug
        bbox_list_debug=list()
        labels_list_debug=list()
        for items, cls in ori_gt_list:
            labels_list_debug.append(class_list[cls])
            xmin, xmax, ymin, ymax = min([one[0] for one in items]), max([one[0] for one in items]), min([one[1] for one in items]), max([one[1] for one in items])
            bbox_list_debug.append([xmin, ymin, xmax, ymax])
        
        img_with_boxes = bbv.draw_multiple_rectangles(ori_img, bbox_list_debug)
        img_with_boxes = bbv.add_multiple_labels(img_with_boxes, labels_list_debug, bbox_list_debug)
        cv2.imwrite(os.path.join('tmp', imgname), img_with_boxes)
        break'''

        # 将图片切成1024*1024的patch，并且stride设置为512
        left, top = 0, 0
        patch_list = list()
        while left<width:
            top=0
            while top<height:
                right, bottom = min(left + const_patch_wh, width), min(top + const_patch_wh, height)
                # 长宽必须满足最小值
                if min(right-left, bottom-top)>const_min_wh:
                    patch_list.append((left, top, right, bottom))
                top = top + stride
            left = left + stride
        
        # 将每个bbox分配给各个patch，并且删除不满足的patch，比如，包含的bbox为0。
        # 同时，对于被分割的bbox和原bbox的面积比小于30%，则视为无效
        for bbox_patch in patch_list:
            poly_patch = shapely.geometry.Polygon([(bbox_patch[0], bbox_patch[1]), (bbox_patch[2], bbox_patch[1]), 
                                                (bbox_patch[2], bbox_patch[3]), (bbox_patch[0], bbox_patch[3])])
            hbb_list = list()
            ori_patch_width, ori_patch_height = bbox_patch[2]-bbox_patch[0], bbox_patch[3]-bbox_patch[1]
            for bbox_ori_gt in ori_gt_list:
                poly_ori_gt = shapely.geometry.Polygon(bbox_ori_gt[0])
                #poly_coords = poly_ori_gt.exterior.coords
                #xmin, ymin, xmax, ymax = min([one[0] for one in poly_coords]), min([one[1] for one in poly_coords]), max([one[0] for one in poly_coords]), max([one[1] for one in poly_coords])
                #print((xmin, ymin, xmax, ymax))
                
                poly_iter = poly_ori_gt.intersection(poly_patch)
                if not isinstance(poly_iter, shapely.geometry.Polygon):
                    #print(type(poly_iter))
                    #print(poly_iter.area)
                    continue
                if (poly_iter.area/poly_ori_gt.area<=0.3) or len(poly_iter.exterior.coords)<3:
                    continue
                #poly_coords = poly_iter.exterior.coords
                #xmin, ymin, xmax, ymax = min([one[0] for one in poly_coords]), min([one[1] for one in poly_coords]), max([one[0] for one in poly_coords]), max([one[1] for one in poly_coords])
                #print((xmin, ymin, xmax, ymax))
                hbb = getHBB(bbox_patch[0], bbox_patch[1], ori_patch_width, ori_patch_height, poly_iter.exterior.coords)
                #hbb = getHBB(bbox_patch[0], bbox_patch[1], width, height, poly_ori_gt.exterior.coords)
                hbb_list.append((hbb, bbox_ori_gt[1]))
                #print('=========')
            if len(hbb_list)<=0:
                continue
            
            # resize patch为长宽相同的图片
            local_left, local_top = 0, 0
            out_patch_img = np.zeros((const_patch_wh, const_patch_wh, 3))
            if ori_patch_width>=ori_patch_height:
                scaled_patch_width, scaled_patch_height = const_patch_wh, int(ori_patch_height*const_patch_wh*1.0/ori_patch_width)
                print('zero:')
                print(const_patch_wh - scaled_patch_height)
                local_left, local_top = 0, 0 #random.randint(0, const_patch_wh - scaled_patch_height)
            else:
                scaled_patch_width, scaled_patch_height = int(ori_patch_width*const_patch_wh*1.0/ori_patch_height), const_patch_wh
                local_left, local_top = 0, 0 #random.randint(0, const_patch_wh - scaled_patch_width), 0
                print('zero:')
                print(const_patch_wh - scaled_patch_width)
            try:
                patch_img = cv2.resize(ori_img[bbox_patch[1]:bbox_patch[3], bbox_patch[0]:bbox_patch[2]], (scaled_patch_width, scaled_patch_height))
                print('resize')
                print(bbox_patch)
                print((scaled_patch_width, scaled_patch_height))
            except:
                print((bbox_patch[1], bbox_patch[3], bbox_patch[0], bbox_patch[2], scaled_patch_width, scaled_patch_height))
                continue

            out_patch_img[local_top:local_top+scaled_patch_height, local_left:local_left+scaled_patch_width] = patch_img
            # 将原始subpatch的hbb值，调整为适应缩放的图片
            hbb_list = updateHBB(scaled_patch_width, scaled_patch_height, const_patch_wh, local_left, local_top, hbb_list)

            patchname_wo_ext = imgname_without_ext+"_"+str(bbox_patch[0])+"_"+str(bbox_patch[1])+"_"+str(ori_patch_width)+"_"+str(ori_patch_height)+"_"+str(local_left)+"_"+str(local_top)
            #cv2.imwrite(os.path.join(output_images_dir, patchname_wo_ext+".png"), out_patch_img)
            #out_patch_img = np.zeros((height, width, 3))
            #out_patch_img[bbox_patch[1]:bbox_patch[3], bbox_patch[0]:bbox_patch[2], :] = ori_img[bbox_patch[1]:bbox_patch[3], bbox_patch[0]:bbox_patch[2], :]
            cv2.imwrite(os.path.join(output_images_dir, patchname_wo_ext+".png"), out_patch_img)
            with open(os.path.join(output_gt_dir, patchname_wo_ext+".txt"), 'w', encoding='utf8') as f:
                for hbb, cls in hbb_list:
                    hbb_str = '\t'.join([str(i) for i in hbb])
                    f.write(str(cls)+'\t'+hbb_str+'\n')

            #showHBB(patchname_wo_ext+".png", class_filepath, output_images_dir, output_gt_dir, 'tmp')        

def showHBB(imgname, classname_filepath, input_images_dir, input_labels_dir, output_images_dir):
    
    classname_list = [line.strip() for line in open(classname_filepath)]
    if True:
        imgname_wo_ext = os.path.splitext(imgname)[0]
        img = cv2.imread(os.path.join(input_images_dir, imgname))
        height, width, _ = img.shape
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        bboxes=[]
        labels=[]
        for line in open(os.path.join(input_labels_dir, imgname_wo_ext+'.txt')):
            arr_tmp = line.strip().split()
            hbb = [ float(arr_tmp[1])*width, float(arr_tmp[2])*height, float(arr_tmp[3])*width, float(arr_tmp[4])*height ]
            #print([int(hbb[0]-hbb[2]/2), int(hbb[1]-hbb[3]/2), int(hbb[0]+hbb[2]/2), int(hbb[1]+hbb[3]/2)])
            bboxes.append([int(hbb[0]-hbb[2]/2), int(hbb[1]-hbb[3]/2), int(hbb[0]+hbb[2]/2), int(hbb[1]+hbb[3]/2)])
            labels.append(classname_list[int(arr_tmp[0])])
        # bboxes的坐标必须转换成int的形式
        img_with_boxes = bbv.draw_multiple_rectangles(img, bboxes)

        img_with_boxes = bbv.add_multiple_labels(img_with_boxes, labels, bboxes)
        # img_final = cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR)
        #print(img_with_boxes.shape)
        cv2.imwrite(os.path.join(output_images_dir, imgname), img_with_boxes)

def showHBBWithDir(classname_filepath, input_images_dir, input_labels_dir, output_images_dir):
    
    classname_list = [line.strip() for line in open(classname_filepath)]
    for imgname in os.listdir(input_images_dir):
        imgname_wo_ext = os.path.splitext(imgname)[0]
        img = cv2.imread(os.path.join(input_images_dir, imgname))
        height, width, _ = img.shape
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        bboxes=[]
        labels=[]
        for line in open(os.path.join(input_labels_dir, imgname_wo_ext+'.txt')):
            arr_tmp = line.strip().split()
            hbb = [ float(arr_tmp[1])*width, float(arr_tmp[2])*height, float(arr_tmp[3])*width, float(arr_tmp[4])*height ]
            #print([int(hbb[0]-hbb[2]/2), int(hbb[1]-hbb[3]/2), int(hbb[0]+hbb[2]/2), int(hbb[1]+hbb[3]/2)])
            bboxes.append([int(hbb[0]-hbb[2]/2), int(hbb[1]-hbb[3]/2), int(hbb[0]+hbb[2]/2), int(hbb[1]+hbb[3]/2)])
            labels.append(classname_list[int(arr_tmp[0])])
        # bboxes的坐标必须转换成int的形式
        img_with_boxes = bbv.draw_multiple_rectangles(img, bboxes)

        img_with_boxes = bbv.add_multiple_labels(img_with_boxes, labels, bboxes)
        # img_final = cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR)
        #print(img_with_boxes.shape)
        cv2.imwrite(os.path.join(output_images_dir, imgname), img_with_boxes)


if __name__ == '__main__':
    oper_type = int(sys.argv[1])
    if oper_type == 1:
        class_filepath = sys.argv[2]
        input_images_dir, input_gt_dir = sys.argv[3], sys.argv[4]
        output_images_dir, output_bigimages_dir, output_gt_dir = sys.argv[5], sys.argv[6], sys.argv[7]
        transformDOTA2YOLOv2(class_filepath, input_images_dir, input_gt_dir, output_images_dir, output_bigimages_dir, output_gt_dir)
    elif oper_type == 2:
        classname_filepath=sys.argv[2]
        input_images_dir, input_labels_dir = sys.argv[3], sys.argv[4]
        output_images_dir = sys.argv[5]
        showHBBWithDir(classname_filepath, input_images_dir, input_labels_dir, output_images_dir)