import numpy as np 
import cv2
from ultralytics import YOLO
import os
import sys
import torch
import time
#os.chdir(os.path.expanduser("~")+"/school/code/WeedID")
#print(os.getcwd())
#from .phoenixbot_weed_detection.weed_detection.weed_detection.submodules.plant_id.segmentation import run_segmentation
#from .school.code.WeedID.plant_tracking.segmentation import run_segmentation






# Helper functions for image segmentation

import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from math import sqrt

LOWER = np.array([40,30,80]) # for outdoor
UPPER = np.array([100,255,255])

# LOWER = np.array([35 ,50, 100]) # for lab
# UPPER = np.array([100,255,255])

EPS = 3
MIN_SAMPLES = 3

def get_green(orig_img: np.ndarray) -> np.ndarray:
    """
    Given numpy array representation of image, return an image with the green parts isolated.

    Args:
        orig_img: original image in numpy array form (width x height x 3)

    Returns:
        green_areas: numpy array representing the green-isolated image
    """

    hsv_image = cv2.cvtColor(orig_img, cv2.COLOR_BGR2HSV)

    # Apply mask to isolate the green areas
    mask = cv2.inRange(hsv_image, LOWER, UPPER)
    green_areas = cv2.bitwise_and(orig_img, orig_img, mask=mask)

    return green_areas


def green_to_bnw(green_areas: np.ndarray) -> np.ndarray:
    """
    Given an image with only green areas, convert green areas to white, rest to black.

    Args:
        green_areas_denoised: numpy array that only has green areas of an image
    
    Returns:
        black_image_flat (np.ndarray) : One layer of black and white image of the green areas

    """
    hsv = cv2.cvtColor(green_areas, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER, UPPER) 
    
    # Create an empty black image
    black_image = np.zeros_like(green_areas)
    
    # Set green areas to white in the black image
    black_image[mask != 0] = 255

    # ----- Denoising ----- #
    open_kernel = np.ones((3,3), np.uint8)
    close_kernel = np.ones((10,10), np.uint8)
    black_image  = cv2.morphologyEx(black_image, cv2.MORPH_OPEN, open_kernel)
    black_image = cv2.morphologyEx(black_image, cv2.MORPH_CLOSE, close_kernel)

    black_image_flat = np.mean(black_image, axis=2) # type: ignore
    black_image_flat[black_image_flat > 150] = 255
    black_image_flat[black_image_flat < 150] = 0
    
    return black_image_flat


def refactor_to_lower_res(img: np.ndarray, total_pixels: int)->tuple[np.ndarray, float]:
    """
    Given an image, refactor it to a new size specified by caller.

    Args: 
        img: numpy array of an image
        total_pixels: new image size after refactoring
    
    Returns:
        A tuple that contains
            resized_img (np.ndarray): Resized image
            old_new_img_ratio (float): Size ratio of old & new image
    """
    frac_x = img.shape[0] / img.shape[1]
    new_x = round(sqrt(total_pixels / frac_x))
    new_y = round(new_x * frac_x)
    old_new_img_ratio = img.shape[0] / new_y
    resized_img = cv2.resize(img, (new_x, new_y))
    return (resized_img, old_new_img_ratio)


def binary_to_cartesian(bnw_image: np.ndarray) -> np.ndarray:
    """
    Given numpy array of 0s and 255s that represent a black and white image,
    return two lists that have the x and y coordinates of white areas.

    Args:
        bnw_image (np.ndarray): black and white image that only have 0s and 255s

    Returns:
        xys (np.ndarray): [x,y] points of all white areas
    """

    xys = []
    for y, row in enumerate(bnw_image):
        for x, value in enumerate(row):
            if value == 255:
                xys.append([x, abs(y - bnw_image.shape[0])])

    return np.array(xys)


def DBSCAN_clustering(white_points):
    """
    Given an array with two columns which stores x and y values representing
    white points in the denoised image, return an array where each row
    classifies which cluster a point is in (same amount of rows as the white
    points array)

    Args:
        white_points (np.ndarray): x and y coordinates of white points

    Returns:
        A tuple consisting of:
            cluster_labels(np.ndarray): Labels for identified clusters 
            cluster_points (np.ndarray): Points belonging in identified clusters
    """

    cluster_points, cluster_labels = np.ndarray([]), np.ndarray([])

    if len(white_points) != 0:
        dbscan_model = DBSCAN(eps=EPS, min_samples=EPS, n_jobs=-1)
        dbscan_model.fit(white_points)
        dbscan_labels = dbscan_model.fit_predict(white_points)

        # Filter out noise points (-1)
        cluster_points = white_points[dbscan_labels != -1]
        cluster_labels = dbscan_labels[dbscan_labels != -1]
    
    return cluster_labels, cluster_points
    
def find_bounding_boxes(img: np.ndarray, cluster_points: np.ndarray, cluster_labels: np.ndarray)->list:
    """
    Find the bounding box for each cluster.

    Args:
        dbscan_points (numpy.ndarray): Array of points in the image.
        dbscan_result (numpy.ndarray): Result of DBSCAN clustering algorithm.

    Returns:
        bounding_boxes (list): Coordinates of bounding boxes ((x1,y1), (x2,y2))
    """
    PADDING = 10
    bounding_boxes = []
    for label in np.unique(cluster_labels):
        crnt_cluster = cluster_points[cluster_labels == label]
        if len(crnt_cluster) > 1:
            x1 = max(0, np.min(crnt_cluster[:, 0]) - PADDING)
            y1 = max(0, min(img.shape[0], abs(np.max(crnt_cluster[:, 1]) - img.shape[0])- PADDING))
            x2 = min(img.shape[1], np.max(crnt_cluster[:, 0]) + PADDING)
            y2 = max(0, abs(np.min(crnt_cluster[:, 1])  - img.shape[0]) + PADDING)

            bounding_boxes.append(((x1, y1), (x2, y2)))
    return bounding_boxes

def find_cluster_centers(bboxes: list):
    """
    Calculate center points of bounding boxes

    Args:
        bboxes (list): List of bounding box coordinates in the form of ((x1,y1),(x2,y2))
    
    Returns:
        cluster_centers (list): List of center coordinates in the form of (x1,y1)
    """

    cluster_centers = []
    for box in bboxes:
        x1,y1 = box[0]
        x2,y2 = box[1]

       
        center = (x1 + (x2-x1)//2 , y1 + (y2-y1)//2) 
        cluster_centers.append(center)

    return cluster_centers


def plot_boxes(image: np.ndarray, bounding_boxes: list[tuple[tuple[int,int], tuple[int,int]]])->np.ndarray:
    """
    Plot bounding boxes around each cluster on the image.

    Args:
        image (numpy.ndarray): Input image.
        bounding_boxes (list): List of bounding boxes for each cluster.
    """
    for box in bounding_boxes:
        (min_x, min_y) ,(max_x, max_y) = box
        cv2.rectangle(
            image,
            (min_x, min_y),
            (max_x, max_y),
            (0, 0, 255),
            2,
        )
    return image


def plot_centers(image: np.ndarray, cluster_centers: list[tuple[int,int]]):
    """
    Plot cluster centers on the image.

    Args:
        image (numpy.ndarray): Input image.
        cluster_centers (list): List of cluster centers.
    """
    for center in cluster_centers:
        
        x,y = center
        center_pos = (x, y)
        cv2.circle(
            image,
            center_pos,
            radius=5,
            color=(0, 0, 255),
            thickness=-1,
        )
    return image


def crop_bbox(box, image)-> np.ndarray:
    """
    Returns an array of an image defined by the specified bounding box.

    Args:
        box (tuple): A tuple containing two tuples representing the coordinates of the top-left and bottom-right corners of the bounding box.
        image (numpy.ndarray): The input image from which the sub-array is extracted.
        min_size (int): Minimum area the bounding box should have for it to be considered a full plant

    Returns:
        numpy.ndarray: An array of the image defined by the bounding box. Returns empty array if the box width or height is non-positive.
    """

    result = np.ndarray([0])
    (min_x, min_y), (max_x, max_y) = box

    box_width = abs(max_x - min_x)
    box_height = abs(max_y - min_y)

    if box_width > 0 and box_height > 0:
        result = image[min_y:max_y, min_x:max_x]
    return result

def run_segmentation(image):
    """
    Segment the given image to identify and extract green areas, perform clustering,
    and return the segmented image, bounding boxes, cluster centers, and cropped images.

    Args:
        image (numpy.ndarray): The input image in which segmentation is to be performed.

    Returns:
        img_segmented (numpy.ndarray): The image with plotted bounding boxes and centers.
        bboxes (list of tuples of tuples): List of bounding boxes in the format [((x1,y1), (x2,y2)), ...].
        centers (list of tuples): List of cluster centers in the format [(x, y), ...].
        cropped (list of numpy.ndarray): List of cropped images from the bounding boxes.
    """
    green_areas = get_green(image)
    bnw_image = green_to_bnw(green_areas)
    
    #bnw_image = green_to_bnw(image)

    low_res_bnw_image, old_new_image_ratio = refactor_to_lower_res(bnw_image, total_pixels=10000)
    white_points_low_res = binary_to_cartesian(low_res_bnw_image)

    cluster_labels, cluster_points = DBSCAN_clustering(white_points_low_res)
    cluster_points = (cluster_points * old_new_image_ratio).round().astype(int)


    bboxes = [(bbox[0], bbox[1]) for bbox in find_bounding_boxes(image, cluster_points, cluster_labels)]
    centers = [(xy[0], xy[1]) for xy in find_cluster_centers(bboxes)]

    cropped = []

    '''
    Notes to self for what this does: 
    Isolate green areas, mask them, eliminate noise in the mask, scale down to smaller size. 
    Pass scaled down black/white image to DBSCAN, which clusters white points. Scale back up to full size.
    For each cluster, construct a bounding box that includes every point in that cluster, plus padding on each side.
    For each bounding box, extract the cropped image inside that bounding box.
    '''

    img_copy = image.copy()
    for bbox in bboxes:
        result = crop_bbox(bbox, img_copy)
        cropped.append(result)
            
    img_segmented = plot_boxes(image, bboxes)
    img_segmented = plot_centers(image, centers)
    return img_segmented, bboxes, centers, cropped

def calc_iou(rect1, rect2):
    ''' 
    Given two numpy arrays of [x1,y1,x2,y2] for a rectangle, calculate the intersection-over-union of the two.
    Args:
        rect1 (np.ndarray)
        rect2 (np.ndarray)

    Returns:
        A float containing the intersection over union of the two rectangles
    '''
    x1 = max(rect1[0],rect2[0])
    y1 = max(rect1[1],rect2[1])
    x2 = min(rect1[2],rect2[2])
    y2 = min(rect1[3],rect2[3])

    intersect = (max(0,x2-x1+1))*max(0,(y2-y1+1))
    union = (1+rect1[2]-rect1[0])*(1+rect1[3]-rect1[1]) + (1+rect2[2]-rect2[0])*(1+rect2[3]-rect2[1]) - intersect
    #print(f"iou:  {intersect}/{union}")
    print(f"rect1 = {rect1}, rect2 = {rect2}, iou = {intersect / union}")
    return intersect / union
 
def update_bboxes(bbox,bbox_lim,iou_threshold,life):
    return 0

if __name__ == "__main__":
    #model = YOLO("yolo11n.pt")
    cwd = os.getcwd()
    print(cwd)
    vidcap = cv2.VideoCapture("school/code/WeedID/plant_tracking/video1.mp4")
    count = 0
    success,frame = vidcap.read()
    success = True
    box_list = -1*np.ones((100,8),dtype=np.int64) # X1, Y1, X2, Y2, Life, GetVelocity, Vx, Vy
    bbox_count = 0
    new_bbox_count = 0
    bbox_life = 2          
    iou_threshold = 0.1
    frame_limit = 300
    t_start = time.time()
    run_classification = 0
    try:
        while vidcap.isOpened():
            success,frame = vidcap.read()
            if success and count <= frame_limit:
                img_seg,bbox,centers,cropped = run_segmentation(frame)
                #results = model.track(frame, persist=True)
                #print(bbox)
                #print(bbox[0])
                #print(bbox[0][0])
                boxes = np.zeros((len(bbox),4))
                for i in range(len(bbox)):
                    boxes[i,0:4] = [bbox[i][0][0],bbox[i][0][1],bbox[i][1][0],bbox[i][1][1]]
                #print(boxes)
                #boxes = results[0].boxes.xyxy.cpu()
                free = np.where(box_list[:,4]<=-1)[0]
                getV = np.where(box_list[:,4]> -1 and box_list[:,5] == 1)[0]
                taken = np.where(box_list[:,4]> -1 and box_list[:,5] >= 1)[0]
                count2 = 0
                gotVelocity = np.zeros

                for index in range(boxes.shape[0]): # For each newly identified bbox
                    bbox_count = bbox_count + 1
                    if getV.shape[0] > 0: 
                        ious = np.zeros((getV.shape[0]))
                        for i in range(ious.shape[0]):
                            #ious[i] = calc_iou(box[0:4].numpy(),box_list[taken[i],0:4]) # ----- Use if bbox is pytorch tensor
                            ious[i] = calc_iou(boxes[index][0:4],box_list[getV[i],0:4])  # ----- Use if bbox is numpy ndarray
                        #print(ious)
                        max_i = np.argmax(ious)
                        #print(max_i)

                        # If new bbox is similar enough to existing bbox, replace old with new; otherwise, add new bbox
                        if ious[max_i] > iou_threshold: 
                            # Calculate velocity
                            vx = 0.5*(boxes[index][0]+boxes[index][2]) - 0.5*(box_list[max_i,0]+box_list[max_i,2])
                            vy = 0.5*(boxes[index][1]+boxes[index][3]) - 0.5*(box_list[max_i,1]+box_list[max_i,3])
                            box_list[max_i,0:4] = boxes[index]
                            box_list[max_i,4] = bbox_life
                            box_list[max_i,5] = 1
                            box_list[max_i,6] = vx
                            box_list[max_i,7] = vy
                            gotVelocity.append(index)
                    
                    
                    # If bounding boxes from last frame are present
                    if taken.shape[0] > 0: 
                        # Calculate IOU of current bbox with remembered bboxes
                        ious = np.zeros((taken.shape[0]))
                        for i in range(ious.shape[0]):
                        
                            #ious[i] = calc_iou(box[0:4].numpy(),box_list[taken[i],0:4]) # ----- Use if bbox is pytorch tensor
                            ious[i] = calc_iou(box[0:4],box_list[taken[i],0:4])  # ----- Use if bbox is numpy ndarray
                        #print(ious)
                        max_i = np.argmax(ious)
                        #print(max_i)

                        # If new bbox is similar enough to existing bbox, replace old with new; otherwise, add new bbox
                        if ious[max_i] > iou_threshold: 
                            box_list[max_i,0:4] = box
                        else:
                            box_list[free[count2],0:4] = box
                            box_list[free[count2],4] = bbox_life
                            box_list[free[count2],5] = 1       # We need to remember to get your velocity
                            new_bbox_count = new_bbox_count + 1
                            box_list[max_i,4] = 0

                    else:
                        box_list[free[count2],0:4] = box
                        box_list[free[count2],4] = bbox_life
                        new_bbox_count = new_bbox_count + 1

                    count2 = count2 + 1

                print(f"where: {np.where(box_list[:,4] > -1)[0]}")
                x1 = box_list[np.where(box_list[:,4] > -1),0][0]
                y1 = box_list[np.where(box_list[:,4] > -1),1][0]
                x2 = box_list[np.where(box_list[:,4] > -1),2][0]
                y2 = box_list[np.where(box_list[:,4] > -1),3][0]
                l = box_list[np.where(box_list[:,4] > -1),4][0]
                print(l)
                print(x1)

                for i in range(x1.shape[0]):
                    val = int(l[i])
                    print(val)
                    print(255,0,255-20*l[i])
                    cv2.rectangle(frame,(x1[i],y1[i]),(x2[i],y2[i]),(255,0,255 - 20*int(l[i])),2)

                #track_ids = results[0].boxes.id.int().cpu().tolist()
            
                cv2.imshow('window',frame)
                if cv2.waitKey(0) == ord("q"):
                   break
                count = count + 1
                box_list[:,4] = box_list[:,4] - 1
                box_list[np.where(box_list[:,4] < -1),4] == -1
                #print(box_list)
            else:
                print(count)
                break
    finally:
        t_end = time.time()
    '''
    t_seg_start = time.time()
    vidcap = cv2.VideoCapture("school/code/WeedID/plant_tracking/video1.mp4")
    count = 0
    try:
        while vidcap.isOpened():
            success,frame = vidcap.read()
            if success and count <= frame_limit:
                img_seg,bbox,centers,cropped = run_segmentation(frame)
                count = count + 1
            else:
                print(count)
                break
        
    finally:
        t_seg_end = time.time()
    '''
    print(f"Classify per bbox = {new_bbox_count/bbox_count}, Classify count: {new_bbox_count}, Bbox count: {bbox_count}\nTime: {t_end - t_start} s")
    print(f"Time with only segmentation: {t_seg_end - t_seg_start} s")

    #results = model.track(source="school/code/WeedID/plant_tracking/testvideo.mp4",show=True,tracker="bytetrack.yaml",vid_stride=1)
    #print(results)

