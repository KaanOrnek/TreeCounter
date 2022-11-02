from heapq import merge
from typing_extensions import Self
import cv2
import numpy as np
import argparse
import tensorflow as tf
import dlib
from matplotlib import pyplot as plt
from object_detection.utils import label_map_util
from object_detection.utils import ops as utils_ops

from trackable_object import TrackableObject
from centroidtracker import CentroidTracker

class ObjectCounting_v1(object):
    # patch tf1 into `utils.ops`
    utils_ops.tf = tf.compat.v1

    # Patch the location of gfile
    tf.gfile = tf.io.gfile

    def count_objects(model_path, labels, label_map_path, image_path, small_object, image_resize_division):
        
        #INNER METHODS

        def extract_crops(img, crop_height, crop_width, step_vertical=None, step_horizontal=None):

            img_height, img_width = img.shape[:2]
            crop_height = min(crop_height, img_height)
            crop_width = min(crop_width, img_width)

            crops = []
            crops_boxes = []

            if not step_horizontal:
                step_horizontal = crop_width
            if not step_vertical:
                step_vertical = crop_height

            height_offset = 0
            last_row = False
            while not last_row:
            # If we crop 'outside' of the image, change the offset
            # so the crop finishes just at the border if it
                if img_height - height_offset < crop_height:
                    height_offset = img_height - crop_height
                    last_row = True
                last_column = False
                width_offset = 0
                while not last_column:
                    # Same as above
                    if img_width - width_offset < crop_width:
                        #width_offset = img_width - crop_width
                        last_column = True
                        ymin, ymax = height_offset, height_offset + crop_height
                        xmin, xmax = width_offset, img_width
                        a_crop = img[ymin:ymax, xmin:xmax]
                        crops.append(a_crop)
                        crops_boxes.append((ymin, xmin, ymax, xmax))
                    else:
                        ymin, ymax = height_offset, height_offset + crop_height
                        xmin, xmax = width_offset, width_offset + crop_width
                        a_crop = img[ymin:ymax, xmin:xmax]
                        crops.append(a_crop)
                        crops_boxes.append((ymin, xmin, ymax, xmax))
                        width_offset += step_horizontal
                height_offset += step_vertical
            return crops, crops_boxes
            #return np.stack(crops, axis=0), crops_boxes

        def load_model(model_path):
            tf.keras.backend.clear_session()
            model = tf.saved_model.load(model_path)
            return model
        
        #DETECTION AND COUNTING

        counter = [0]
        total_frames = 0

        ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
        trackers = []
        trackableObjects = {}

        #load model - label map - image
        model = load_model(model_path)
        label_map = label_map_util.load_labelmap(label_map_path)
        categories = label_map_util.convert_label_map_to_categories(label_map, 1, use_display_name=True)
        category_index = label_map_util.create_category_index(categories)
        
        detection_count = 0

        image = cv2.imread(image_path)
        height, width, _ = image.shape
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        rects = []

        #if the image has small objects, small_object
        #must be specified as true in the function call
        #this part counts the objects by cutting the
        #original image into smaller pieces to increase
        #the accuracy. 
        if small_object == True:

            if height < width:
                crop_height = (height // 2) + 1
            else:
                crop_height = (width // 2) + 1
            crop_width = crop_height
            crop_size = crop_width
            crop_step_vertical = crop_step_horizontal = crop_size 
            #- 20

            crops, crops_coordinates = extract_crops( image, crop_height, crop_width, crop_step_vertical, crop_step_horizontal)
            

            length = (len(crops) // 2) - 1

            #crop_row_no = 0
            #crop_col_no = 0
            crop_counter = 0
            for a_crop in crops:
                
                a_crop = np.asarray(a_crop)
                tensor_crop = tf.convert_to_tensor(np.expand_dims(a_crop,0))
                output = model(tensor_crop)

                num_detections = int(output.pop('num_detections'))
                output = {key: value[0, :num_detections].numpy()
                               for key, value in output.items()}
                output['num_detections'] = num_detections

                output['detection_classes'] = output['detection_classes'].astype(np.int64)

                treshold = 0.5

                #print("checkpt-1")
                for i, (y_min, x_min, y_max, x_max) in enumerate(output['detection_boxes']):
                    if output['detection_scores'][i] > treshold and (labels == None or category_index[output['detection_classes'][i]]['name'] in labels):
                       detection_count += 1
                       tracker = dlib.correlation_tracker()

                       rect = dlib.rectangle((( crops_coordinates[crop_counter][1]) + int(x_min * crop_size)),((crops_coordinates[crop_counter][0]) + int(y_min * crop_size)), ((crops_coordinates[crop_counter][1]) + int(x_max * crop_size)),((crops_coordinates[crop_counter][0]) + int(y_max * crop_size)))
                       #rect = dlib.rectangle((( crop_row_no * crop_size) + int(x_min * crop_size)),((crop_col_no * crop_size) + int(y_min * crop_size)), ((crop_row_no * crop_size) + int(x_max * crop_size)),((crop_col_no * crop_size) + int(y_max * crop_size)))
                       tracker.start_track(rgb, rect)
                       trackers.append(tracker)
                
                crop_counter = crop_counter + 1
                
                #print("checkpt-2")
            for tracker in trackers:
                        # update the tracker and grab the updated position
                tracker.update(rgb)
                pos = tracker.get_position()

                     # unpack the position object
                x_min, y_min, x_max, y_max = int(pos.left()), int(
                pos.top()), int(pos.right()), int(pos.bottom())
                    # add the bounding box coordinates to the rectangles list
                rects.append((x_min, y_min, x_max, y_max))

            objects = ct.update(rects)  

            for (objectID, centroid) in objects.items():
                to = trackableObjects.get(objectID, None)

                if to is None:
                    to = TrackableObject(objectID, centroid)
                elif not to.counted:
                    counter[0] += 1
                    to.counted = True
                    to.centroids.append(centroid)

                trackableObjects[objectID] = to
                text = "ID {}".format(objectID)
                cv2.putText(image, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, (0.3 * image_resize_division), (255, 255, 255), (1* image_resize_division))
                cv2.circle(image, (centroid[0], centroid[1]), 4, (255, 255, 255), -1)

                #keep track of where the current crop in the original image is
                
                #crop_col_no = crop_col_no + 1
                #if crop_col_no == length:
                #    crop_col_no = 0
                #    crop_row_no = 1
           
            
        else:
            image = np.asarray(image)
            tensor_image = tf.convert_to_tensor(np.expand_dims(image,0))
            
            output = model(tensor_image)
            num_detections = int(output.pop('num_detections'))
            output = {key: value[0, :num_detections].numpy()
                           for key, value in output.items()}
            output['num_detections'] = num_detections

            output['detection_classes'] = output['detection_classes'].astype(
                 np.int64)

            treshold = 0.5

            for i, (y_min, x_min, y_max, x_max) in enumerate(output['detection_boxes']):
                if output['detection_scores'][i] > treshold and (labels == None or category_index[output['detection_classes'][i]]['name'] in labels):
                    detection_count += 1
                    tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(int(x_min * width), int(y_min * height), int(x_max * width), int(y_max * height))
                    tracker.start_track(rgb, rect)
                    trackers.append(tracker)

            for tracker in trackers:
                    # update the tracker and grab the updated position
                tracker.update(rgb)
                pos = tracker.get_position()

                    # unpack the position object
                x_min, y_min, x_max, y_max = int(pos.left()), int(
                pos.top()), int(pos.right()), int(pos.bottom())
                    # add the bounding box coordinates to the rectangles list
                rects.append((x_min, y_min, x_max, y_max))

            objects = ct.update(rects)

            for (objectID, centroid) in objects.items():
                to = trackableObjects.get(objectID, None)

                if to is None:
                    to = TrackableObject(objectID, centroid)
                elif not to.counted:
                    counter[0] += 1
                    to.counted = True
                    to.centroids.append(centroid)

                trackableObjects[objectID] = to
                text = "ID {}".format(objectID)
                cv2.putText(image, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, (0.5 * image_resize_division), (255, 255, 255), (2* image_resize_division))
                cv2.circle(image, (centroid[0], centroid[1]), 4, (255, 255, 255), -1)

            
        image = cv2.resize(image, (int(image.shape[1]/image_resize_division), int(image.shape[0]/image_resize_division)))
        cv2.putText(image, f'No. of Trees: {detection_count}', (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0xFF, 0xFF), 2, cv2.FONT_HERSHEY_SIMPLEX)
        cv2.imshow('No. of Trees - Object Counting', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()