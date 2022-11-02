from object_counting_v1 import ObjectCounting_v1 as oc

model_path = r'path to saved_model'
label_map_path = r'path to label_map.pbtxt'
image_path = r'path to image'
labels = "TREE"
image_resize_division = 2
small_object = 1
#small_object = True
#image_path = r'path to image'

oc.count_objects(model_path, labels, label_map_path, image_path, small_object, image_resize_division)
