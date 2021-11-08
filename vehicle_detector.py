import os
os.environ["DARKNET_PATH"] = "/home/alyaan/code/darknet/"
import glob
import random
import time
import cv2
import numpy as np
from darknet import darknet
##PYTHONUNBUFFERED=1;DARKNET_PATH=/home/alyaan/code/darknet/


def load_images(images_path):
    """
    If image path is given, return it directly
    For txt file, read it and return each line as image path
    In other case, it's a folder, return a list with names of each
    jpg, jpeg and png file
    """
    input_path_extension = images_path.split('.')[-1]
    if input_path_extension in ['jpg', 'jpeg', 'png']:
        return [images_path]
    elif input_path_extension == "txt":
        with open(images_path, "r") as f:
            return f.read().splitlines()
    else:
        return glob.glob(
            os.path.join(images_path, "*.jpg")) + \
            glob.glob(os.path.join(images_path, "*.png")) + \
            glob.glob(os.path.join(images_path, "*.jpeg"))


def image_detection(image_path, network, class_names, class_colors, thresh):
    # Darknet doesn't accept numpy images.
    # Create one with image we reuse for each detect
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 3)

    image = cv2.imread(image_path)
    print ("inside detections")
    #print (image.shape)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height),
                               interpolation=cv2.INTER_LINEAR)

    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
    print ("**************")
    print (detections)
    darknet.free_image(darknet_image)
    image = darknet.draw_boxes(detections, image_resized, class_colors)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), detections

def image_classification(image, network, class_names):
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height),
                                interpolation=cv2.INTER_LINEAR)
    darknet_image = darknet.make_image(width, height, 3)
    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = darknet.predict_image(network, darknet_image)
    predictions = [(name, detections[idx]) for idx, name in enumerate(class_names)]
    darknet.free_image(darknet_image)
    return sorted(predictions, key=lambda x: -x[1])


def convert2relative(image, bbox):
    """
    YOLO format use relative coordinates for annotation
    """
    x, y, w, h = bbox
    height, width, _ = image.shape
    return x/width, y/height, w/width, h/height


def save_annotations(name, image, detections, class_names):
    """
    Files saved with image_name.txt and relative coordinates
    """
    #file_name = name.split(".")[:-1][0] + ".txt"
    #file_name = (name.split(".")[-1][0]).split("/")[-1] + ".txt"
    file_name = name.split(".")[0].split("/")[-1]
    #print ("name %s" %name)
    file_path = os.path.join("output",file_name)
    print (file_path)
    with open(file_path, "w") as f:
        for label, confidence, bbox in detections:
            #print ("bounding box ")
            #print (bbox)
            x, y, w, h = convert2relative(image, bbox)
            label = class_names.index(label)
            f.write("{} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}\n".format(label, x, y, w, h, float(confidence)))


def main():
    random.seed(3)  # deterministic bbox colors
    config_file = "darknet/weights/truck_load/yolov4-custom.cfg"
    data_file = "data/obj.data"
    weights = "darknet/weights/truck_load/yolov4-custom_final.weights"
    input_img = "/media/alyaan/hdd/storage/truck_load/ground_truth/annotations/1620184030362.jpeg"
    image_folder = "/media/alyaan/hdd/storage/truck_load/ground_truth/annotations/"
    thresh = 0.5
    batch_size = 1
    ext_output = True

    image_list = load_images(image_folder)
    print (len(image_list))
    print (image_list[0])
    network, class_names, class_colors = darknet.load_network(
        config_file,
        data_file,
        weights,
        batch_size = batch_size)

    #images = load_images(input_path)

    prev_time = time.time()

    for image in image_list:
        print (image)
        image_yolo, detections = image_detection(image, network, class_names, class_colors, thresh)
        save_annotations(image, image_yolo, detections, class_names)
        darknet.print_detections(detections, ext_output)


    #image_yolo, detections = image_detection(input_img, network, class_names, class_colors, thresh)

    #print (image_yolo.shape)
    orig_image = cv2.imread(input_img)
    #print ("#############################")
    #print (detections)

    #save_annotations(input_img, image_yolo, detections, class_names)
    #darknet.print_detections(detections, ext_output)
    #output_image_name = "inference_" + input_img.split("/")[-1]
    #output_image_path = os.path.join("output",output_image_name)
    #print (output_image_name)
    #print (image_yolo.shape)
    #cv2.imwrite(output_image_path,image_yolo,[int(cv2.IMWRITE_JPEG_QUALITY),100])


if __name__ == "__main__":
    main()
