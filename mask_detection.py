import cv2
from imutils.video import VideoStream
from main import Main
import numpy as np
import os
import sys
import tarfile
import tensorflow as tf
import zipfile
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from distutils.version import StrictVersion
import datetime
from io import StringIO

from PIL import Image


main = Main()
detection_graph = main.graph()
category_index = main.index()
print(category_index)
cap = VideoStream(0).start()
start_time = datetime.datetime.now()
num_frames = 0
with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        while True:
            image_np = cap.read()
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name(
                'num_detections:0')
            # Actual detection.
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections], feed_dict={image_tensor: image_np_expanded})
            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8)
            num_frames += 1
            elapsed_time = (datetime.datetime.now() -
                            start_time).total_seconds()
            fps = (num_frames / elapsed_time)
            fps = str(round(fps, 2))
            cv2.putText(image_np, f"fps:{fps}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (77, 255, 9), 2)

            cv2.imshow('Face Mask Detection', cv2.resize(image_np, (800, 600)))

            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

cap.release()
