import numpy as np
import os
import sys
import tarfile
import tensorflow as tf
import zipfile
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import warnings

from object_detection.utils import ops as utils_ops


class Main:
    def __init__(self):
        # Path to frozen detection graph. This is the actual model that is used for the object detection.
        self.PATH_TO_FROZEN_GRAPH = os.getcwd() + '/inference_graph/frozen_inference_graph.pb'
        # List of the strings that is used to add correct label for each box.
        self.PATH_TO_LABELS = os.getcwd() + '/training/labelmap.pbtxt'

    def graph(self):
        print("> ====== Loading frozen graph into memory")
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.PATH_TO_FROZEN_GRAPH, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return detection_graph

    def index(self):
        category_index = label_map_util.create_category_index_from_labelmap(
            self.PATH_TO_LABELS, use_display_name=True)
        return category_index
