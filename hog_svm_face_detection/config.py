import configparser
import json

config = configparser.RawConfigParser()
config.read('./ComputerVisionProject_FaceDetection/config/config.cfg')

# HOG Feature directory paths
pos_feat_dir = config.get("paths", "pos_feat_dir")
neg_feat_dir = config.get("paths", "neg_feat_dir")

# SVM model path
model_path = config.get("paths", "model_path")

# Parameters
min_window_size = json.loads(config.get("hog","min_window_size"))
step_size = json.loads(config.get("hog", "step_size"))
orientations = config.getint("hog", "orientations")
pixels_per_cell = json.loads(config.get("hog", "pixels_per_cell"))
cells_per_block = json.loads(config.get("hog", "cells_per_block"))
visualize = config.getboolean("hog", "visualize")
normalize = config.get("hog", "normalize")
threshold = config.getfloat("nms", "threshold")
