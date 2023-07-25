# Importing libraries
from skimage.feature import hog
from skimage.io import imread
import joblib
import argparse as ap
import glob
import os
from skimage.transform import  resize
import sys
# sys.path = sys.path + ['/data/temp_Sagar/temp/object-detector']
from config import *

if __name__ == "__main__":
    # Argument Parser
    parser = ap.ArgumentParser()
    parser.add_argument('-p', "--pos_dir", help="Positive images directory path",
            required=True)
    parser.add_argument('-n', "--neg_dir", help="Negative images directory path",
            required=True)
    args = vars(parser.parse_args())

    pos_im_dir = args["pos_dir"]
    neg_im_dir = args["neg_dir"]

    # Create directories to store features if not already present
    feature_path_list = [pos_feat_dir ,neg_feat_dir]
    for dir_name in feature_path_list :
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)

    print ("Calculate HOG descriptors for the positive images and store them")
    for img_path in glob.glob(os.path.join(pos_im_dir, "*")):
        img = imread(img_path, as_gray=True)
        img = resize(img, (min_window_size[0], min_window_size[1]),anti_aliasing=True)
        feature_hog = hog(img, orientations, pixels_per_cell, cells_per_block, visualize = visualize, block_norm = normalize)
        fd_name = os.path.split(img_path)[1].split(".")[0] + ".feat"
        fd_path = os.path.join(pos_feat_dir, fd_name)
        joblib.dump(feature_hog, fd_path)
    print ("Positive features saved in {}".format(pos_feat_dir))

    print ("Calculate HOG descriptors for the negative images and store them")
    for img_path in glob.glob(os.path.join(neg_im_dir, "*")):
        img = imread(img_path, as_gray=True)
        img = resize(img, (min_window_size[0], min_window_size[1]),anti_aliasing=True)
        feature_hog = hog(img,  orientations, pixels_per_cell, cells_per_block, visualize = visualize, block_norm = normalize)
        fd_name = os.path.split(img_path)[1].split(".")[0] + ".feat"
        fd_path = os.path.join(neg_feat_dir, fd_name)
        joblib.dump(feature_hog, fd_path)
    print ("Negative features saved in {}".format(neg_feat_dir))

    print ("Done...")
