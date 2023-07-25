# Import the required modules
from sklearn.svm import LinearSVC
import joblib
import argparse as ap
import glob
import os
from config import *

if __name__ == "__main__":
    
    # Parse the command line arguments
    parser = ap.ArgumentParser()
    parser.add_argument('-p', "--pos_feat", help="Positive sample image hog features directory path", required=True)
    parser.add_argument('-n', "--neg_feat", help="Negative sample image hog features directory path", required=True)
    args = vars(parser.parse_args())

    pos_feat_dir =  args["pos_feat"]
    neg_feat_dir = args["neg_feat"]

    feature_list = []
    labels = []
    # Load the positive features
    for feat_path in glob.glob(os.path.join(pos_feat_dir,"*.feat")):
        feat = joblib.load(feat_path)
        feature_list.append(feat)
        labels.append(1)

    # Load the negative features
    for feat_path in glob.glob(os.path.join(neg_feat_dir,"*.feat")):
        feat = joblib.load(feat_path)
        feature_list.append(feat)
        labels.append(0)

    # Training the SVM Classifier
    clf = LinearSVC()
    print("Training SVM ...")
    clf.fit(feature_list, labels)
    
    # Model Directory
    if not os.path.isdir(os.path.split(model_path)[0]):
        os.makedirs(os.path.split(model_path)[0])
    joblib.dump(clf, model_path)
    print("Classifier saved to {}".format(model_path))
