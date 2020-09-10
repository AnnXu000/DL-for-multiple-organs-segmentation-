# DL-for-multiple-organs-segmentation-
Deep learning for segmentation of multiple organs from radiotherapeutics applications

# About Data
The data was collected from: The Cancer Imaging Archive (TCIA)

My google drive directory link (view only):
https://drive.google.com/drive/folders/1SjknpV7FsqoT5L-4_H3zUp0GRywgOfZv

# Imaging Data Description

1.All files are stored in Nifti format with 32-bit floating-point data, which includes one folder for CT images named ”all datasets“ and one folder for labels called “labels”, total size of the dataset is 33 GB after unzipping.

2.The number of CT pathograms and the corresponding ground truth images is 140 each. CT pathograms are named in the format of “volume-XX.nii.gz” and the ground truth images are named as “labels-XX.nii.gz”, where XX is the case number and ranges from 0 to 139.

3.As all the scans were formated in NIfTI format (i.e. .nii.gz), so we have used SimpleITK library for converting .nii.gz format to numpy array.

4.Provided labelled data was done by professional physicians.

# Task
Segmentation of multiple organs from radiotherapeutics applications. Use the provided clinically-acquired training data to produce segmentation labels. Compare and analysis the two modules in the project.

# Data Pre-Processing
The dimensions of the data set are (N, H, W, X). Here N is the total amount of data from the CT images, H, W is the size of each 2D slice, and X is the z-axis’ dimension. The shape of our dataset is (140, 512, 512, 1).
So, here as Google Colab Free GPU was used to do all pre-processing and training.

# Proposed Model
Here we have proposed U-Net and GhostUnet for our semnatic segmentation problem.

# Dice Coefficient & Dice Coefficient Loss Function

Dice = 2* |X∩Y| / (|X|+|Y|)

Here |X| and |Y| are the cardinalities of the two sets (i.e. the number of elements in each set). The Sørensen index equals twice the number of elements common to both sets divided by the sum of the number of elements in each set.

The implemeted python code for dice coefficient:-

def dice_coef(y_true, y_pred, epsilon=1e-6):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection) / (K.sum(K.square(y_true),axis=-1) + K.sum(K.square(y_pred),axis=-1) + epsilon)
    
In order to formulate a loss function which can be minimized, we'll simply use 1−dice_coef:-

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)
    
# Results
Network model	   Global Dice%	      Iteration Numbers till converage
U-Net	            93.10	            4750
GhostUnet	        92.78	            5000

# Conclusions
Both neural networks showed good automatic image segmentation ability, with U-Net having a global Dice of 93.1%, which was better than GhostUnet (92.78%); however, GhostUnet's Dice for automatic sketching of bladder was slightly higher than U-Net, which indicated that this network model might be useful to radiotherapy but not very effective at the moment, and at least it can achieve automatic segmentation of multiple organs.
