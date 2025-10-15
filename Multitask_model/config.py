
import os
import torch
# ======================
# CONFIGURATION
# ======================
SEG_DIR = '/home/tibia/Projet_Hemorragie/Seg_hemorragie/split_MONAI'
CLASSIFICATION_DATA_DIR = '/home/tibia/Projet_Hemorragie/MBH_label_case'
NUM_CLASSES = 6
CLASS_NAMES = ['any', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']
num_epochs = 1000
batch_size = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR = "/home/tibia/Projet_Hemorragie/MBH_multitask_slidingggro_patch"
os.makedirs(SAVE_DIR, exist_ok=True)