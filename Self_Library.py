import os
import shutil
import glob
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from mpi4py import MPI
import torch
import torchvision
import random
import numpy as np
from mpi4py import *
import mpi4py
import sigpy as sp
import sigpy.mri as mr
import sigpy.plot as pl
import matplotlib.pyplot as plt
# import torchvision.transforms
import pydicom
import cv2
import os
import scipy.io as sio
import tensorflow as tf
from scipy.interpolate import griddata
from scipy.ndimage import rotate
from torchvision import transforms
import nibabel as nib
from sklearn.model_selection import GridSearchCV
from torch.optim.lr_scheduler import StepLR



# Function used to split the dataset into training and testing images.
def train_test_split(split_ratio,img_directory):    
    
    all_idx = np.array(random.sample(range(0, len(img_directory)), len(img_directory)))
    train_elements = int(split_ratio*len(img_directory))
    train_idx = random.sample(range(0, len(img_directory)), train_elements)
    train_idx = np.array(train_idx)
    test_idx = np.setdiff1d(all_idx,train_idx)
    train_imgs = img_directory[train_idx]
    test_imgs = img_directory[test_idx]
    
    return (train_imgs,test_imgs)

# Function to create the downsampling pattern
def Downsampling_pattern(R,array_length,channels): 
    
    """"
    R: Downsampling Acceleration Rate
    array_length: Total Number of Elements in the Dicom Array
    channels: Number of channels in the k-space
    """
    
    result = []

    # iterating
    for i in range(array_length):
       # checking the index
       if i % R == 0:
          # appending 1 on even index
          result.append(1)
       else:
          # appending 0 on odd index
          result.append(0)
    result = np.array(result)
    result[(int(len(result)/2)-12):(int(array_length/2)+12)] = np.ones(24)
    result = np.tile(result,(channels, array_length,1)) #Generates (16,256,256) matrix
    for i in range(0,channels):    #Changing downsampling pattern from vertical to horizontal
        result[i] = result[i].T    # Use ".T" to change the downsampling pattern to change from vertical to horizontal
    return result     #Use pl.ImagePlot to plot any of the channel from result and visualize

# Function to extract k-space from 2D image
def extract_k_space(image):
    return  np.fft.fftshift(np.fft.fft2(np.fft.fftshift(image)))

# Function to apply inverse Fourier transform and reconstruct image from k-space
def reconstruct_image(k_space):
    return np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(k_space)))

# Function to convert Fully Sampled Data Generated from COMSOL to Undersampled data for AI Model
def Downsampling_MRE_Data(R, spatial_dimensions, slices, phase_offsets, fully_sampled_file_path,output_file_path):
    downsampling_pattern = (Downsampling_pattern(R,256,slices))
    downsampling_pattern = downsampling_pattern.transpose(1,2,0)
    new_pattern = np.expand_dims(downsampling_pattern, axis=-1)
    new_pattern = (np.repeat(new_pattern, phase_offsets, axis=-1))
    mri_data_dict = sio.loadmat(fully_sampled_file_path)
    mri_data = mri_data_dict["z_df"]
    mri_data = mri_data[:, :,:,:]
    width = spatial_dimensions
    height = spatial_dimensions
    

    downsampled_images = np.empty((width, height, slices, phase_offsets))     #, dtype=np.complex64) Use this is np.empty when doing downsampling
    for offset in range(phase_offsets):
        for slice_num in range(slices):
            # Extract the 2D slice for the current offset and slice
            current_slice = mri_data[:, :, slice_num, offset]
            max_value = np.max(current_slice)

            # Commented the lines below, because I will be adding noise while creating training and testing data
            # noise_std = 0.02 * max_value
            # noise =  np.random.normal(0, noise_std, (256,256))
            current_slice = current_slice #+noise
            # Commenting ends right here
            
            # Extract k-space data for the current 2D slice
            # k_space = extract_k_space(current_slice)                                # Uncomment if you want to do downsampling

            # Apply the downsampling pattern to the k-space data                    
            # k_space_downsampled = k_space * new_pattern[:,:,slice_num,offset]       # Uncomment if you want to do downsampling

            # Reconstruct the downsampled image from the downsampled k-space
            # downsampled_image = reconstruct_image(k_space_downsampled)              # Uncomment if you want to do downsampling

            # Store the downsampled image
            downsampled_images[:, :, slice_num, offset] = current_slice           # Replace current_slice with downsampled_image if we enable downsampling
    
    # Create a dictionary with a key for the 'downsampled_images' array
    data_dict = {os.path.splitext(os.path.basename(output_file_path))[0]: downsampled_images}
    # Save the 'data_dict' to the .mat file
    sio.savemat(output_file_path, data_dict)
    return downsampled_images

# Function to present three images side by side
def display_real_tensors(tensor1, tensor2, tensor3):
    # Convert tensors to numpy arrays
    array1 = tensor1.numpy().real
    array2 = tensor2.numpy().real
    array3 = tensor3.numpy().real

    # Create a figure with subplots
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    # Display real parts of arrays in the subplots
    axs[0].imshow(array1, cmap='jet')
    axs[0].set_title('Reconstructed Stiffness')

    axs[1].imshow(array2, cmap='jet')
    axs[1].set_title('Actual Stiffness')

    axs[2].imshow(array3, cmap='gray')
    axs[2].set_title('Undersampled input')

    # Adjust spacing between subplots
    plt.tight_layout()

    # Show the plot
    plt.show()

# Function to remove previosly existing training and testing data
def delete_items_in_folders(folder_paths):
    for folder_path in folder_paths:
        try:
            if os.path.exists(folder_path) and os.path.isdir(folder_path):
                items = os.listdir(folder_path)
                for item in items:
                    item_path = os.path.join(folder_path, item)
                    if os.path.isfile(item_path):
                        os.remove(item_path)
                    elif os.path.isdir(item_path):
                        os.rmdir(item_path)
                # Optionally, you can delete the folder itself
                # os.rmdir(folder_path)  # Uncomment if desired
            else:
                print(f"Folder does not exist: {folder_path}")
        except Exception as e:
            print(f"Error while deleting items in {folder_path}: {str(e)}")

# Function to move testing data to desired locations
def move_items_to_folder(source_folder, destination_folder):
    try:
        # Verify that the source folder exists
        if os.path.exists(source_folder) and os.path.isdir(source_folder):
            # Verify that the destination folder exists; create it if it doesn't
            if not os.path.exists(destination_folder):
                os.makedirs(destination_folder)
            
            # List all items (files and subfolders) in the source folder
            items = os.listdir(source_folder)
            
            # Iterate through the items and move them to the destination folder
            for item in items:
                item_path = os.path.join(source_folder, item)
                destination_path = os.path.join(destination_folder, item)
                if os.path.isfile(item_path):
                    shutil.move(item_path, destination_path)  # Move file
                elif os.path.isdir(item_path):
                    shutil.move(item_path, destination_path)  # Move subfolder
        else:
            print(f"Source folder does not exist: {source_folder}")
    except Exception as e:
        print(f"Error while moving items: {str(e)}")

