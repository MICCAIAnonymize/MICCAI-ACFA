# nnUNet inference guide

This repository provides a pretrained nnUNet baseline for breast tumour segmentation on the DCE MRI dataset. Use the instructions below to set up nnUNet, download the pretrained weights, preprocess your MRI volumes, and run inference.

## Pretrained model link

You can access the trained nnUNet weights on here:  
[https://www.synapse.org/Synapse:syn61247992] 

The official MAMA MIA dataset project on Synapse:  
[https://github.com/LidiaGarrucho/nnUNet/blob/master/documentation/installation_instructions.md]

## Step-by-step instructions

### Running Inference with nnUNet pre-trained model
The pre-trained vanilla nnUNet model was trained on 1506 full-image DCE-MRIs and expert segmentations from the MAMA-MIA dataset. The metrics correspond to Dice Coefficient, Intersection over Union (IoU), the 95 Percentile of Hausdorff Distance (HD95) and to the Mean Surface Distance (MSD). The distances are in mm.

### Step 1. Clone the repository
Clone the repository and enter the nnUNet folder.

### Step 2. Install the necessary dependencies
To run the pre-trained nnUNet model, follow the installation instructions.

### Step 3. Download the pre-trained weights
The nnUNet pretrained weights can be dowloaded from Synapse. Unzip the folder inside nnUNet GitHub repository under nnUNet/nnunetv2/nnUNet_results.

### Step 4. (Recommended) Preprocess your input MRI images
The recommended preprocessing steps to get optimum performance are:
* z-score normalization. For DCE-MRI, use the mean and standard deviation of all the phases (from pre to last post-contrast) to z-score the DCE-MRI sequence.
* isotropic pixel spacing. The MRIs were resampled using a uniform pixel spacing of [1,1,1].
* 
### Step 5. Run the nnUNet inference
nnUNetv2_predict -i /path/to/your/images -o /path/to/output -d 101 -c 3d_fullres
* Replace /path/to/your/images with the directory containing your input images.
* Replace /path/to/output with the directory where you want to save the output segmentations.
  *Note: An error might arise if your images are not in compressed NifTI format (.nii.gz).


@article{garrucho2025,
  title={A large-scale multicenter breast cancer DCE-MRI benchmark dataset with expert segmentations},
  author={Garrucho, Lidia and Kushibar, Kaisar and Reidel, Claire-Anne and Joshi, Smriti and Osuala, Richard and Tsirikoglou, Apostolia and Bobowicz, Maciej and Riego, Javier del and Catanese, Alessandro and Gwoździewicz, Katarzyna and Cosaka, Maria-Laura and Abo-Elhoda, Pasant M and Tantawy, Sara W and Sakrana, Shorouq S and Shawky-Abdelfatah, Norhan O and Salem, Amr Muhammad Abdo and Kozana, Androniki and Divjak, Eugen and Ivanac, Gordana and Nikiforaki, Katerina and Klontzas, Michail E and García-Dosdá, Rosa and Gulsun-Akpinar, Meltem and Lafcı, Oğuz and Mann, Ritse and Martín-Isla, Carlos and Prior, Fred and Marias, Kostas and Starmans, Martijn P A and Strand, Fredrik and Díaz, Oliver and Igual, Laura and Lekadir, Karim},
  journal = {Scientific Data},
  year = {2025},
  doi = {10.1038/s41597-025-04707-4},
  pages = {453},
  number = {1},
  volume = {12}
}
