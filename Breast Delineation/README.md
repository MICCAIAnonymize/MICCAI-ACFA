## Breast mask generation using BreastDivider (nnU Net)

BreastDivider provides a pretrained nnU Net model for 3D breast MRI that segments the breast region and separates left and right breasts. We use it to generate a breast mask for our pipeline.

### Option A, Recommended, Docker inference

1. Install Docker. For faster inference, use a machine with an NVIDIA GPU and NVIDIA Container Toolkit.

2. Prepare the input folder
   - Place your 3D breast MRI volumes in NIfTI format, for example `.nii.gz`, inside a single folder.

3. Pull the Docker image

   ```bash
   docker pull ykirchhoff/breastdivider
   ```
    
4. Run inference (mount input and output folders)
 ```bash
   docker run --ipc=host --rm --gpus all \
  -v "/path/to/input/folder:/mnt/input" \
  -v "/path/to/output/folder:/mnt/output" \
  ykirchhoff/breastdivider
   ```
5. Collect the predicted masks

  The output folder will contain the predicted segmentation masks produced by the model.
  
6. Create the breast mask you need
   If your downstream pipeline expects a single binary breast mask, convert the model output into one mask by merging the left and right breast labels into a single foreground mask (keep everything else as background).

