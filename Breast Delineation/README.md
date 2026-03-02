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

   - The output folder will contain the predicted segmentation masks produced by the model.
  
6. Create the breast mask you need
   If your downstream pipeline expects a single binary breast mask, convert the model output into one mask by merging the left and right breast labels into a single foreground mask (keep everything else as background).

### Alternative, Manual nnU Net inference, no Docker

1. **Install nnU Net v2**
   - Follow the official nnU Net installation instructions.

2. **Download the pretrained model**
   - Download `BreastDividerModel` from the BreastDivider resources on Hugging Face.

3. **Run prediction from the downloaded model folder**
      ```bash
      nnUNetv2_predict_from_modelfolder \
        -i input_folder \
        -o output_folder \
        -m /path/to/BreastDividerModel
      ```
4. Postprocess to your desired mask format
`  - As above, merge left and right into a single breast mask if your workflow needs one.

## Reference

If you use BreastDivider in your work, please cite the following paper.
Rokuss M, Hamm B, Kirchhoff Y, Maier Hein K. *Divide and Conquer: A Large Scale Dataset and Model for Left and Right Breast MRI Segmentation*. arXiv:2507.13830, 2025.

```bibtex
@article{rokuss2025breastdivider,
  title   = {Divide and Conquer: A Large-Scale Dataset and Model for Left and Right Breast MRI Segmentation},
  author  = {Rokuss, Maximilian and Hamm, Benjamin and Kirchhoff, Yannick and Maier-Hein, Klaus},
  journal = {arXiv preprint arXiv:2507.13830},
  year    = {2025}
}
