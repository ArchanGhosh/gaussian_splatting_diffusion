# Using 2D Gaussian Splatting in Diffusion Models

>>> Abstract : Diffusion models have achieved remarkable success in image synthesis but remain computationally expensive due to iterative denoising in high-dimensional pixel space. Latent Diffusion Models (LDMs) alleviate this cost through compressed neural representations, yet often sacrifice high frequency spatial detail during latent compression. In this
work, we introduce a diffusion framework based on 2D Gaussian Splatting, replacing dense pixel-space representations with structured Gaussian primitives. Our approach employs an encoder-diffusion-renderer pipeline, where input images are first mapped into a compact Gaussian latent space parameterized by spatial location, scale, opacity, and color attributes. Diffusion is then performed directly over these Gaussian latents, enabling efficient generation while preserving fine-grained spatial structure. The denoised Gaussian representations are subsequently rendered back into the image domain using differentiable splatting. By operating on compact geometric primitives rather than dense latent tensors, our framework significantly reduces computational overhead while maintaining visual fidelity. 


# Paper

The Pre-print version of the Work can be found at [Here](https://doi.org/10.5281/zenodo.20357255)

# 1. Requirements

All the components have been written using pytorch and torchvision.

```python version > 3.11 ```

To install torch please follow the official guide on [pytorch.org](https://pytorch.org/get-started/locally/)

Example: ```pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126 ```

#### Note: It is recommended to Install Pytorch from official website with version >=2.12. Due to Hardware differences. The only other library used is Matplotlib that is listed in requirements.txt

# Running The code:

#### NOTE : All the configuration parameters are set inside of ```SRC/CONFIG.PY```. Please Ensure that before changing any parameter (especially training params) the other params are recalculated to avoid any memory, or overlapping issues.

## Training the Gaussian Splat Encoder

#### Data : Currently the model is trained on a subset of STL10 images. These can be easily changed by changing or adding a similar data loader file like ```src/stl10_loader.py ```. And replace the corresponding function inside the training file.

To Run the GS Encoder training pipeline, from the base folder you can simple do 

```python/python3 -m src.training_splat_encoder```.


Training data is auto-downloaded.


All the images generated during training should be available in the ```splat_encoder_training_imgs``` folder created automatically during the training process.

The weights of the encoder are saved in the ``` CHECKPOINTS ```, along with that of the renderer. (Rendered weights are only saved for verification)

## Training Gaussian Splat Diffusion Model

To Run the GS Splat Diffusion ```python/python3 -m src.training_splat_unet_encoder --run_flag --start_long_run_epochs --end_long_run_epochs --save_interval --log_interval ```

``` --run_flag```: can be **'start'** or **'resume'**

```--start_long_run_epochs```: The starting epoch of the training. To resume training make sure that starting is equal to the last training phase ending epoch

```--end_long_run_epochs```: The target epoch till what you want to run.

```--save_interval```: The epoch interval post which  a checkpoint is saved

```--log_interval```: The epoch interval post which a sample image is saved and loss is stored.

example command to start training : ```python -m src.training_splat_unet_diffu --run_flag 'start' --start_long_run_epochs 0 --end_long_run_epochs 200 --save_interval 100 --log_interval 50```

example command to resume training : ```python -m src.training_splat_unet_diffu --run_flag 'resume' --start_long_run_epochs 200 --end_long_run_epochs 400 --save_interval 100 --log_interval 50```


##### Data: STL10 is auto download during training phase again if data is not found or the directory cannot be resolved. If data is already there training will start directly


### Default Params

All the default params are find within ```src/config.py``` including ```DEVICE```.
The current code supports ```CUDA``` and ```MPS```. For MPS make sure that the correct version of pytorch is installed and supported.


### NOTE: This is still a WIP and changes are being added.
