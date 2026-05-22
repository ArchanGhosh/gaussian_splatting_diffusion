# Using 2D Gaussian Splatting in Diffusion Models

>>> Abstract : Diffusion models have achieved remarkable success in image synthesis but remain computationally expensive due to iterative denoising in high-dimensional pixel space. Latent Diffusion Models (LDMs) alleviate this cost through compressed neural representations, yet often sacrifice high frequency spatial detail during latent compression. In this
work, we introduce a diffusion framework based on 2D Gaussian Splatting, replacing dense pixel-space representations with structured Gaussian primitives. Our approach employs an encoder-diffusion-renderer pipeline, where input images are first mapped into a compact Gaussian latent space parameterized by spatial location, scale, opacity, and color attributes. Diffusion is then performed directly over these Gaussian latents, enabling efficient generation while preserving fine-grained spatial structure. The denoised Gaussian representations are subsequently rendered back into the image domain using differentiable splatting. By operating on compact geometric primitives rather than dense latent tensors, our framework significantly reduces computational overhead while maintaining visual fidelity. 


# Paper

The Work can be found at [WIP]()

# 1. Requirements

All the components have been written using pytorch and torchvision.

```python version > 3.11 ```

To install torch please follow the official guide on [pytorch.org](https://pytorch.org/get-started/locally/)

Example: ```pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126 ```

# Running The code:

#### NOTE : All the configuration parameters are set inside of ```SRC/CONFIG.PY```. Please Ensure that before changing any parameter (especially training params) the other params are recalculated to avoid any memory, overlapping issues.

## Training the Gaussian Splat Encoder

#### Data : Currently the model is trained on a subset of STL10 images. These can be easily changed by changing or adding a similar data loader file like ```src/stl10_loader.py ```. And replace the corresponding function inside the training file.

To Run the GS Encoder pipeline, from the base folder you can simple do ```python/python3 training_splat_encoder.py ```.
Training data is auto-downloaded.
All the images generated during training should be available in the ```splat_encoder_training_imgs``` folder created automatically during the training process.

The weights of the encoder are saved in the ``` CHECKPOINTS ```, along with that of the renderer.
