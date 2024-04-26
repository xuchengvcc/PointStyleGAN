# [Point-StyleGAN: Multi-scale Point Cloud Synthesis with Style Modulation] 

[Yang Zhou](https://zhouyangvcc.github.io/), [Cheng Xu](https://github.com/xuchengvcc), [Zhiqiang Lin](), [Xinwei He](https://eriche2016.github.io/), [Hui Huang](https://vcc.tech/~huihuang)
<!-- Shenzhen University -->

[[paper]](https://linkinghub.elsevier.com/retrieve/pii/S0167839624000438)

<p>
    <img src="pictures/teaser.png" style="width: 80%;" alt="teaser"/>
</p>


### Abstract
A point cloud is a set of discrete surface samples. As the simplest 3D representation, it is widely used in
3D reconstruction and perception. Yet developing a generative model for point clouds remains challenging
due to the sparsity and irregularity of points. Drawn by StyleGAN, the forefront image generation model,
this paper presents Point-StyleGAN, a generator adapted from StyleGAN2 architecture for point cloud
synthesis. Specifically, we replace all the 2D convolutions with 1D ones and introduce a series of multiresolution 
discriminators to overcome the under-constrained issue caused by the sparsity of points. We further add a metric 
learning-based loss to improve generation diversity. Besides the generation task, we show several applications 
based on GAN inversion, among which an inversion encoder Point-pSp is designed and applied to point cloud 
reconstruction, completion, and interpolation. To our best knowledge, Point-pSp is the first inversion encoder 
for point cloud embedding in the latent space of GANs. The comparisons to prior work and the applications of 
GAN inversion demonstrate the advantages of our method. We believe the potential brought by the Point-StyleGAN 
architecture would further inspire massive follow-up works.

### Overview

<p>
    <img src="pictures/overview.png" style="width: 80%;" alt="overview"/>
</p>


### Dataset

[ShapeNetCoreV2](https://drive.google.com/drive/folders/1mjP_dhl1DQL42a9bS3NH20oo_XxlNBzQ?usp=sharing)


## Citation

```
@article{ZHOU2024102309,
title = {Point-StyleGAN: Multi-scale Point Cloud Synthesis with Style Modulation},
journal = {Computer Aided Geometric Design},
pages = {102309},
year = {2024},
issn = {0167-8396},
doi = {https://doi.org/10.1016/j.cagd.2024.102309},
url = {https://www.sciencedirect.com/science/article/pii/S0167839624000438},
author = {Yang Zhou and Cheng Xu and Zhiqiang Lin and Xinwei He and Hui Huang},
keywords = {Point cloud synthesis, StyleGAN, Point cloud inversion},
abstract = {A point cloud is a set of discrete surface samples. As the simplest 3D representation, it is widely used in 3D reconstruction and perception. Yet developing a generative model for point clouds remains challenging due to the sparsity and irregularity of points. Drawn by StyleGAN, the forefront image generation model, this paper presents Point-StyleGAN, a generator adapted from StyleGAN2 architecture for point cloud synthesis. Specifically, we replace all the 2D convolutions with 1D ones and introduce a series of multi-resolution discriminators to overcome the under-constrained issue caused by the sparsity of points. We further add a metric learning-based loss to improve generation diversity. Besides the generation task, we show several applications based on GAN inversion, among which an inversion encoder Point-pSp is designed and applied to point cloud reconstruction, completion, and interpolation. To our best knowledge, Point-pSp is the first inversion encoder for point cloud embedding in the latent space of GANs. The comparisons to prior work and the applications of GAN inversion demonstrate the advantages of our method. We believe the potential brought by the Point-StyleGAN architecture would further inspire massive follow-up works.}
}
```
<!-- ## Acknowledgments -->