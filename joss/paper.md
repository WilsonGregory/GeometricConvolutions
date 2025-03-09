---
title: 'GeometricConvolutions: E(d)-Equivariant CNN for Tensor Images'
tags:
  - Python
  - Jax
  - machine learning
  - E(d)-equivariance
  - tensor images
  - Equinox
authors:
  - name: Wilson G. Gregory
    orcid: 0000-0002-5511-0683
    corresponding: true
    affiliation: 1 
  - name: Kaze W. K. Wong
    affiliation: 1
affiliations:
 - name: Department of Applied Mathematics and Statistics, Johns Hopkins University, Baltimore, MD, USA
   index: 1
date: 18 February 2025
bibliography: paper.bib
---

# Summary

Many data sets encountered in machine learning exhibit symmetries that can be exploited to improve performance, a technique known as equivariant machine learning. 
The classical example is image translation equivariance that is respected by convolutional neural networks [@lecun1989backpropagation]. 
For data sets in the physical sciences and other areas, we would also like equivariance to rotations and reflections. 
This Python package implements a convolutional neural network that is equivariant to translations, rotations of 90 degrees, and reflections. 
We implement this by _geometric convolutions_ [@gregory2024ginet] which use tensor products and tensor contractions. 
This additionally enables us to perform functions on geometric images, or images where each pixel is a higher order tensor. 
These images appear as discretizations of fields in physics, such as velocity fields, vorticity fields, magnetic fields, polarization fields, and so on. 

This package includes basic functionality to create, manipulate, and plot geometric images using JAX [@jax2018github]. 
It also provides equivariant neural network layers such as convolutions, activation functions, group norms, and others using the Equinox framework [@kidger2021equinox]. 
These layers ingest a special data structure, the `MultiImage`, that allows combining geometric images or any tensor order or parity into a single model. 
Finally, the package provides full-fledged versions of popular models such as the UNet or ResNet to allow researchers to quickly train and test on their own data sets with standard tools in the JAX ecosystem.

The key features and targeted use cases are summarized in the following list:

# Key features

Comment: Make the bullet points into actual sentences.

1. Geometric filters implemented in `jax` and `equinox`
2. A special data structure `MultiImage`
3. Interactive inspection of geometric filters and `MultiImage`
4. Standalone networks

# Statement of need


The geometric convolutions introduced in [@gregory2024ginet] are defined on geometric images–an image where every pixel is a tensor.
If $A$ is a geometric image of tensor order $k$ and $C$ is a geometric image of tensor order $k'$, then value of $A$ convolved with $C$ at pixel $\bar\imath$ is given by:

$$
(A \ast C)(\bar\imath) = \sum_{\bar a} A(\bar\imath - \bar a) \otimes C(\bar a) ~,
$$

where the sum is over all pixels $\bar a$ of $C$, and $\bar\imath - \bar a$ is the translation of $\bar\imath$ by $\bar a$. 
The result is a geometric image of tensor order $k+k'$. 
To produce geometric images of smaller tensor order, the tensor contraction can be applied to each pixel. 
Convolution and contraction are combined into a single operation to form linear layers. 
By restricting the convolution filters $C$ to rotation and reflection invariant filters, we can create linear layers which are rotation-, reflection-, and translation-equivariant.

The space of equivariant machine learning software is largely still in its infancy, and this is currently the only package implementing geometric convolutions.
However, there are alternative methods for solving $O(d)$-equivariant image problems.
One such package is [escnn](https://github.com/QUVA-Lab/escnn) which uses Steerable CNNs [@cohen2016steerablecnns;@wweiler2021steerable].
Steerable CNNs use irreducible representations to derive a basis for $O(d)$-equivariant layers, but it is not straightforward to apply on higher order tensor images.
The escnn package is built with pytorch, although there is a JAX port [escnn_jax](https://github.com/emilemathieu/escnn_jax).

Another alternative method is are those based on Clifford Algebras, in particular [@brandstetter2023clifford].
This method has been implemented in the [Clifford Layers](https://github.com/microsoft/cliffordlayers) package.
Like escnn, this method is also built with pytorch rather than JAX. Additionally, Clifford based methods can process vectors and pseudovectors, but cannot handle higher order tensors.


\[package name\] targets two main use cases:

## As a drop-in replacement for CNN

Comment: Expand on how you expext the users to use it as a drop in replacement. Also I feel like some of the other papers can be mentioned here instead.

For practitioners, we provide both equivariant layers for building models, and full fledged model implementations such as the UNet, ResNet, and Dilated ResNet.

## For designing and understanding geometric images

Comment: Expand on this.


For equivariance researchers, we provide all the common operations on geometric images such as addition, scaling, convolution, contraction, transposition, norms plus visualization methods.

# References