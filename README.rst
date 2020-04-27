Image Encoders
==============

When generating images, pixel colors aren't the most useful way to store an image!  You need ways to capture the patterns of an image so they can be modified to generate new images. 🖼️

In deep learning, this is done with an ``Encoder``: a neural network that takes an input image and outputs statistics about the higher-level features found.  It's useful for a variety of downstream applications for high-quality image synthesis. 📊

This repository contains a collection of deep encoders implemented in *PyTorch* that computes features of the image at different scales.  You can integrate these into your own *Python* applications! ⚙️
