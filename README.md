## Holistically-Nested Edge Detection

Created by Saining Xie at UC San Diego

### Introduction:

<img src="http://pages.ucsd.edu/~ztu/hed.jpg" width="400">

We develop a new edge detection algorithm, holistically-nested edge detection (HED), which performs image-to-image prediction by means of a deep learning model that leverages fully convolutional neural networks and deeply-supervised nets.  HED automatically learns rich hierarchical representations (guided by deep supervision on side responses) that are important in order to resolve the challenging ambiguity in edge and object boundary detection. We significantly advance the state-of-the-art on the BSD500 dataset (ODS F-score of .790) and the NYU Depth dataset (ODS F-score of .746), and do so with an improved speed (0.4s per image). Detailed description of the system can be found in our [paper](http://arxiv.org/abs/1504.06375).

### Citations

If you are using the code/model/data provided here in a publication, please cite our paper:

    @InProceedings{xie15hed,
      author = {"Xie, Saining and Tu, Zhuowen"},
      Title = {Holistically-Nested Edge Detection},
      Booktitle = "Proceedings of IEEE International Conference on Computer Vision",
      Year  = {2015},
    }

### Changelog

If you have downloaded the previous version (testing code) of HED, please note that we updated the code base to the new version of Caffe. We uploaded a new pretrained model with better performance. We adopted the python interface written for the FCN paper instead of our own implementation for training and testing. The evaluation protocol doesn't change.

### Pretrained model

We provide the pretrained model and training/testing code for the edge detection framework Holistically-Nested Edge Detection (HED). Please see the Arxiv or ICCV paper for technical details. The pretrained model (fusion-output) gives ODS=.790 and OIS=.808 result on BSDS benchmark dataset.
  0. Download the pretrained model (56MB) from (http://vcl.ucsd.edu/hed/hed_pretrained_bsds.caffemodel) and place it in examples/hed/ folder.

### Installing 
 0. Install prerequisites for Caffe(http://caffe.berkeleyvision.org/installation.html#prequequisites)
 0. Modified-caffe for HED: https://github.com/s9xie/hed.git

### Training HED
To reproduce our results on BSDS500 dataset:
 0. data: Download the augmented BSDS data (1.2GB) from (http://vcl.ucsd.edu/hed/HED-BSDS.tar) and extract it in data/ folder
 0. initial model: Download fully convolutional VGG model (248MB) from (http://vcl.ucsd.edu/hed/5stage-vgg.caffemodel) and put it in examples/hed folder
 0. run the python script **python solve.py** in examples/hed

### Testing HED
Please refer to the IPython Notebook in examples/hed/ to test a trained model. The fusion-output, and individual side-output from 5 scales will be produced after one forward pass.
 
Note that if you want to evaluate the results on BSDS benchmarking dataset, you should do the standard non-maximum suppression (NMS) and edge thinning. We used Piotr's Structured Forest matlab toolbox available here **https://github.com/pdollar/edges**. Some helper functions are also provided in the [eval/ folder](https://github.com/s9xie/hed_release-deprecated/tree/master/examples/eval). 

### Batch Processing

[Jun-Yan Zhu](https://people.eecs.berkeley.edu/~junyanz/) from UC Berkeley recently applied HED for their [Image-to-Image Translation](https://phillipi.github.io/pix2pix/) work. A nice script for batch-processing HED edge detection can be found [here](https://github.com/phillipi/pix2pix/tree/master/scripts/edges). Thanks Jun-Yan!

### Precomputed Results
If you want to compare your method with HED and need the precomputed results, you can download them from (http://vcl.ucsd.edu/hed/eval_results.tar).


### Acknowledgment: 
This code is based on Caffe. Thanks to the contributors of Caffe. Thanks @shelhamer and @longjon for providing fundamental implementations that enable fully convolutional training/testing in Caffe.

    @misc{Jia13caffe,
      Author = {Yangqing Jia},
      Title = { {Caffe}: An Open Source Convolutional Architecture for Fast Feature Embedding},
      Year  = {2013},
      Howpublished = {\url{http://caffe.berkeleyvision.org/}}
    }

If you encounter any issue when using our code or model, please let me know.
