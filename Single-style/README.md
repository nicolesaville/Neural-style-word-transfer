## Neural Style Word Transfer 

Our single-style model is a modified version of Engstrom's (https://github.com/lengstrom/fast-style-transfer) implementation of Johnson's [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](http://cs.stanford.edu/people/jcjohns/eccv16/), and Ulyanov's [Instance Normalization](https://arxiv.org/abs/1607.08022). 



## Single-style Transfer Network
Below are various transforms of a photo of Cornell from our single-style model.

<div align='center'>
<img src = 'Examples/Content/cornell.jpg' height="200px">
</div>
     
<div align = 'center'>
<img src = 'Examples/Results/Single-style/cornell-brightMonet.jpg' height = '200px'>
<img src = 'Examples/Results/Single-style/cornell-dullMonet.jpg' height = '200px'>

<br>
<img src = 'Examples/Results/Single-style/cornell-sketch.jpg' height = '200px'>
<img src = 'Examples/Results/Single-style/cornell.jpg' height = '200px'>

</div>

### Stylizing an Image
To stylize an image, download one of our pre-trained models and use `evaluate.py` as detailed below.
* link to pre-trained model download


### Training a Single-Style Transfer Network
Use `train.py` to train a new single-style transfer network. Run `python train.py` to view all the possible parameters. 
Example usage:

    python train.py --style-dir path/to/style/dir \
      --checkpoint-dir checkpoint/path \
      --test-img path/to/test/img.jpg \
      --test-dir path/to/test/dir \
      --content-weight 1.5e1 \
      --checkpoint-iterations 200 \
      --batch-size 4


### Evaluating a Single-Style Transfer Network
Use `evaluate.py` to evaluate a single-style transfer network. Run `python evaluate.py` to view all the possible parameters. 
Example usage:

    python evaluate.py --checkpoint path/to/style/model.ckpt \
      --in-path dir/of/test/imgs/ \
      --out-path dir/for/results/




### License
Original model is "Copyright (c) 2016 Logan Engstrom. Contact me for commercial use (or rather any use that is not academic research) (email: engstrom at my university's domain dot edu). Free for research use, as long as proper attribution is given and this copyright notice is retained."



