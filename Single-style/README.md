## Neural Style Word Transfer 

Our single-style model is a modified version of Engstrom's (https://github.com/lengstrom/fast-style-transfer) implementation of Johnson's [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](http://cs.stanford.edu/people/jcjohns/eccv16/), and Ulyanov's [Instance Normalization](https://arxiv.org/abs/1607.08022). 



## Single-style Transfer Network
Below are various transforms of a photo of fruit from our single-style model.

     
<div align='center'>
<img src = 'Single-style/Examples/Content/fruit.jpg' height="200px">
</div>
     
<div align = 'center'>
<img src = 'Single-style/Examples/Results/fruit-brightMonet.jpg' height = '200px'>
<img src = 'Single-style/Examples/Results/cornell-sketch.jpg' height = '200px'>

<br>
<img src = 'Single-style/Examples/Results/fruit-vanGogh.jpg' height = '200px'>
<img src = 'Single-style/Examples/Results/fruit-dullMonet.jpg' height = '200px'>

</div>

## Stylizing an Image
To stylize an image, download one of our pre-trained models and use `evaluate.py` as detailed below.
* Van Gogh - https://drive.google.com/file/d/1b97VrRVC_G6P8migaaB_vveyqW1bV6ne/view?usp=sharing
* Sketch - https://drive.google.com/file/d/1prdzaQJDqYS62DT3gbuw0jVOYfvy-msZ/view?usp=sharing

## Training a Single-Style Transfer Network
Before training can begin, style targets must first be extracted from a set of style images using `extractstyle.py` and a pre-trained VGG-19 model will need to be downloaded.

Once style targets have been generated, use `train.py` to train a new single-style transfer network. Run `python train.py` to view all the possible parameters. 
Example usage:

    python train.py --style-dir path/to/style/dir \
      --checkpoint-dir checkpoint/path \
      --test-img path/to/test/img.jpg \
      --test-dir path/to/test/dir \
      --content-weight 1.5e1 \
      --checkpoint-iterations 200 \
      --batch-size 4


## Evaluating a Single-Style Transfer Network
Use `evaluate.py` to evaluate a single-style transfer network. Run `python evaluate.py` to view all the possible parameters. 
Example usage:

    python evaluate.py --checkpoint path/to/style/model.ckpt \
      --in-path dir/of/test/imgs/ \
      --out-path dir/for/results/




### License
Original model is "Copyright (c) 2016 Logan Engstrom. Contact me for commercial use (or rather any use that is not academic research) (email: engstrom at my university's domain dot edu). Free for research use, as long as proper attribution is given and this copyright notice is retained."



