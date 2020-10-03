

# VoicePuppet #
 - This repository provided a common pipeline to generate speaking actor by voice input automatically

## The archecture of the network ##
 - Composed of 2 parts, one for predict 3D face coeffcients of each frame align to a certain stride window size of waveform, called BFMNet(basel face model network). The another for redraw the real face foreground using the rasterized face which produced by the rendered 3D face coeffcients of previous step, called PixReferNet.

<table>
  <tr>
    <th>
      <img src="https://github.com/taylorlu/voicepuppet/blob/master/res/1.png" >
    </th>
  </tr>
  <tr><th>
    BFMNet component
  </th></tr>
    <th>
      <img src="https://github.com/taylorlu/voicepuppet/blob/master/res/2.png" >
    </th>
  </tr>
  <tr><th>
    PixReferNet component
  </th></tr>
</table>


## Run the predict pipeline ##
------------------------

 1. Download the pretrained model and required models.
    Extract the pretrained model to ckpt and ckpt, extract the allmodels to current root dir
 2. `cd utils/cython` && `python3 setup.py install`
 3. Install ffmpeg tool if you want to merge the png sequence and audio file to video container like mp4.
 4. `python3 voicepuppet/pixrefer/infer_bfmvid.py --config_path config/params.yml sample/22.jpg sample/test.aac`

## Run the training pipeline ##
------------------------

#### Requirements ####
------------

 - tensorflow>=1.14.0
 - pytorch>=1.4.0, only for data preparation (face foreground segmentation and matting)
 - mxnet>=1.5.1, only for data preparation (face alignment)
 tips: you can use other models to do the same label marking instead, such as dlib

#### Data preparation ####
------------

 1. Check your `config/params.yml` to make sure the dataset folder in specified structure (same as the [grid dataset](http://spandh.dcs.shef.ac.uk/gridcorpus/), you can extend the dataset by using the same folder structure which contains common video files)
```
|- srcdir/
|    |- s10/
|        |- video/
|            |- mpg_6000/
|                |- bbab8n.mpg
|                |- bbab9s.mpg
|                |- bbac1a.mpg
|                |- ...
|    |- s8/
|        |- video/
|            |- mpg_6000/
|                |- bbae5n.mpg
|                |- bbae6s.mpg
|                |- bbae7p.mpg
|                |- ...
```
 2. Extract audio stream from mpg video file, `todir` was a output folder which you want to store the labels.</br>
    `python3 datasets/make_data_from_GRID.py --gpu 0 --step 2  srcdir todir`

 3. Face detection and alignment</br>
    `python3 datasets/make_data_from_GRID.py --gpu 0 --step 3 srcdir todir ./allmodels`

 4. 3D face reconstruction</br>
    `python3 datasets/make_data_from_GRID.py --gpu 0 --step 4 todir ./allmodels`

 5. It will take several hours to finish the above steps, subsequently, you'll find there's `*.jpg, landmark.txt, audio.wav, bfmcoeff.txt` in each output subfolder. The above labels(`audio.wav`, `bfmcoeff.txt`) are used for BFMNet training, the others are only temp files.
```
|- todir/
|    |- s10/
|        |- bbab8n/
|            |- landmark.txt
|            |- audio.wav
|            |- bfmcoeff.txt
|            |- 0.jpg
|            |- 1.jpg
|            |- ...
|        |- bbab9s/
|            |- ...
|    |- s8/
|        |- bbae5n/
|            |- landmark.txt
|            |- audio.wav
|            |- bfmcoeff.txt
|            |- 0.jpg
|            |- 1.jpg
|            |- ...
|        |- bbae6s/
|            |- ...
```
 6. Face(human foreground) segmentation and matting for PixelReferNet training. Before invoke the python shell, you should make sure the width and height of the video was in the same size(1:1). In general, 3-5 minutes video was enough for training the PixelReferNet network, the trained model will only take effect on this specified person too.</br>
    `python3 datasets/make_data_from_GRID.py --gpu 0 --step 6 src_dir to_dvp_dir ./allmodels`</br>
the `src_dir` has the same folder structure as tip1, when finish the above step, you will find `*.jpg` in subfolders, like this
<div align="center">
<img src="https://github.com/taylorlu/voicepuppet/blob/master/sample/22.jpg">
</div>

#### Train BFMNet ####
------------

1. Prepare train and eval txt, check the `root_path` parameter in `config/params.yml` is the output folder of tips1</br>
    `python3 datasets/makelist_bfm.py --config_path config/params.yml`
2. train the model</br>
    `python3 voicepuppet/bfmnet/train_bfmnet.py --config_path config/params.yml`
3. Watch the evalalute images every 1000 step in `log/eval_bfmnet`, the upper was the target sequence, and the under was the evaluated sequence.
<div align="center">
<img src="https://github.com/taylorlu/voicepuppet/blob/master/res/3.jpg">
</div>

#### Train PixReferNet ####
------------

1. Prepare train and eval txt, check the `root_path` parameter in `config/params.yml` is the output folder of tips6</br>
    `python3 datasets/makelist_pixrefer.py --config_path config/params.yml`
2. train the model</br>
    `python3 voicepuppet/pixrefer/train_pixrefer.py --config_path config/params.yml`
3. Use tensorboard to watch the training process</br>
    `tensorboard --logdir=log/summary_pixrefer`

## Acknowledgement ##
1. The face alignment model was refer to [Deepinx's work](https://github.com/deepinx/deep-face-alignment), it's more stable than Dlib.
2. 3D face reconstruction model was refer to [microsoft's work](https://github.com/microsoft/Deep3DFaceReconstruction)
3. Image segmentation model was refer to [gasparian's work](https://github.com/gasparian/PicsArtHack-binary-segmentation)
4. Image matting model was refer to [foamliu's work](https://github.com/foamliu/Deep-Image-Matting)
