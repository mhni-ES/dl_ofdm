# Acknowledgement 
This repository is a fork from Dr. Zhongyuan Zhao paper "Deep-Waveform: A Learned OFDM Receiver Based on Deep Complex-valued Convolutional Networks, in IEEE Journal on Selected Areas in Communications, vol. 39, no. 8, pp. 2407-2420, Aug. 2021, doi: [10.1109/JSAC.2021.3087241](https://doi.org/10.1109/JSAC.2021.3087241)."
This fork of the original repository incorporates modifications to address the effects of Power Amplifier (PA) non-linearity in the OFDM receiver design. The changes aim to enhance the receiver's performance by mitigating the distortion introduced by PA non-linearities, which are prevalent in practical communication systems.


## About this code
Modified source code for Deep Learning-Based OFDM Receiver.

+ Modulation: BPSK, QPSK, 8-QAM, 16-QAM, of Gray mapping.
+ SNR: -10:1:29 dB

### Key Changes
- Implementation of a new module within the receiver's architecture to create PA non-linear effects.
- Adaptation of the learning algorithm to include training data affected by PA non-linearities, ensuring robustness and improved accuracy.
- Evaluation of the modified receiver's performance in various scenarios, demonstrating its enhanced capability to handle PA-induced distortions.


### Software platform
+ Matlab R2017b, R2018a (replace `rayleighchan` in the code for newer release)
+ Python3 compatiable
+ TensorFlow 1.x: `tensorflow-gpu==1.15`, docker tensorflow image [1.15.5-gpu-jupyter](https://hub.docker.com/layers/tensorflow/tensorflow/1.15.5-gpu-jupyter/images/sha256-5f2338b5816cd73ea82233e2dd1ee0d8e2ebf539e1e8b5741641c1e082897521?context=explore
) is highly recommended if you just want a quick tryout. If your GPU is not supported by that docker image, checkout latest docker images with tag `[year.month]-tf1-py3` on [Nvidia NGC](https://ngc.nvidia.com/catalog/containers/nvidia:tensorflow/tags).
+ **Note:** newer versions of Matlab and Tensorflow are possible, but require quite some work on the code (read [here](#for-newer-versions-of-matlab-and-tensorflow)). 

### Contents of directories
```bash
.
├── dev # latest working source code
├── test_v1 # archived for old version https://arxiv.org/abs/1810.07181v3
├── README.md 
└── LICENSE
```

### Usage
1. Run `script_rayleigh` in Matlab for benchmarks
2. Run `python3 run_local_ofdm.py --awgn=True` in terminal for training and testing results. 

### For newer versions of Matlab and Tensorflow
**Matlab**: `rayleighchan` was removed and replaced by ['comm.RayleighChannel'](https://www.mathworks.com/help/comm/ref/comm.rayleighchannel-system-object.html) from later versions of matlab. You may uncomment the lines 202-210, lines 293-300 in [/dev/m/OFDM_Benchmark_dev.m](/dev/m/OFDM_Benchmark_dev.m) to use the newer function. However, you need either replace `parfor` to `for` in that code to directly use the newer function, or change the code to initialize multiple identical objects of `comm.RayleighChannel` in lines 202-210 and use different objects (lines 293-300) in each parallel loop to enable the `parfor`. 

**Tensorflow:** I use `tf.contrib.graph_editor` to enable the transfer learning scheme described in the paper. However, in Tensorflow 2, the `tf.contrib` is removed and `graph_editor` no longer exists. If you want to use TF2 rather than TF1, you will need to re-write the transfer learning (lines 264-365 in [ofdmreceiver_np_mp.py](/dev/py//ofdmreceiver_np_mp.py)) with whatever equivalent new methods in TF 2.

### Update
1. A few missing CSV files are uploaded in commit [26ea69](https://github.com/zhongyuanzhao/dl_ofdm/commit/26ea69b48469b194c3f4bac2de1a81be8137f8cf). These CSV files are the FIR filter coefficients for the Rayleigh channels used in the evaluation. They are necessary for experiments in fading channel. These CSV files are computed according to equation (2) in the paper, based on the channel tapes configured [here](https://github.com/zhongyuanzhao/dl_ofdm/blob/26ea69b48469b194c3f4bac2de1a81be8137f8cf/dev/m/OFDM_Benchmark_dev.m#L176-L197). They can also be found as `h.AlphaMatrix` in OFDM_Benchmark_dev.m after stepping over line 283 (change `parfor` to `for` in line 282 in order to step into that block of code in Matlab).
