# this repo is now maintenance only; please develop a fork || use the mrq repo if you have large features to submit

**NOTICE**: this repo is not endorsed by @neonbjb

15 May 2023 Full Youtube Tutorial For Voice Training With DLAS: [Link](https://youtu.be/OiMRlqcgDL0)

[![image](https://github.com/FurkanGozukara/DL-Art-School/assets/19240467/964f07d1-7be3-43c7-896b-ed1cbb7463da)](https://youtu.be/OiMRlqcgDL0)

#### in progress: [Diffusion model training](#training-the-diffusion-model-wip). Track progress [here](https://github.com/152334H/DL-Art-School/issues/3#issuecomment-1436541887)
#### in progress: 8bit optimizers (works on Linux, see [here](https://github.com/152334H/DL-Art-School/issues/8#issuecomment-1441850067))
## **NEW**: (windows) [Training UI](#windows-training-ui-with-conda)
## [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1-KmMuexR9Mv40QHNt_If2YY8nx3rFi4a) [Colab training notes](./COLAB_USAGE.md)

# Tortoise fine-tuning with DLAS

like the [tortoise-tts-fast](https://github.com/152334H/tortoise-tts-fast) project, **the changes in this repo are also licensed as AGPL**, but if I'm being realistic I haven't actually made any new code, the only thing I can really copyright is the config files and this readme.md

## INSTALLATION
(this will be updated often)

### CLI Install

```sh
git clone https://github.com/152334H/DL-Art-School
cd DL-Art-School
wget https://huggingface.co/Gatozu35/tortoise-tts/resolve/main/dvae.pth -O experiments/dvae.pth # https://huggingface.co/jbetker/tortoise-tts-v2/resolve/3704aea61678e7e468a06d8eea121dba368a798e/.models/dvae.pth
sha256sum ../DL-Art-School/experiments/dvae.pth | grep a990825371506c16bcf0e8167bf24ccf82f65bb6a1dbcbfcf058d76f9b197e35 # extra check on vqvae checkpoint integrity, feel free to skip this
cp ~/.cache/tortoise/models/autoregressive.pth experiments # copy the gpt model
pip install -r codes/requirements.laxed.txt # ONLY TESTED ON python=3.9; use your existing tortoise env if possible
pip uninstall tensorboard # this is only needed if you want to view tensorboard logs. see https://github.com/pytorch/pytorch/issues/22676
```

### Windows Training UI with Conda

[![Tutorial](https://img.youtube.com/vi/lnIq4SFFXWs/0.jpg)](https://www.youtube.com/watch?v=lnIq4SFFXWs)


This script by @devilismyfriend will install a conda environment named `DLAS` that uses a newer version of cuDNN for faster training:

```bat
git clone https://github.com/152334H/DL-Art-School
cd DL-Art-School
".\Setup DLAS.bat"
```

This requires that you have [miniconda](https://docs.conda.io/en/latest/miniconda.html)/[anaconda](https://www.anaconda.com/) (with python 3.10) and git installed.

After it finishes setup, you can run it with `".\Start DLAS.cmd"`:


![](https://i.imgur.com/knj39lE.png)


## CLI USAGE
1. prepare a dataset (**LJSpeech format** is what's configured; if you can read the code you can use other formats like voxpopuli)
2. **edit `experiments/EXAMPLE_gpt.yml`**. Read & possibly edit **every line** **with `CHANGEME`** in it. Especially,
   * change the dataset config (obviously)
   * reduce batch size if you have less-than 16GB vram
   * possibly change the learning rate and other hyperparams
3. run `cd codes && python3 train.py -opt ../experiments/EXAMPLE_gpt.yml`
   * DO NOT CANCEL THIS until you see `INFO: Saving models and training states.`. All training progress before that line is LOST. (feel free to cancel it if you used bad data or something)
4. load up the [tortoise-tts-fast](https://github.com/152334H/tortoise-tts-fast) fork, and use the new `--ar-checkpoint` option with `/path/to/DL-Art-School/experiments/<INSERT EXPERIMENT NAME HERE>/models/<MOST RECENT STEPS>_gpt.pth`.

### Training the diffusion model (WIP)
1. Make sure you have an LJSpeech dataset & a fine-tuned GPT model from the previous instructions.
2. **edit `experiments/EXAMPLE_diff.yml`**. Everything above applies, but there's also some new things:
  * More vram is consumed compared to the GPT training; you should reduce batch size accordingly
	* **`gpt_path`** needs to point to a GPT model.
	* `after:` parameters adjust warmup steps
3. run `cd cdoes && python3 train.py -opt ../experiments/EXAMPLE_diff.yml`. Same warning applies
4. The [tortoise-tts-fast](https://github.com/152334H/tortoise-tts-fast) fork now has a `--diff-checkpoint` flag as well.

### continuing a training session
_This section will be removed once I write a script to automate it_

If you have to continue an incomplete training session for any reason, then:

1. Find the most recently saved training state file. If the last saved step was `$step`, and your training session was named `$trainingName`, then the file will be at `../experiments/$trainingName/training_state/$step.state`
2. Edit the `.yml` config file's `path:`
   ```yml
   path:
     #pretrain_model_gpt: '../experiments/autoregressive.pth' # COMMENT THIS LINE OUT
     strict_load: true
     #resume_state: ../experiments/$trainingName/training_state/$step.state
   ```

### Hyperparameters vs dataset size
_The colab training nodebook will automatically find good hyperparams based on the dataset size. This info is merely included for CLI users || public information_.

Note that `epoch_steps == dataset_size//batch_size`; partial batches are discarded by the trainer.

The hyperparameters configured in the example yml are sane defaults for a _reasonably large dataset_ of a few thousand samples. If your dataset is much smaller (50-500), you should make the following changes:
* batch size reduction. Specifically, ensure the batch size used is reasonably close to being a clean divisor of the provided dataset; the DLAS trainer (currently) drops partially filled batches.
* `gen_lr_steps`. The smaller your dataset, the faster you should be decaying the learning rate. You should have no more than 10 epoches before the first decay, i.e. the first number in `gen_lr_steps` should be smaller than `epoch_steps * 10`. (personal tests indicate that loss starts going up if there's no decay by epoch 20)
* `print_freq`, `val_freq`, `save_checkpoint_freq`: These should all be adjusted to dataset size as well. Recommendation: `val_freq == save_checkpoint_freq == print_freq*3`; `print_freq == min(epoch_steps,100)`

## RESULTS
For a very basic and simple task, I trained the ar model for 500 steps, with batch size 128, on a dataset of [Kim Kitsuragi](https://discoelysium.fandom.com/wiki/Kim_Kitsuragi) that contained ~4.5k wav files. This means I trained for about 11 epoches, which I'm not sure is a good thing or not.

| CONDITIONING | CHECKPOINT | SAMPLEs | verdict
| - | - | - | - |
| KK | original | [here](./voice_samples/kk_orig) | - |
| KK | 500_gpt.pth | [here](./voice_samples/kk_500) | Much closer to the real character! | 
| emma | original | [here](https://github.com/152334H/tortoise-tts-fast/tree/main/optimized_examples/A/very_fast-ar16) | - | 
| emma | 500_gpt.pth | [here](./voice_samples/kk_500_emma) | A mix of "well-transferred accent" (surprising!) && "catastropic memorisation" (expected) | 

![image](https://user-images.githubusercontent.com/54623771/219252253-7ca44efe-5d49-4ae5-9d4a-5add62f5cd77.png)

### Other experiments so far

- Fine-tuning with [non-English datasets (Marathi)](https://github.com/152334H/DL-Art-School/discussions/12), [YouTube examples](https://youtu.be/kzBOrMw7oBk)
- other experiments [by me](https://github.com/152334H/DL-Art-School/issues/1)

## [Helping out](https://github.com/152334H/DL-Art-School/discussions/4)
This project is in its infancy, and is in desperate need of contributors. If you have any suggestions / ideas, please visit the [discussions](https://github.com/152334H/DL-Art-School/discussions/4) page. If you have programming skills, try making a pull request! 

### Extra community stuff

- (windows only) [Automatic LJSpeech dataset creation && audio transcription with Whisper](https://github.com/devilismyfriend/ozen-toolkit)
- (linux only) Saving and transcribing audio played from speakers, [link](https://github.com/tekakutli/tortoise-scripts)

## todo
- [X] run at least 1 epoch of autoregressive training with clear loss decrease
- [X] upload training configs
- [X] check that results are actually good (!!)
- [X] create a colab training notebook
- [ ] train other submodels (diffusion, clvp)
- [ ] offload all of the work to other contributors


---

# Deep Learning Art School

Send your Pytorch model to art class!

This repository is both a framework and a set of tools for training deep neural networks that create images. It started 
as a branch of the [open-mmlab](https://github.com/open-mmlab) project developed by [Multimedia Laboratory, CUHK](http://mmlab.ie.cuhk.edu.hk) 
but has been almost completely re-written at every level.

## Why do we need another training framework

These are a dime a dozen, no doubt. DL Art School (*DLAS*) differentiates itself by being configuration driven. You write 
the model code (specifically, a torch.nn.Module) and (possibly) some losses, then you cobble together a config file written 
in yaml that tells DLAS how to train it. Swapping model architectures and tuning hyper-parameters is simple and often 
requires no changes to actual code. You also don't need to remember complex command line incantations. This effectively 
enables you to run multiple concurrent experiments that use the same codebase, as well as retain backwards compatibility 
for past experiments.

Training effective generators often means juggling multiple loss functions. As a result, DLAS' configuration language is 
specifically designed to make it easy to support large number of losses and networks that interact with each other. As an 
example: some GANs I have trained in this framework consist of more than 15 losses and use 2 separate discriminators and 
require no bespoke code.

Generators are also notorious GPU memory hogs. I have spent substantial time streamlining the training framework to support 
gradient checkpointing and FP16. DLAS also supports "mega batching", where multiple forward passes contribute to a single 
backward pass. Most models can be trained on midrange GPUs with 8-11GB of memory.

The final value-added feature is interpretability. Tensorboard logging operates out of the box with no custom code. 
Intermediate images from within the training pipeline can be intermittently surfaced as normal PNG files so you can 
see what your network is up to. Validation passes are also cached as images so you can view how your network improves 
over time.

## Modeling Capabilities

DLAS was built with extensibility in mind. One of the reasons I'm putting in the effort to better document this code is the 
incredible ease with which I have been able to train entirely new model types with no changes to the core training code.

I intend to fill out the sections below with sample configurations which can be used to train different architectures. 
You will need to bring your own data.

### Super-resolution
-  [GAN-based SR (ESRGAN)](https://github.com/neonbjb/DL-Art-School/tree/gan_lab/recipes/esrgan)
- [SRFlow](https://github.com/neonbjb/DL-Art-School/tree/gan_lab/recipes/srflow)
- [GLEAN](https://github.com/neonbjb/DL-Art-School/tree/gan_lab/recipes/glean)
-  Video SR (TecoGAN) (*documentation TBC*)

### Style Transfer
* Stylegan2 (*documentation TBC*)

### Latent development
* [BYOL](https://github.com/neonbjb/DL-Art-School/tree/gan_lab/recipes/byol)
* iGPT (*documentation TBC*)

## Dependencies and Installation

- Python 3
- [PyTorch >= 1.6](https://pytorch.org)
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)
- Python packages: `pip install -r requirements.txt`
- Some video utilities require [FFMPEG](https://ffmpeg.org/)

## User Guide
TBC

### Development Environment
If you aren't already using [Pycharm](https://www.jetbrains.com/pycharm/) - now is the time to try it out. This project was built in Pycharm and comes with
an IDEA project for you to get started with. I've done all of my development on this repo in this IDE and lean heavily
on its incredible debugger. It's free. Try it out. You won't be sorry.

### Dataset Preparation
DLAS comes with some Dataset instances that I have created for my own use. Unless you want to use one of the recipes above, you'll need to provide your own. Here is how to add your own Dataset:

1.  Create a Dataset in codes/data/ which takes a single Python dict as a constructor and extracts options from that dict.
2.  Register your Dataset in codes/data/__init__.py
3.  Your Dataset should return a dict of tensors. The keys of the dict are injected directly into the training state, which you can interact within your configuration file.

### Training and Testing
There are currently 3 base scripts for interacting with models. They all take a single parameter, `-opt` which specifies the configuration file which controls how they work. Configs (will be) documented above in the user guide.

#### train.py
Start (or continue) a training session:
`python train.py -opt <your_config.yml>`

Start a distributed training session:
`python -m torch.distributed.launch --nproc_per_node=<gpus> --master_port=1234 train.py -o <opt> --launcher=pytorch`

#### test.py
Runs a model against a validation or test set of data and reports metrics (for now, just PSNR and a custom perceptual metric)
`python test.py -opt <your_config.yml>`

#### process_video.py
Breaks a video into individual frames and uses a network to do processing on it, then reassembles the output back into video form.
`python process_video -opt <your_config.yml>`

## Contributing
At this time I am not taking feature requests or bug reports, but I appreciate all contributions.

## License
This project is released under the Apache 2.0 license.
