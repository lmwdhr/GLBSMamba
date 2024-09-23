# GLBSMamba

***MMRS*** is a python tool to perform deep learning experiments on multi-modal remote sensing data.

This repository is developed on the top of [MMRS](https://github.com/likyoo/Multimodal-Remote-Sensing-Toolkit) . 


## Usage

Start a Visdom server: `python -m visdom.server` and go to [`http://localhost:8097`](http://localhost:8097/) to see the visualizations.

Then, run the script `main.py`.

The most useful arguments are:

- `--model` to specify the model (e.g. 'Multimodality_Mamba', 'S2ENet'),
- `--dataset` to specify which dataset to use (e.g. 'Houston2013', 'Muufl'),
- the `--cuda` switch to run the neural nets on GPU. The tool fallbacks on CPU if this switch is not specified.

There are more parameters that can be used to control more finely the behaviour of the tool. See `python main.py -h` for more information.

Examples:

```
!python main.py --model S2ENet --flip_augmentation --patch_size 7 --epoch 128 --lr 0.001 --batch_size 64 --seed 0 --dataset Houston2013 --folder '../' --train_set '../Houston2013/TRLabel.mat' --test_set '../Houston2013/TSLabel.mat' --cuda 0
```

For more features please refer to [DeepHyperX](https://github.com/nshaud/DeepHyperX).

