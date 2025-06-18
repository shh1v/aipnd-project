# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

## Usage Examples

#### Using train.py
```
python train.py flowers/ --save_dir script_checkpoints/ --epochs 10 --hidden_units 640 320 160 --gpu
```

#### Using predict.py
```
python predict.py flowers/test/1/image_06743.jpg script_checkpoints/checkpoint_2025-06-18_00-27-35.pth --top_k 4 --gpu
```