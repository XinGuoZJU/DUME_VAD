Note: Our code is partly based on the MNAD(https://github.com/cvlab-yonsei/MNAD).

Installation:

Replace cudatoolkit=10.1 with your CUDA version: https://pytorch.org/
```
conda install -y pyyaml docopt matplotlib scikit-image opencv pytorch cudatoolkit=10.1 -c pytorch
```


Double Memory Anomaly Detection

Train:
```
python train.py --gpu 0 1 --batch_size 4 --test_batch_size 4 --config_file configs/shanghaitech.ini --kfolds 0
```
Test:
```
python evaluate.py --gpu 0 --model_dir exp/ShanghaiTech_AUG --dataset shanghaitech --kfolds 0 --alpha 0.6 --save_pred_img --save_query
```

Make Demo:
```
cd viz/tools
python create_demo.py
```
Video Compress:
```
ffmpeg -i input.mp4 -r 10 -b:a 32k output.mp4
```
