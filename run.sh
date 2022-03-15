# Train:
python train.py --gpu 0 1 --batch_size 4 --test_batch_size 4 --config_file configs/shanghaitech.ini --kfolds 0

# Test:
# python evaluate.py --gpu 0 --model_dir exp/ShanghaiTech_AUG --dataset shanghaitech --kfolds 0 --alpha 0.6 --save_pred_img --save_query
# python evaluate_org.py --gpu 1 --model_dir exp/ShanghaiTech_AUG  --kfolds 0 --save_pred_img


# python distance.py --gpu 0 --model_dir exp/Ped2_AUG --dataset ped2 --kfolds 0 --alpha 0.6
