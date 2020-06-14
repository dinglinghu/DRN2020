#python3 src/demo.py rodet --demo ./images/sku/ --debug 4 --dataset rosku --exp_id rosku_hg_demo --K 500 --arch hourglass --vis_thresh 0.1 --fix_res --gpus 1 --load_model exp/rodet/rosku_hg/model_last.pth

#python3 src/demo.py rodet --demo ./images/sku/test --debug 4 --dataset rosku --exp_id rosku_hg_demo --K 500 --arch hourglass --vis_thresh 0.1 --fix_res --gpus 1 --load_model ./models/rosku_hg0623_angle_l1_w1_model_15model_best.pth


#python3 src/test.py rodet  --dataset rosku --exp_id rosku_hg_test --K 500 --arch hourglass --fix_res --gpus 0,1  --batch_size 2 --load_model ./models/rosku_hg0623_angle_l1_w1_model_15model_best.pth
python3 src/test.py rodet  --dataset rosku --exp_id rosku_hg_test --K 1000 --nms --arch hourglass --fix_res --gpus 2 --batch_size 2 --load_model exp/rodet/rosku_hg0623-mse100/model_best.pth
#python3 src/test.py rodet  --dataset rosku --exp_id rosku_hg_test --K 1000 --nms --flip_test --test_scales 0.8,1,1.2 --arch hourglass --fix_res --gpus 2 --batch_size 2 --load_model ./models/rosku_hg0623_angle_l1_w1_model_15model_best.pth

#python3 src/test.py rodet  --dataset rosku --exp_id rosku_hg_test --K 500 --arch hourglass --fix_res --gpus 0,1  --batch_size 2 --load_model ./exp/rodet/rosku_hg0623-mse100/model_last.pth
