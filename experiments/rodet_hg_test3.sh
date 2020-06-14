cd src

# train
#python main.py rodet --dataset dota --exp_id dota_hg_l1 --K 600 --arch hourglass --batch_size 8 --master_batch 2 --lr 4e-4 --load_model ../models/ExtremeNet_500000.pth --gpus 0,1,2,3
## test
#python test.py rodet --dataset dota  --exp_id dota_hg_drmr3 --K 800 --arch hourglass --keep_res --nms  --load_model ../exp/rodet/dota_hg_drmr/model_43.pth --drmr --gpus 3
#python test.py rodet --dataset dota  --exp_id dota_hg_drn_s1_retrain_01_3 --K 800 --debug 2 --arch hourglass --keep_res --nms --load_model ../exp/rodet/dota_hg_drn_s1_retrain_da_.01/model_160.pth --gpus 3 --number_stacks 1  --rot --fsm
python3 test.py rodet --dataset rosku --debug 1 --exp_id rosku_hg_drn_20 --K 1000 --arch hourglass --fix_res --nms  --load_model ../exp/rodet/rosku_hg_drn_da05_drmc_drmr_ang1.5/model_20.pth --gpus 3 --trainval --fsm --rot --drmc --drmr
## flip test
#python3 test.py ctdet --dataset sku --exp_id sku_hg --K 300 --arch hourglass --keep_res --resume --flip_test
## multi scale test
#python3 test.py ctdet --dataset sku --exp_id sku_hg --K 300 --arch hourglass --keep_res --resume --flip_test --test_scales 0.5,0.75,1,1.25,1.5
#cd ..
