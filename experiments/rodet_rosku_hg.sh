cd src
# train
python3 main.py rodet --dataset rosku --exp_id rosku_hg_drn_da05_drmc_drmr_ang1.5 --K 600 --arch hourglass --batch_size 15 --master_batch 3 --lr 2e-4 --load_model ../models/ExtremeNet_500000.pth --gpus 0,1,2,3 --save_all --cache_model 5 --dense_angle --fsm --rot --drmc --drmr --e2e --angle_weight 1.5
## test
#python3 test.py ctdet --dataset sku --exp_id sku_hg --K 300 --arch hourglass --keep_res --resume
## flip test
#python3 test.py ctdet --dataset sku --exp_id sku_hg --K 300 --arch hourglass --keep_res --resume --flip_test
## multi scale test
#python3 test.py ctdet --dataset sku --exp_id sku_hg --K 300 --arch hourglass --keep_res --resume --flip_test --test_scales 0.5,0.75,1,1.25,1.5
#cd ..
