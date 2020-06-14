cd src
# train
#python main.py rodet --dataset dota --exp_id dota_hg_l1 --K 600 --arch hourglass --batch_size 8 --angle_weight 1 --master_batch 2 --lr 4e-4 --resume --gpus 1
#python3 main.py rodet --dataset dota --exp_id dota_hg_l1_RCN --K 200 --arch hourglass --batch_size 6 --master_batch 2 --lr 2e-4 --load_model ../../CenterNet/models/ExtremeNet_500000.pth --gpus 0,1,2
python3 main.py rodet --dataset dota --exp_id dota_hg_df12 --K 200 --arch hourglass --batch_size 2 --master_batch 2 --lr 2.5e-5 --save_all --val_intervals 1 --load_model ../models/ExtremeNet_500000.pth --gpus 0
## test
#python3 test.py ctdet --dataset sku --exp_id sku_hg --K 300 --arch hourglass --keep_res --resume
## flip test
#python3 test.py ctdet --dataset sku --exp_id sku_hg --K 300 --arch hourglass --keep_res --resume --flip_test
## multi scale test
#python3 test.py ctdet --dataset sku --exp_id sku_hg --K 300 --arch hourglass --keep_res --resume --flip_test --test_scales 0.5,0.75,1,1.25,1.5
#cd ..
