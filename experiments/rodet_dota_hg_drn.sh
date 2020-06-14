cd src
# train
#python main.py rodet --dataset dota --exp_id dota_hg_l1 --K 600 --arch hourglass --batch_size 8 --angle_weight 1 --master_batch 2 --lr 4e-4 --resume --gpus 1
#python3 main.py rodet --dataset dota --exp_id dota_hg_l1_RCN --K 200 --arch hourglass --batch_size 6 --master_batch 2 --lr 2e-4 --load_model ../../CenterNet/models/ExtremeNet_500000.pth --gpus 0,1,2
python3 main.py rodet --dataset dota --exp_id dota_hg_df_stack1_retrain_fp16 --K 600 --arch hourglass --batch_size 15 --master_batch 3 --lr 4e-4 --save_all --load_model ../exp/rodet/dota_hg_df_stack1_retrain_fp16/model_last.pth --gpus 0,1,2,3 --resume --angle_weight 1 --number_stacks 1 --cache_model 5 --fp16
## test
#python3 test.py ctdet --dataset sku --exp_id sku_hg --K 300 --arch hourglass --keep_res --resume
## flip test
#python3 test.py ctdet --dataset sku --exp_id sku_hg --K 300 --arch hourglass --keep_res --resume --flip_test
## multi scale test
#python3 test.py ctdet --dataset sku --exp_id sku_hg --K 300 --arch hourglass --keep_res --resume --flip_test --test_scales 0.5,0.75,1,1.25,1.5
#cd ..
