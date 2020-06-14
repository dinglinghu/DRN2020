cd src

# train
#python main.py rodet --dataset dota --exp_id dota_hg_l1 --K 600 --arch hourglass --batch_size 8 --master_batch 2 --lr 4e-4 --load_model ../models/ExtremeNet_500000.pth --gpus 0,1,2,3
## test
python test.py rodet --dataset dota --exp_id dota_hg_dfc4 --K 800 --arch hourglass --keep_res --nms --load_model ../exp/rodet/dota_hg_dfc4/model_25.pth --gpus 3 --drmc

#python3 test.py rodet --dataset dota --debug 4 --exp_id dota_hg_l1 --K 800 --arch hourglass --keep_res --resume --gpus 2
## flip test
#python3 test.py ctdet --dataset sku --exp_id sku_hg --K 300 --arch hourglass --keep_res --resume --flip_test
## multi scale test
#python3 test.py ctdet --dataset sku --exp_id sku_hg --K 300 --arch hourglass --keep_res --resume --flip_test --test_scales 0.5,0.75,1,1.25,1.5
#cd ..
