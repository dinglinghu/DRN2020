cd src
# train
python3 main.py ctdet --dataset sku --exp_id sku_hg_temp --K 300 --arch hourglass --batch_size 8 --master_batch 4 --lr 1e-4 --load_model ../models/ExtremeNet_500000.pth --gpus 2,3
## test
#python3 test.py ctdet --dataset sku --exp_id sku_hg --K 300 --arch hourglass --keep_res --resume
## flip test
#python3 test.py ctdet --dataset sku --exp_id sku_hg --K 300 --arch hourglass --keep_res --resume --flip_test
## multi scale test
#python3 test.py ctdet --dataset sku --exp_id sku_hg --K 300 --arch hourglass --keep_res --resume --flip_test --test_scales 0.5,0.75,1,1.25,1.5
cd ..