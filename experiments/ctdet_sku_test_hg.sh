#python3 src/test.py ctdet --dataset sku --exp_id sku_hg --K 300 --arch hourglass --fix_res --load_model exp/ctdet/sku_hg/model_last.pth
python3 src/test.py ctdet --debug 4 --dataset sku --exp_id sku_hg --K 500 --arch hourglass --fix_res --load_model exp/ctdet/sku_hg/model_120.pth