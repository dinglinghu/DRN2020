cd src
exp_id_0=dota_drn_tv3_$1
exp_id_1=dota_drn_tv3_$2
exp_dir_0=dota_hg_drn_s1_retrain_05
exp_dir_1=dota_hg_drn_s1_retrain_05
model_0=$1
model_1=$2
# train
#python main.py rodet --dataset dota --exp_id dota_hg_l1 --K 600 --arch hourglass --batch_size 8 --master_batch 2 --lr 4e-4 --load_model ../models/ExtremeNet_500000.pth --gpus 0,1,2,3
## test
#python test.py rodet --dataset dota  --exp_id dota_hg_df_tv2 --debug 2 --trainval --K 600 --arch hourglass --keep_res --nms --load_model ../exp/rodet/dota_hg_df_trainval/model_last.pth --gpus 1 --test_split 2  --test_scales 0.5,0.75,1.0,1.25,1.5
python test.py rodet --dataset dota  --exp_id ${exp_id_0} --debug 2 --trainval --K 800 --arch hourglass --keep_res --nms --load_model ../exp/rodet/${exp_dir_0}/${model_0} --gpus 3 --test_split 3 --fsm --rot --drmc --number_stacks 1 #--test_scales 0.5,0.75,1.0,1.25,1.5,2.0
if [ -n "$2" ]
then
python test.py rodet --dataset dota  --exp_id ${exp_id_1}_1 --debug 2 --trainval --K 800 --arch hourglass --keep_res --nms --load_model ../exp/rodet/${exp_dir_1}/${model_1} --gpus 3 --test_split 3 --fsm --rot --drmc --number_stacks 1  #--test_scales 0.5,1.0,1.5
fi
#python3 test.py rodet --dataset dota --debug 4 --exp_id dota_hg_l1 --K 800 --arch hourglass --keep_res --resume --gpus 2
## flip test
#python3 test.py ctdet --dataset sku --exp_id sku_hg --K 300 --arch hourglass --keep_res --resume --flip_test
## multi scale test
#python3 test.py ctdet --dataset sku --exp_id sku_hg --K 300 --arch hourglass --keep_res --resume --flip_test --test_scales 0.5,0.75,1,1.25,1.5
#cd ..
