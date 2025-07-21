# nvidia-smi

# - - - - - - - -     Testing      - - - - - - - # 
expname="ACDC_runs"
version="cps_hs"
numlb=3 # 3, 7, 14
gpuid=5

python3 ./code/test_performance_2d.py \
    --root_path /home/chenyu/SSMIS/data/ACDC \
    --res_path ./results/ACDC \
    --gpu_id=${gpuid} \
    --exp ${expname}/v${version} \
    --labeled_num ${numlb} \
    --model unet_hsseg  \
    --model_ext unet_hsseg \
    --model_i model1
