# nvidia-smi

# - - - - - - - -     Testing      - - - - - - - # 
expname="ACDC_base"
version="base"
numlb=7 # 3, 7, 14
gpuid=0

python3 ./code/test_performance_2d.py \
    --root_path /home/chenyu/SSMIS/data/ACDC \
    --res_path ./results/ACDC \
    --gpu_id=${gpuid} \
    --exp ${expname}/v${version} \
    --flag_check_with_best_stu \
    --labeled_num ${numlb}
