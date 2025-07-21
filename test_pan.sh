# # - - - - - - - -      Testing      - - - - - - - # 

# nvidia-smi

##############################################################

# # - - - - - - - - - - - - - - - - - - - - - # 
# #                   Pancrease
# # - - - - - - - - - - - - - - - - - - - - - # 

expname="Pancrease_base"
version="base"
numlb=6 # 6, 12
gpuid=2

python3 ./code/test_performance_3d.py \
    --root_path /home/chenyu/SSMIS/data/Pancreas/ \
    --res_path ./results/Pancreas/ \
    --dataset "Pancreas" \
    --gpu ${gpuid} \
    --exp ${expname}/v${version} \
    --labeled_num ${numlb} \
    --flag_check_with_last

