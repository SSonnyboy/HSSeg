# # - - - - - - - -      Testing      - - - - - - - # 

# nvidia-smi

##############################################################

# - - - - - - - - - - - - - - - - - - - - - # 
#                   LA
# - - - - - - - - - - - - - - - - - - - - - # 

expname="LA_base"
version="base"
numlb=4 # 4, 8, 16
gpuid=0
# /home/chenyu/SSMIS/data/LA/data/UPT6DX9IQY9JAZ7HJKA7/mri_norm2.h5
python3 ./code/test_performance_3d.py \
    --root_path /home/chenyu/SSMIS/data/LA/ \
    --res_path ./results/LA/ \
    --gpu ${gpuid} \
    --exp ${expname}/v${version} \
    --flag_check_with_last \
    --labeled_num ${numlb} 

