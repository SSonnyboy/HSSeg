# nvidia-smi
# # # - - - - - - - - - - - Pancrease - - - - - - - - - - - - - #

expname="Pancrease_base"
log_dir=./logs/${expname}
mkdir -p ${log_dir}
weight_his=0.8
version="base"
gpuid=2
labeled_num=6                # TBD:   6, 12
nohup python3 ./code/train_post_3d_aut.py \
    --gpu_id=${gpuid} \
    --cfg config_3d_pan_aut.yml \
    --patch_size 96 96 96 \
    --exp ${expname}/v${version} \
    --labeled_num ${labeled_num} \
    --weight_his ${weight_his} \
    >${log_dir}/log_v${version}.log 2>&1 &


# python3 ./code/train_post_3d_aut.py \
#     --gpu_id=${gpuid} \
#     --cfg config_3d_pan_aut.yml \
#     --patch_size 96 96 96 \
#     --exp ${expname}/v${version} 
