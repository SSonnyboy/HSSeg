# nvidia-smi

# - - - - - - - - - - - - - - - - - - - - - - - - #

expname="ACDC_base"
log_dir=./logs/${expname}
mkdir -p ${log_dir}
labeled_num=7                # TBD:   3, 7, 14
weight_his=0.4
version="base"
gpuid=2
loss_type="dice"              # ce dice ce+dice

nohup python3 ./code/train_post_2d_aut.py \
    --gpu_id=${gpuid} \
    --cfg config_2d_aut.yml \
    --exp ${expname}/v${version} \
    --labeled_num ${labeled_num} \
    --loss_type ${loss_type} \
    --weight_his ${weight_his} \
    >${log_dir}/log_v${version}.log 2>&1 &

# python3 ./code/train_post_2d_aut.py \
#     --gpu_id=${gpuid} \
#     --cfg config_2d_aut.yml \
#     --exp ${expname}/v${version}

# - - - - - - - - - - - - - - - - - - - - - - - - #
