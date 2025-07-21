# nvidia-smi

# - - - - - - - - - - - - LA - - - - - - - - - - - - #

expname="LA_base"
log_dir=./logs/${expname}
mkdir -p ${log_dir}
weight_his=0.8
version="base"
gpuid=1
labeled_num=4                # TBD:   4, 8, 16
nohup python3 ./code/train_post_3d_aut.py \
    --gpu_id=${gpuid} \
    --cfg config_3d_la_aut.yml \
    --exp ${expname}/v${version} \
    --labeled_num ${labeled_num} \
    --weight_his ${weight_his} \
    >${log_dir}/log_v${version}.log 2>&1 &

# python3 ./code/train_post_3d_aut.py \
#     --gpu_id=${gpuid} \
#     --cfg config_3d_la_aut.yml \
#     --exp ${expname}/v${version}
#     >${log_dir}/log_v${version}.log 2>&1 &
