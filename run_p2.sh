export PYTHONPATH=./:$PYTHONPATH
export TORCH_HOME=/home/zhfeing/model-zoo

export CUDA_VISIBLE_DEVICES="3"
python model_generator.py \
    --global-config configs/global-config-p2.yml \
    --cfg-path logs/run_p2/configs \
    --file_name_cfg arch-{}-encoder-{}-pretrain-{}-data-{} \
    --log-dir logs/run_p2/ \
    --gpu-preserve True \
    --debug False

