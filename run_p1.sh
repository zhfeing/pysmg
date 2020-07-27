export PYTHONPATH=./:$PYTHONPATH
export TORCH_HOME=/home/zhfeing/model-zoo

export CUDA_VISIBLE_DEVICES="1, 2"
python model_generator.py \
    --global-config configs/global-config-p1.yml \
    --cfg-path logs/run_p1/configs \
    --file_name_cfg arch-{}-encoder-{}-pretrain-{}-data-{} \
    --log-dir logs/run_p1/ \
    --gpu-preserve True \
    --debug False

