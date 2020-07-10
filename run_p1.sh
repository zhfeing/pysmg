export PYTHONPATH=./:/home/zhfeing/project/generate-models/segmentation_models.pytorch:$PYTHONPATH
export TORCH_HOME=/home/zhfeing/model-zoo

export CUDA_VISIBLE_DEVICES="1, 2"
python model_generator.py \
    --global-config configs/global-config-p1.yml \
    --cfg-path logs/run_p1/configs \
    --cfg-formatter arch-{}-encoder-{}-data-{}.yml \
    --log-dir logs/run_p1/ \
    --gpu-preserve True \
    --debug False

