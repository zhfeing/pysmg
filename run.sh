export PYTHONPATH=./:/home/zhfeing/project/generate-models/segmentation_models.pytorch:$PYTHONPATH
export TORCH_HOME=/home/zhfeing/model-zoo

export CUDA_VISIBLE_DEVICES="0, 1"
python model_generator.py \
    --global-config configs/global-config.yml \
    --cfg-path logs/test/configs \
    --cfg-formatter arch-{}-encoder-{}-data-{}.yml \
    --log-dir logs/test/ \
    --gpu-preserve True \
    --debug False

