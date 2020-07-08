export PYTHONPATH=./:/home/zhfeing/project/generate-models/segmentation_models.pytorch:$PYTHONPATH
export TORCH_HOME=/home/zhfeing/model-zoo

export CUDA_VISIBLE_DEVICES="2, 3"
python model_generator.py \
    --global-config configs/global-config-p1.yml \
    --cfg-path logs/test_p1/configs \
    --cfg-formater arch-{}-encoder-{}-data-{}.yml \
    --log-dir logs/test_p1/ \
    --gpu-preserve True \
    --debug False

