export PYTHONPATH=./:/home/zhfeing/projects/generate-models/segmentation_models.pytorch:$PYTHONPATH
export TORCH_HOME=/home/zhfeing/model-zoo

export CUDA_VISIBLE_DEVICES="0, 2"
# python train.py --config=test/test.yml
python model_generator.py \
    --global-config configs/global-config.yml \
    --cfg-filepath logs/test/arch-{}-encoder-{}-data-{}.yml \
    --log-dir logs/test/ \
    --gpu-preserve False \
    --debug False

