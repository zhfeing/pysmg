export PYTHONPATH=./:/home/zhfeing/projects/generate-models/segmentation_models.pytorch:$PYTHONPATH
export TORCH_HOME=/home/zhfeing/model-zoo

export CUDA_VISIBLE_DEVICES="0, 1"
python train.py --config=configs/test-voc-config.yml &
export CUDA_VISIBLE_DEVICES="2, 3"
python train.py --config=configs/test-camvid-config.yml