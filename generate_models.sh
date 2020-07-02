#!/bin/bash

datasets=(
        pascal
        camvid
        ade20k
        # mit_sceneparsing_benchmark
        cityscapes
        nyuv2
        sunrgbd
        # vistas
)

architectures=(
    Unet
    Linknet
    FPN
    PSPNet
    PAN
    DeepLabV3
)

for ds in ${datasets[@]};
do
    for arch in ${architectures[@]};
    do
        echo $ds, $arch
    done
done

