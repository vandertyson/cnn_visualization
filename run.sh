echo "Unet visualization"
iteration=20
img_path="./SX9193_028_8band.TIF"
weights_path="./8band.h5"
channel=8
layer="conv2d_16"
python viz.py \
        --iteration=${iteration} \
        --img=${img_path} \
        --weights_path=${weights_path} \
        --channel=${channel} \
        --layer=${layer} 
$SHELL