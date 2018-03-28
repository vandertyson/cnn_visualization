source activate vis
echo "Unet visualization"
iteration=20
img_path="./9293cpy/SX9293_216_180.TIF"
weights_path="./3band.h5"
channel=3
layer="conv2d_2"
trained=1
python viz.py \
        --iteration=${iteration} \
        --img=${img_path} \
        --weights_path=${weights_path} \
        --channel=${channel} \
        --layer=${layer} \
        --trained=${trained} 
# $SHELL