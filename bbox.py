# -*- coding: utf-8 -*-
from __future__ import (
    division,
    print_function,
)

import skimage.data
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import selectivesearch
import rasterio
import numpy as np
import cv2


def main():

    # loading astronaut image
    img = skimage.data.astronaut()
    # print(img.shape)
    # print(img)

    with rasterio.open('./SX9192.TIF', 'r') as f:
        values = f.read().astype(np.float64)
        values = np.swapaxes(values,0,1)
        values = np.swapaxes(values,1,2)
        # img = values

    p = cv2.imread('./SX9192.TIF')
    p = cv2.cvtColor(p,cv2.COLOR_BGR2RGB)
    # print(p.shape)
    img =  cv2.resize(p,(512,512))


    # perform selective search
    img_lbl, regions = selectivesearch.selective_search(
        img, scale=100, sigma=0.5, min_size=5)

    candidates = set()
    for r in regions:
        # excluding same rectangle (with different segments)
        if r['rect'] in candidates:
            continue
        # excluding regions smaller than 2000 pixels
        if r['size'] < 2000:
            continue
        # distorted rects
        x, y, w, h = r['rect']
        if w / h > 1.2 or h / w > 1.2:
            continue
        candidates.add(r['rect'])

    # draw rectangles on the original image
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.imshow(img)
    for x, y, w, h in candidates:
        print(x, y, w, h)
        rect = mpatches.Rectangle(
            (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)

    plt.show()

if __name__ == "__main__":
    main()
