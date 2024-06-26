from __future__ import absolute_import

import os
import glob
import numpy as np

from siamfc import TrackerSiamFC


if __name__ == '__main__':
    seq_dir = os.path.expanduser('~/dataset/OTB2015/OTB100/Crossing/')
    img_files = sorted(glob.glob(seq_dir + 'img/*.jpg'))
    # anno = np.loadtxt(seq_dir + 'groundtruth_rect.txt')
    anno = np.genfromtxt(seq_dir + 'groundtruth_rect.txt', delimiter=',', dtype=int)
    net_path = 'pretrained/siamfc_alexnet_e50.pth'
    tracker = TrackerSiamFC(net_path=net_path)
    tracker.track(img_files, anno[0], visualize=True)
