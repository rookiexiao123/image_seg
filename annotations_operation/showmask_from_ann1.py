# -*- coding:utf-8 -*-

from __future__ import print_function
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import skimage.io as io
import pylab

pylab.rcParams['figure.figsize'] = (8.0, 10.0)

ann_file = 'G:/xiao/dataset_molcreate/create_ann/instances_train2017.json'
dataType = 'G:/xiao/dataset_molcreate/create_ann/image/'
# ann_file = 'G:/xiao/mol_recognition/dataset_via/data_maskexpand/annotations/instances_train2017.json'
# dataType = 'G:/xiao/mol_recognition/dataset_via/data_maskexpand/train2017'

coco = COCO(ann_file)

# cats = coco.loadCats(coco.getCatIds())
# nms = [cat['name'] for cat in cats]
# # supercategory
# nms = set([cat['supercategory'] for cat in cats])

imgIds = coco.getImgIds()

for i in range(len(imgIds)):
    img = coco.loadImgs(imgIds[i])[0]
    print(img['file_name'])
    I = io.imread('%s/%s' % (dataType, img['file_name']))
    # plt.axis('off')
    # plt.imshow(I)
    # plt.show()

    # 加载和可视化instance标注信息
    catIds = []
    for ann in coco.dataset['annotations']:
        if ann['image_id'] == imgIds[0]:
            catIds.append(ann['category_id'])

        m = coco.annToMask(ann)

        print(m.shape)
        plt.imshow(m)
        plt.show()

    plt.imshow(I)
    #plt.axis('off')
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    coco.showAnns(anns)
    plt.show()

