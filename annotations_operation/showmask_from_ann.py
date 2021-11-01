# -*- coding:utf-8 -*-

from __future__ import print_function
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import skimage.io as io
import pylab

pylab.rcParams['figure.figsize'] = (8.0, 10.0)

ann_file = 'G:/xiao/dataset_molcreate/create_ann/instances_train2017.json'
coco = COCO(ann_file)

cats = coco.loadCats(coco.getCatIds())
nms = [cat['name'] for cat in cats]
# supercategory
nms = set([cat['supercategory'] for cat in cats])

imgIds = coco.getImgIds()
img = coco.loadImgs(imgIds[1])[0]
print(img)
dataType = 'G:/xiao/dataset_molcreate/create_ann/image/'
I = io.imread('%s/%s' % (dataType, img['file_name']))

plt.axis('off')
plt.imshow(I)
plt.show()

# 加载和可视化instance标注信息
catIds = []
for ann in coco.dataset['annotations']:
    if ann['image_id'] == imgIds[0]:
        catIds.append(ann['category_id'])

plt.imshow(I)
plt.axis('off')
annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
anns = coco.loadAnns(annIds)
coco.showAnns(anns)
plt.show()

