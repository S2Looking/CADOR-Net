# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 13:56:39 2023

@author: user
"""


import matplotlib.pyplot as plt

Label = ('__background__',  # always index 0
        'aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train', 'tvmonitor')

Num = 0
img = im_data[Num,:,:,:]
plt.imshow(img.permute(1,2,0).numpy()/50+ 1 )
currentAxis = plt.gca()

colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()

# [xmin, ymin, xmax, ymax]
for i in range(num_boxes[Num]):
    # roi = labels[Num][i,:].numpy() * [height,width,height,width,1]
    roi = gt_boxes[Num][i,:].numpy()
    print(roi)
    item = int(gt_boxes[Num][i,4].item()) # 类别
    color = colors[item]
    currentAxis.add_patch(
        plt.Rectangle((roi[0], roi[1]),
                      roi[2] - roi[0],
                      roi[3] - roi[1], fill=False,
                      edgecolor=color, linewidth=2)
        )
    currentAxis.text(roi[0], roi[1], '{}: {}'.format(Label[item],item), bbox={'facecolor':color, 'alpha':0.7})
plt.show()