from x.features.visual_genome import register_all_vg
import random
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data.catalog import DatasetCatalog
import cv2
import os
import matplotlib.pyplot as plt

if __name__ =="__main__":
    register_all_vg()
    vg_train_metadata = MetadataCatalog.get("visual_genome_train")
    dataset_dicts = DatasetCatalog.get("visual_genome_train")

    for d in random.sample(dataset_dicts,3):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1],
                                metadata=vg_train_metadata,scale=0.5)
        vis = visualizer.draw_dataset_dict(d)
        plt.imshow(vis.get_image()[:,:,::-1])
        plt.show()
        #cv2.imshow("",vis.get_image()[:, :, ::-1])