import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer


class make_cfg:
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config())