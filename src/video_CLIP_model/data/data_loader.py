try:
    from data_utils import load_text,load_video,make_video_transform
except ImportError:
    from data.data_utils import load_text,load_video,make_video_transform
import torch 
from torch.utils.data import Dataset
import json

class TextRetrivalDataset(Dataset):
    def __init__(self, 
                 annotation_path, 
                 splits = None, 
                 load_all_text = True,
                 load_all_video = False,
                 **kwargs):
        with open(annotation_path,'r',encoding = 'utf-8') as f:
            annotation = json.load(f)

        for data_point in annotation:
            try: 
                if data_point['split'] in splits:
                    if not load_all_text:
                        data_point
            except Exception as e:
                pass