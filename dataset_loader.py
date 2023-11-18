import torch
import torch.utils.data as data
import os
import numpy as np
import utils 

class XDVideo(data.DataLoader):
    def __init__(self, root_dir, mode, num_segments, len_feature, seed=-1, is_normal=None):
        if seed >= 0:
            utils.set_seed(seed)
        self.data_path=root_dir
        self.mode=mode
        self.num_segments = num_segments
        self.len_feature = len_feature
        
        self.feature_path = self.data_path
        split_path = os.path.join("list",'XD_{}.list'.format(self.mode))
        split_file = open(split_path, 'r',encoding="utf-8")
        self.vid_list = []
        for line in split_file:
            self.vid_list.append(line.split())
        split_file.close()
        if self.mode == "Train":
            if is_normal is True:
                self.vid_list = self.vid_list[9525:]
            elif is_normal is False:
                self.vid_list = self.vid_list[:9525]
            else:
                assert (is_normal == None)
                print("Please sure is_normal = [True/False]")
                self.vid_list=[]
        
    def __len__(self):
        return len(self.vid_list)

    def __getitem__(self, index):
        data,label = self.get_data(index)
        return data, label

    def get_data(self, index):
        vid_name = self.vid_list[index][0]
        label=0
        if "_label_A" not in vid_name:
            label=1  
        video_feature = np.load(os.path.join(self.feature_path, vid_name )).astype(np.float32)
        if self.mode == "Train":
            new_feature = np.zeros((self.num_segments, self.len_feature)).astype(np.float32)

            sample_index = np.linspace(0, video_feature.shape[0], self.num_segments+1, dtype=np.uint16)

            for i in range(len(sample_index)-1):
                if sample_index[i] == sample_index[i+1]:
                    new_feature[i,:] = video_feature[sample_index[i],:]
                else:
                    new_feature[i,:] = video_feature[sample_index[i]:sample_index[i+1],:].mean(0)
                    
            video_feature = new_feature
        return video_feature, label    
