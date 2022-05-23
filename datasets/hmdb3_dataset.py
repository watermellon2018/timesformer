from torch.utils.data import Dataset

import json
import cv2
import torch

class ActionDataset(Dataset):
    def __init__(self, json_path, count_frame=8, gap=2, transform=None):
        with open(json_path, "r") as read_file:
            self.vids_path = json.load(read_file)
        self.transform = transform
        self.count_frame = count_frame
        self.gap = gap

    def __len__(self):
        return len(self.vids_path)

    def read_video(self, path):
        cap = cv2.VideoCapture(path)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
        frames = []

        if length < self.count_frame * self.gap:
            raise Exception('Not enougth frames')

        for num_frame in range(0, self.count_frame * self.gap, self.gap):
            cap.set(cv2.CAP_PROP_POS_FRAMES, num_frame)
            ret, frame = cap.read()
            if ret == False:
                raise Exception('Not read frame')

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if self.transform:
                frame = self.transform(image=frame)['image']
            frames.append(frame)

        frames = torch.tensor(frames, dtype=torch.float)
        frames = torch.permute(frames, (0, 3, 1, 2))
        return frames

    def __getitem__(self, ind):
        path, lbl = self.vids_path[ind]
        frames = self.read_video(path)
        return frames, lbl