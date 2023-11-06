import os

class Dataloader():
    def __init__(self, data_folder_l, data_folder_r) -> None:
        self.data_folder_l = data_folder_l
        self.data_folder_r = data_folder_r

        # получаю списки изображений
        self.filenames1 = sorted(os.listdir(self.data_folder_l))
        self.filenames2 = sorted(os.listdir(self.data_folder_r))