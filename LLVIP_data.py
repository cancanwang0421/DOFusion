import os
from PIL import Image
from torch.utils import data
from torchvision import transforms
import cv2

to_tensor = transforms.Compose([transforms.ToTensor()])
    
class LLVIP_data(data.Dataset):
    def __init__(self, data_dir, transform=to_tensor):
        super().__init__()
        dirname = os.listdir(data_dir)
        for sub_dir in dirname:
            temp_path = os.path.join(data_dir, sub_dir)
            if sub_dir == 'infrared':
                self.Inf_path = temp_path
            if sub_dir == 'visible':
                self.vi_path = temp_path
        self.name_list = os.listdir(self.Inf_path)
        self.transform = transform
        

    def __getitem__(self, index):
        name = self.name_list[index]
        Inf = Image.open(os.path.join(self.Inf_path, name)).convert('L')
        vi = Image.open(os.path.join(self.vi_path, name)).convert('L')

        Inf = self.transform(Inf)
        vi = self.transform(vi)
        return Inf,vi,name
        

    def __len__(self):
        return len(self.name_list)
        
