import torch
import os
import csv 
# import OpenEXR
# import Imath 
import numpy as np

from glob import glob 
from natsort import natsorted
from PIL import Image 
from torch.utils.data import Dataset 

class BlockWorld(Dataset):
    '''
    Dataloader which loads all data generated in the Synthetic dataset. 
    Namely: render, edges, depth, ssphere, 11_params, sun, sky, hdr, prompt
    returns: 
            render: torch.Tensor (Batch_size, Channels, Hight, width) 
            depth: torch.Tensor (Batch_size, Channels, Hight, width) 
            edge: torch.Tensor (Batch_size, Channels, Hight, width) 
            distance_map: torch.Tensor (Batch_size, Channels, Hight, width) 
            spiky_sphere: torch.Tensor (Batch_size, Channels, Hight, width) 
            param_dic: dict 
            --> (keys: 'name', 'wsun', 'kappa', 'beta', 't', 'wsky', 'sunpos_u', 'sunpos_v', 'Prompt')
                when you use param_dic['Prompt'] you will get a list [Batch_size] of Prompts

    '''

    def __init__(self, root_folder, transform = None):
        '''
        Args: 
            root_folder: str (Path to Generated_Data folder)
                --> Contains: Depth/, Distance_Map/, Edges/, Env_Maps/, Render/, SpikySphere/, Labels.csv
            
            transform: Callable (optional, Transformation which schould be applied to data)
        
        returns: 
            render: torch.Tensor (Batch_size, Channels, Hight, width) 
            depth: torch.Tensor (Batch_size, Channels, Hight, width) 
            edge: torch.Tensor (Batch_size, Channels, Hight, width) 
            distance_map: torch.Tensor (Batch_size, Channels, Hight, width) 
            spiky_sphere: torch.Tensor (Batch_size, Channels, Hight, width) 
            param_dic: dict 
            --> (keys: 'name', 'wsun', 'kappa', 'beta', 't', 'wsky', 'sunpos_u', 'sunpos_v', 'Prompt')
                when you use param_dic['Prompt'] you will get a list [Batch_size] of Prompts
        '''
        self.root_folder = root_folder
        self.transform = transform

        self.Env_Maps_folder = os.path.join(self.root_folder, 'Env_Maps')
        self.Depth_folder = os.path.join(self.root_folder, 'Depth')
        self.Distance_Map_folder = os.path.join(self.root_folder, 'Distance_Map')
        self.Edges_folder = os.path.join(self.root_folder, 'Edges')
        self.Render_folder = os.path.join(self.root_folder, 'Render')
        self.SpikySphere_folder = os.path.join(self.root_folder, 'SpikySphere')

        self.Labels = self.__load_labels__()
        
        self.Renders = natsorted(glob('Render_*.png', root_dir = self.Render_folder))
        self.Depths = natsorted(glob('Depth_*.png', root_dir = self.Depth_folder )) 
        self.Distances = natsorted(glob('DisMap_*.png', root_dir = self.Distance_Map_folder))
        self.Edges = natsorted(glob('Edges_*.png', root_dir = self.Edges_folder))
        self.SpikySpheres = natsorted(glob('SSphere_*.png', root_dir = self.SpikySphere_folder))
        self.EnvMaps = natsorted(glob('Env_*.exr', root_dir = self.Env_Maps_folder))
        self.consistant = self.__consistant__()
     
    def __len__(self):
        '''
        Returns the number of Renders
        '''
        return len(self.Renders)

    def __consistant__(self):
        '''
        Checks if the Dataset is self consistant

        returns:   
            consitant: bool 
        '''
        if (len(self.Renders) == len(self.Depths)) & (
            len(self.Depths) == len(self.Distances)) & (
            len(self.Distances) == len(self.Edges)) & (
            len(self.Edges) == len(self.SpikySpheres)):
            return True
        else: 
            print('WARNING: There are inconsitencys in the datase: ')
            print(f'Renders: {len(self.Renders)}') 
            print(f'Dephts: {len(self.Depths)}') 
            print(f'Distance Maps: {len(self.Distances)}') 
            print(f'Edge Maps: {len(self.Edges)}')
            print(f'Spiky Spheres: {len(self.SpikySpheres)}') 
            return False
    
    def __load_labels__(self):
        '''
        Loads Labels cvs as list of dictionarys.
        '''
        param_dic = {}
        with open(os.path.join(self.root_folder, 'Labels.csv'), newline='') as tab:
            reader = csv.DictReader(tab, delimiter=',', )
            for row in reader: 
                dic = {}
                for key in row: 
                    if key == 'wsun' or key == 'wsky':
                        vec = np.array([float(row[key].split(' ')[0]),
                                            float(row[key].split(' ')[1]),
                                            float(row[key].split(' ')[2]),])
                        dic[key] = vec
                    elif key == 'name' or key == 'Prompt':
                        dic[key] = row[key]
                    else:
                        dic[key] = float(row[key])
                param_dic[row['name']] = dic

        return param_dic
    
    def __load_PNG__(self, picture_path):
        '''
        Loads PNG data

        returns: 
            Render: np.array(hight, width, channels)
            
            torch.Tensor (Channels, hight, width)
        ''' 
        png = np.asarray(Image.open(picture_path), dtype=np.float32)/255.

        return png 

    # def __load_exr__(self, instance_index):
    #     '''
    #     loads exr data: 
        
    #     returns:
    #         sun: torch.Tensor (Sun model)
    #         sky: torch.Tensor (Sky model)
    #         env: torch.Tensor (Environment map) 
    #     '''
    #     exr_s = {}
    #     for key in ['Sun', 'Sky', 'Env']:
    #         path = os.path.join(self.Env_Maps_folder, key+f'_{instance_index:06d}.exr')
    #         exr = OpenEXR.InputFile(path)
    #         dw = exr.header()['dataWindow']
    #         size = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)

    #         # Get correct data type
    #         if exr.header()['channels']['R'] == Imath.Channel(Imath.PixelType(Imath.PixelType.HALF)):
    #             dt = np.float16
    #         else:
    #             dt = np.float32

    #         # Fill numpy array
    #         arr = np.zeros((size[0], size[1], 3), dt)
    #         for i, c in enumerate(['R', 'G', 'B']):
    #             arr[:,:,i] = np.frombuffer(exr.channel(c), dt).reshape(size)
            
    #         exr = arr.astype(np.float32)
    #         exr_s[key] = torch.from_numpy(np.moveaxis(exr, -1, 0))
                
    #     return exr_s['Sun'], exr_s['Sky'], exr_s['Env']

    def __getitem__(self, idx):
        '''
        Loads Item of Dataset

        returns: 
            render:
            depth:
            edge:
            distance_map:
            spiky_sphere: 
            param_dic:
        '''

        if torch.is_tensor(idx):
            idx = idx.tolist()
        

        # Normalize traget image to [-1, 1]
        render = np.asarray(Image.open(os.path.join(self.Render_folder, self.Renders[idx])), dtype=np.float32)/127.5 - 1.0
        
        # Load Hint images
        depth = self.__load_PNG__(os.path.join(self.Depth_folder, self.Depths[idx]))
        edge = self.__load_PNG__(os.path.join(self.Edges_folder, self.Edges[idx]))
        distance_map = self.__load_PNG__(os.path.join(self.Distance_Map_folder, self.Distances[idx]))
        spiky_sphere = self.__load_PNG__(os.path.join(self.SpikySphere_folder, self.SpikySpheres[idx]))
        
        keys = self.EnvMaps[idx].split('.exr')[0]

        return dict(jpg=render, txt=self.Labels[keys]['Prompt'], hint=depth)

