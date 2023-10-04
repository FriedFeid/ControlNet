import torch
import os
import csv 
import OpenEXR
import Imath 
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

    def __init__(self, root_folder, control= ['Depth'],  transform = None):
        '''
        Args: 
            root_folder: str (Path to Generated_Data folder)
                --> Contains: Depth/, Distance_Map/, Edges/, Env_Maps/, Render/, SpikySphere/, Labels.csv
            control: list of Strings: [Depth, DisMap, Edges, SSphere, Env]
                --> deafult : ['Depth']
            
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

        self.Labels = self.__load_labels__()
        
        self.Render_folder = os.path.join(self.root_folder, 'Render')
        self.Renders = natsorted(glob(os.path.join(self.Render_folder, 'Render_*.png') ))
        
        self.hint_dic = {}
        for key in ['Depth', 'DisMap', 'Edges', 'SSphere', 'Env']:
            # Collecting all Folder Paths
            if key == 'Env':
                self.hint_dic[key+'_folder'] = os.path.join(self.root_folder, 'Env_Maps')
            elif key == 'DisMap':
                self.hint_dic[key+'_folder'] = os.path.join(self.root_folder, 'Distance_Map')
            elif key == 'SSphere':
                self.hint_dic[key+'_folder'] = os.path.join(self.root_folder, 'SpikySphere')
            else: 
                self.hint_dic[key+'_folder'] = os.path.join(self.root_folder, key)
            # Loading paths to Images
            if key == 'Env':
                self.hint_dic[key] = natsorted(glob(os.path.join(self.hint_dic[key+'_folder'], key + '_*.exr'))) 
            else: 
                self.hint_dic[key] = natsorted(glob(os.path.join(self.hint_dic[key+'_folder'], key + '_*.png')))
        
        self.consistant = self.__consistant__()

        self.control = control
     
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
        amount_data = len(self.Renders)
        consis = True 
        if amount_data != len(self.Labels):
            consis = False

        for key in ['Depth', 'DisMap', 'Edges', 'SSphere', 'Env']:
            if amount_data != len(self.hint_dic[key]):
                consis = False
        
        if not consis:
            print('WARNING: There are inconsitencys in the datase: ')
            print(f'Renders: {len(self.Renders)}')
            print(f'Labels: {len(self.Labels)}')
            for key in ['Depth', 'DisMap', 'Edges', 'SSphere', 'Env']:
                print(f'{key}: {len(self.hint_dic[key])}') 
            return False
        else: 
            return True
    
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

    def __load_exr__(self, exr_path):
        '''
        loads exr data: 
        
        returns:
            sun: np.array (hight, width, channels)
        '''
        path = os.path.join(exr_path)
        exr = OpenEXR.InputFile(path)
        dw = exr.header()['dataWindow']
        size = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)

        # Get correct data type
        if exr.header()['channels']['R'] == Imath.Channel(Imath.PixelType(Imath.PixelType.HALF)):
            dt = np.float16
        else:
            dt = np.float32

        # Fill numpy array
        arr = np.zeros((size[0], size[1], 3), dt)
        for i, c in enumerate(['R', 'G', 'B']):
            arr[:,:,i] = np.frombuffer(exr.channel(c), dt).reshape(size)
        
        exr = arr.astype(np.float32)
                
        return exr

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
        render = np.asarray(Image.open( self.Renders[idx] ), dtype=np.float32)/127.5 - 1.0

        # Load Hint images
        for i, key in enumerate(self.control):
            if i == 0:
                if key == 'Env':
                    hint_ = self.__load_exr__( self.hint_dic[key][idx] )
                else: 
                    hint_ = self.__load_PNG__( self.hint_dic[key][idx] )

                # reshaping if wrong size 
                # TODO: make this nicer Does not for Distance map: dimensions (hight, witdh)
                if hint_.shape == (128, 256, 3):
                    hint_temp = np.zeros_like(render)
                    hint_temp[192:320, 128:384, :] = hint_
                    hint_ = hint_temp

            else: 
                if key == 'Env':
                    temp = self.__load_exr__( self.hint_dic[key][idx] )
                else: 
                    temp = self.__load_PNG__( self.hint_dic[key][idx] )

                if temp.shape == (128, 256, 3):
                    hint_temp = np.zeros_like(render)
                    hint_temp[192:320, 128:384, :] = temp
                    temp = hint_temp

                hint_ = np.concatenate((hint_, temp), axis=2)
            

        # Get key
        keys = self.hint_dic['Env'][idx].split('.exr')[0]
        if '/' in keys:
            keys = keys.split('/')[-1]

        return dict(jpg=render, txt=self.Labels[keys]['Prompt'], hint=hint_)


# from torch.utils.data import DataLoader
# dataset = BlockWorld('/export/data/vislearn/rother_subgroup/feiden/data/ControlNet/training/Generated_Data', 
#                      ['Env'])
# dataloader = DataLoader(dataset, num_workers=0, batch_size=4, shuffle=True)

# for i , data_dic in enumerate(dataloader):
#     if i == 3: 
#         break
#     print(data_dic['txt'], data_dic['jpg'].shape, data_dic['hint'].shape)