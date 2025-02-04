from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from VirtualWorldNoShadows_dataset import BlockWorld
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from pytorch_lightning.callbacks import ModelCheckpoint


# Configs
resume_path = '/export/data/vislearn/rother_subgroup/feiden/models/pretrained/ControlNet/control_sd21_ini.ckpt'
# '.\models\control_sd21_ini.ckpt'
batch_size = 5
logger_freq = 10_000
learning_rate = 1e-5
sd_locked = True
only_mid_control = False


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('models/cldm_v21.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# Misc
checkpoint_callback_top_model = ModelCheckpoint(filename='top_model',
                                                save_top_k=1,
                                                monitor='train/loss_epoch',
                                                mode='min', )

dataset = BlockWorld('/export/data/vislearn/rother_subgroup/rother_datasets/SimpleCubesfull/',
                     ['Depth'])
dataloader = DataLoader(dataset, num_workers=5, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger, checkpoint_callback_top_model])


# Train!
trainer.fit(model, dataloader)
