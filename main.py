from model import *
from utils import *
from dataset import *
from other_imports import *
from losses import *
from config import *
from trainers.train_centerloss import *
from run.trainer import *

if __name__ == '__main__':
  conf = BaseConfig()
  args = conf.ConfigParser()

  print("\n------------ Data config ------------")
  print('Backbone CNN: ', args.backbone)
  print('Data: ', args.data)
  print('Palm train: ', args.palm_train)
  print('Palm test: ', args.palm_test) 

  print("\n------------ Arch config ------------")
  print('Number of trainable layers: ', args.num_trainable)
  print('Metric learning head: ', args.metric_head)
  print('Learning rate of the centers: ', args.lr_centers)  
  print('Alpha paramter: ', args.alpha)
  print('Number of epochs: ', args.n_epochs)
  print('Patch config: ', args.patches)
  print('Save path: ', args.save_path)
  print("------------ ---------- ------------")


  n_classes = 230 if args.data == 'IITD' else(312 if args.data == 'CASIA' else None)
  n_palms = 2
  
  device = get_default_device()
                      
  df = create_dataframe(args.data, n_classes, n_palms, args.palm_train, False)  
  df_w = create_dataframe(args.data, n_classes, n_palms, args.palm_train, True)
  
  base = Base_Palmprint_Trainer(args.backbone,
                                n_classes, 
                                args.patches,
                                args.n_epochs,
                                args.num_trainable,
                                args.metric_head,
                                args.lr_centers,
                                args.alpha,
                                device)


  base.run_fold(df,
                df_w = None,
                mode = 'train',
                save_path = args.save_path)
