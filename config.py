from other_imports import *

class BaseConfig(object):

    def __init__(self):
        self.args = self.ConfigParser()
        self.seed = 42
        self.image_size = 150
        self.batch_size = 90
        self.lr = 1e-4
        self.weight_decay = 1e-6
        self.fold_num = 5
        self.num_workers = 4
        self.patches = self.args.patches
        self.optim = 'Adam'
        self.sched = 'step'
        self.step_size = 45
        self.gamma = 0.1
        self.T_0 = 75
        self.min_lr = 1e-6
        self.n_accumulate = 1
        self.verbose_step = 1

    def ConfigParser(self):

        parser = argparse.ArgumentParser(description='Process hyper-parameters')
        parser.add_argument('--backbone', type=str, default="Vgg", help='Backbone CNN')
        parser.add_argument('--data', type=str, default="IITD", help='Dataset')
        parser.add_argument('--palm_train', type=str, default="left", help='Palm train')
        parser.add_argument('--palm_test', type=str, default="right", help='Palm test')

        parser.add_argument('--n_epochs', type=int, default=80, help='Number of epochs')
        parser.add_argument('--num_trainable', type=int, default=10, help='Number of trainable layers') 
        parser.add_argument('--metric_head', type=str, default='arc_margin', help='Metric learning head')

        parser.add_argument('--patches', nargs="+", type=int, default=[75, 1, 0, 30], help='patches')
        parser.add_argument('--lr_centers', type=int, default=0.5, help='Learning rate of the centers')
        parser.add_argument('--alpha', type=float, default=0.001, help='The alpha parameter')
        parser.add_argument('--save_path', type=str, default="saved_models", help='Save path')
        parser.add_argument('--model_type', type=str, default="VGG_16", help='Dual-path')

        args = parser.parse_args()

        return args

