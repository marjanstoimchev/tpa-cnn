from other_imports import *
from dataset import *
from config import *

def select_path(data, warping):
    
    if warping:
        
        if data == 'IITD':
            path = ['../datasets/IITD_ROI/Segmented/warp_iitd_left',
                    '../datasets/IITD_ROI/Segmented/warp_iitd_right']
                    
        elif data == 'CASIA':
            path = '../datasets/CASIA_ROI_warp/'    
    
    else:
        
        if data == 'IITD':
            path = ['../datasets/IITD_ROI/Segmented/Left',
                    '../datasets/IITD_ROI/Segmented/Right']
                    
        elif data == 'CASIA':
            path = '../datasets/CASIA_ROI/'
            
    return path

def create_dataframe(data, n_classes, n_palms, palm_train, warp):
    
    Path = select_path(data, warp)
    
    if data == 'IITD':
        
        data_path_left = os.path.join(Path[0],'*bmp')
        data_path_right = os.path.join(Path[1],'*bmp')
        
        images_left = np.sort(glob.glob(data_path_left))
        images_right = np.sort(glob.glob(data_path_right))
        
        paths = [images_left, images_right]
        
        samples_per_class = []
        groups = []    
        # extracting specific sub-strings from the paths #
        
        for path in paths: 
            for names in path:   
                samples_per_class.append(''.join(filter(lambda d: d.isdigit(), names))[0:-1])
            groups.append(sorted([[value for value in path if key in value] for key in set(samples_per_class)]))
        
        Max = [[0 for x in range(len(groups))] for y in range(n_classes)]
        
        for group_number,group in enumerate(groups):
            for i, sub_group in enumerate(group):
                Max[i][group_number] = np.max(len(sub_group))
        
        minimum_samples = np.min(Max, axis = 0)
        
        new_groups = np.array(copy.deepcopy(groups),  dtype="object")
        targets = [[0 for x in range(minimum_samples[0])] for y in range(np.shape(new_groups)[1])]
        image_paths = [ [ [0 for x in range(n_palms)] for x in range(minimum_samples[1])] for y in range(np.shape(new_groups)[1]) ]
        images = [ [ [0 for x in range(n_palms)] for x in range(minimum_samples[1])] for y in range(np.shape(new_groups)[1]) ]
        

        for p in range(np.shape(new_groups)[0]):
            for i in range(np.shape(new_groups)[1]):
                for j in range(minimum_samples[0]):
                    image_paths[i][j][p] = new_groups[p][i][j]         
                    targets[i][j] = i
                
    
        image_paths = np.array(image_paths)
        targets = list(np.array(targets).flatten())
        
        final_left = image_paths[:,:,0].flatten()
        final_right = image_paths[:,:,1].flatten()
                
        df_left = pd.DataFrame({"image_id": final_left, "label": targets})
        df_right = pd.DataFrame({"image_id": final_right, "label": targets})
        
    elif data == 'CASIA':
            
        data = os.path.join(Path,'*jpg')
        digits = []
        image_list = []
    
        for filename in tqdm(glob.glob(data)):
            
            image_list.append(filename)
            d = ''.join(filter(lambda d: d.isdigit(), filename))[0:-2]
            digits.append(d)
        
        groups = sorted([[value for value in image_list if key in value] for key in set(digits)])
        
        strings = ['_l_','_r_']
        palms = []
        
        for i in range(len(groups)):
            palms.append(sorted([[value for value in groups[i] if key in value] for key in set(strings)]))
            
        left_palms = []
        right_palms = []
        
        for i,p in enumerate(palms):
            left_palms.append(p[0])
            right_palms.append(p[1])
            
        #% reduce missing
        indices = 23, 29, 43, 109, 151, 160, 164, 246
        
        reduced_list_left = [i for j, i in enumerate(left_palms) if j not in indices]
        reduced_list_right = [i for j, i in enumerate(right_palms) if j not in indices]
        
        left_images = list(chain.from_iterable(reduced_list_left))
        right_images = list(chain.from_iterable(reduced_list_right))
        
        targets_left = []
        targets_right = []
        
        digits_left = []
        digits_right = []
        
        image_list_left = []
        image_list_right = []
        
        for r_l in left_images:
            image_list_left.append(r_l)
            d_l = ''.join(filter(lambda d_l: d_l.isdigit(), r_l))[1:-2]
            digits_left.append(d_l)
            targets_left.append(int(d_l) - 1)
            
        
        for r_r in right_images:
            image_list_right.append(r_r)
            d_r = ''.join(filter(lambda d_r: d_r.isdigit(), r_r))[1:-2]
            digits_right.append(d_r)
            targets_right.append(int(d_r) - 1)
            
        groups_left = sorted([[value for value in image_list_left if key in value] for key in set(digits_left)])
        groups_right = sorted([[value for value in image_list_right if key in value] for key in set(digits_right)])
        grouped_targets_left = [list(j) for i, j in groupby(targets_left)]
        grouped_targets_right = [list(j) for i, j in groupby(targets_right)]
        
        # balance data #
        
        count_left = []
        count_right = []
        
        for (i,palm_left), (j, palm_right) in zip(enumerate(groups_left), enumerate(groups_right)):
            count_left.append(len(palm_left))
            count_right.append(len(palm_right)) 
        
        min_left = min(count_left)
        min_right = min(count_right)
        
        final_left = []
        final_targets_left = []
        
        for i,reduced_left in enumerate(groups_left):
            final_left.append(reduced_left[0:min_left])
            final_targets_left.append(grouped_targets_left[i][0:min_left])
            
        final_right = []
        final_targets_right = []
        
        for i,reduced_right in enumerate(groups_right):
            final_right.append(reduced_right[0:min_right])
            final_targets_right.append(grouped_targets_right[i][0:min_right])
                        

        final_left = [item for sublist in final_left for item in sublist]
        final_right = [item for sublist in final_right for item in sublist]

        final_targets_left = [item for sublist in final_targets_left for item in sublist]
        final_targets_right = [item for sublist in final_targets_right for item in sublist]

        print('right: ', len(final_targets_right))
        df_left = pd.DataFrame({"image_id": final_left, "label": final_targets_left})
        df_right = pd.DataFrame({"image_id": final_right, "label": final_targets_right})
    
    if palm_train == 'left':
        df = df_left
    elif palm_train == 'right':
        df = df_right
    
    return df


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

class DataLoaders(BaseConfig):
    
    def __init__(self, device):
        super(DataLoaders, self).__init__()
        self.device = device
    
    def get_loaders_imgaug(self, train_df, val_df, train_df_w = None, val_df_w = None):
        
        
        train_ds = PalmprintDatasetImgAug(train_df,
                                          df_w = train_df_w,
                                          transform = transforms.Compose([transforms.ToTensor()]), 
                                          augment=True,
                                          patches = self.patches)
        
        
        val_ds = PalmprintDatasetImgAug(val_df,
                                        df_w = None, # val_df_w
                                        transform = transforms.Compose([transforms.ToTensor()]), 
                                        augment=False, 
                                        patches = self.patches)
                
                               
    
        train_loader =  DataLoader(train_ds, 
                                   batch_size=self.batch_size, 
                                   shuffle=True,
                                   num_workers=self.num_workers, 
                                   pin_memory = True)
        
        val_loader =    DataLoader(val_ds, 
                                   batch_size=self.batch_size, 
                                   shuffle=False, 
                                   num_workers=self.num_workers, 
                                   pin_memory = True)
    
        dataloaders = {"train": DeviceDataLoader(train_loader, self.device),
                        "val": DeviceDataLoader(val_loader, self.device)}            
    
        dataset_sizes = {"train": len(train_ds), 
                         "val":   len(val_ds)}  
        
       
        return dataloaders['train'], dataloaders['val'] 



def calculate_n_patches(df, patches = [75, 1, 0, 20]):

  dataset = PalmprintDatasetImgAug(df, 
                                    transform = transforms.Compose([transforms.ToTensor()]),
                                    augment = False, patches = patches)      

  test_loader =  DataLoader(dataset, 
                      batch_size=1,
                      shuffle=False,
                      num_workers = 0, 
                      drop_last=False,
                      pin_memory = True)                                       
  (img, patch), label = next(iter(test_loader))
  n_patches = patch.shape[1]

  return n_patches


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters", "Requires grad"])

    trainable = 0
    frozen = 0

    for name, parameter in model.named_parameters():

        param = parameter.numel()
        table.add_row([name, param, parameter.requires_grad])

        if parameter.requires_grad: 
            trainable+=param
        else:
            frozen+=param

    percent_frozen = round(100*frozen/(trainable+frozen),4)
    percent_trainable = round(100*trainable/(trainable+frozen),4)

    print(table)
    print('============================================')
    print("Total Params: {}".format(trainable+frozen))
    print("Total Trainable Params: {}".format(trainable))
    print("Total Non-Trainable Params: {}".format(frozen))
    print("Percent frozen: {}".format(percent_frozen), "%")
    print("Percent trainable: {}".format(percent_trainable), "%")
    print('=============================================')

def freeze_model(model, num_layer, freeze_bn = False):
    ct = 0
    for param in model.parameters():
        ct += 1
        if ct <= num_layer - 2:
            param.requires_grad = False
    
    if freeze_bn:
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                if hasattr(module, 'weight'):
                    module.weight.requires_grad_(False)
                if hasattr(module, 'bias'):
                    module.bias.requires_grad_(False)
                module.eval()

    return model 

def unfreeze_model(model):
    for param in model.parameters():
        param.requires_grad = True

    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            if hasattr(module, 'weight'):
                module.weight.requires_grad_(True)
            if hasattr(module, 'bias'):
                module.bias.requires_grad_(True)
            module.train()

    return model


def seed_all(seed: int = 1930):
    print("Using Seed Number {}".format(seed))

    os.environ["PYTHONHASHSEED"] = str(
        seed)  # set PYTHONHASHSEED env var at fixed value
    #torch.manual_seed(seed) #
    #torch.cuda.manual_seed_all(seed) #
    #torch.cuda.manual_seed(seed)  # pytorch (both CPU and CUDA) #
    np.random.seed(seed)  # for numpy pseudo-random generator
    random.seed(seed)  # set fixed value for python built-in pseudo-random generator
    #torch.set_num_threads(1)
    #torch.cuda.set_device(2)
    #torch.backends.cudnn.deterministic = True #
    #torch.backends.cudnn.benchmark = False # False #
    #torch.backends.cudnn.enabled = False #



def seed_worker(_worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)



