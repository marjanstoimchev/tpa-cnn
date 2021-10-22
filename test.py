from model import *
from utils import *
from dataset import *
from other_imports import *
from config import *

class Evaluation_Protocol:
    
    def __init__(self, data, mode, n_classes, n_samples):
        
        self.data = data
        self.mode = mode
        self.n_classes = n_classes
        self.n_samples = n_samples
        
    def CreateGenImp(self, Data):
    
        tests = []
        probes = []
        
        genuines = []
        impostors = []
        
        Targets = []
        
            
        n_probe = np.shape(Data)[1]
    
        Data = Data[0:self.n_classes*n_probe]
        Data = Data.reshape(self.n_classes,n_probe)
    
        for i in range(self.n_classes):
            tests.append(Data[i][0])
            for k in range(n_probe):
                probes.append(Data[i][k])
                
        tests = np.array(tests)
        probes = np.array(probes).reshape(self.n_classes,n_probe)
        
        for i in range(self.n_classes): #tqdm
            for j in range(self.n_classes):
                for k in range(n_probe):
                    if i==j:   
                        genuines.append([tests[i], probes[j][k]])
                        Targets.append(1)
                    else:
                        impostors.append([tests[i], probes[j][k]])
                        Targets.append(0)
        
        genuines = np.array(genuines)   
        impostors = np.array(impostors)  
        Targets = np.array(Targets)       
     
        N = len(genuines) + len(impostors)
        FinalData = [' ']*N
         
        genuine_indices = np.where(Targets == 1)[0]
        impostor_indices = np.where(Targets == 0)[0]
             
        for (g,index_g) in enumerate(genuine_indices):
            left_g, right_g = genuines[g]
            FinalData[index_g] = left_g, right_g
         
        for (i,index_i) in enumerate(impostor_indices):
            left_i, right_i = impostors[i]
            FinalData[index_i] = left_i, right_i
    
        return np.array(FinalData), Targets
    
    def generate_scores(self, pair_left, pair_right):
        denom = np.linalg.norm(pair_left-pair_right)
        similarity = 1 / (denom + 1e-6)
        return similarity
    
    def norm2(self, x):
        return np.sqrt(np.sum(np.square(x)) + 1e-6)

    def cosDistance(self, x, y):
        denom = self.norm2(x) * self.norm2(y)
        return np.dot(x, y) / (denom + 1e-6)  # T.dot(x, y) / denom


    def CalculateScores(self, ImagePaths, Features, ScoreFunction):

        n_samples = self.n_samples
        Images = np.array(ImagePaths).reshape(self.n_classes, n_samples)
            
        Data, Targets = self.CreateGenImp(Images)
        
        d = dict()
        for i,f in enumerate(Features):  
            d[ImagePaths[i]] = Features[i]
    
        Scores = []
        
        pbar = tqdm(enumerate(Data), total=len(Data),
            desc=f"Genuine-impostor scores generation (mode = {self.mode}):", position=0, leave=True)
        
        for i, (data) in pbar: 
            
            left = data[0]
            right = data[1]
        
            pair_left = d[left]
            pair_right = d[right]
        
            Scores.append(ScoreFunction(pair_left, pair_right))
        
        return Targets, Scores
   
#--------------------------------------------------------------------------------------------------#

class EvaluateModel:
    
    def __init__(self, dir_name = None,
                       start_evaluate_from = 1,
                       patches = [],
                       evaluate_on = ('IITD', 'right'),
                       load_model_from = ('IITD', 'left')):
 
        self.load_model_from = load_model_from
        self.evaluate_on = evaluate_on
        self.patches = patches
        
        self.start_evaluate_from = start_evaluate_from
        self.dir_name = dir_name
                
        self.n_classes_train = 230 if self.load_model_from[0] == 'IITD' else(312 if self.load_model_from[0] == 'CASIA' else None)
        
        
        n_palms = 2
        
        self.df = create_dataframe(self.evaluate_on[0], self.n_classes_train, n_palms, self.evaluate_on[1], False)

        self.n_patches = calculate_n_patches(self.df, patches = self.patches)

        self.n_classes_test = len(self.df.label.unique())
        
        itemList =  os.listdir(self.dir_name)
        self.paths = [self.dir_name + item for item in itemList]
        
    def feature_extraction(self, model):  
            
        device = get_default_device()
        test_ds = PalmprintDatasetImgAug(self.df, 
                        transform = transforms.Compose([transforms.ToTensor()]),
                        augment = False, patches = self.patches)
                        
        test_loader =  DataLoader(test_ds, 
                               batch_size=opt.batch_size,
                               shuffle=False, num_workers=opt.num_workers, 
                               drop_last=False,
                               pin_memory = True)
    

        
        test_loader = DeviceDataLoader(test_loader, device)
        features_h, forward_features_p, features_c = [], [], []
        labels = []
        
        for step, (data, y) in enumerate(test_loader):
            
            imgs, patch = data
            
            #features_h += [model.forward_features_holistic(imgs).cpu().detach().numpy()]
            #forward_features_p += [model.forward_features_patches(patch).cpu().detach().numpy()]
            #features_c += [model.forward_combined_features_fc2(imgs, patch).cpu().detach().numpy()]
            features_c += [model.forward(imgs, patch, y)[1].cpu().detach().numpy()]
            
            labels += [y.cpu().detach().numpy()]
            
        #features_h = np.array(list(itertools.chain.from_iterable(features_h)))
        #features_p = np.array(list(itertools.chain.from_iterable(forward_features_p)))
        features_c = np.array(list(itertools.chain.from_iterable(features_c)))
        
        features = {
         #   "holistic": features_h,
          #   "patch"   : features_p,
             "combined": features_c
             }
        
        labels = np.array(list(itertools.chain.from_iterable(labels)))
        
        return features, labels

    
    def load_models(self, path, model, device):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint)
        for parameter in model.parameters():
            parameter.requires_grad = False
        model.eval()
        model = to_device(model, device)
        return model
    
    def minimum_per_class(self):
               
        data_test, palm_test = self.evaluate_on
        if data_test == 'IITD':
            min_per_class = 5
        elif data_test == 'CASIA':
            if palm_test == 'left':
                min_per_class = 6
            elif palm_test == 'right':
                min_per_class = 7
        return min_per_class

    
    def standardize_features(self, features):  
        scaler = StandardScaler()
        scaler.fit(features)
        sc_features = scaler.transform(features)
        return features
    
    def calculate_eer(self, architecture, device):

        sorted_paths = sorted(self.paths, key=lambda x: int(x.split("_")[-1].split(".")[0]))
        new_paths = sorted_paths[self.start_evaluate_from:]
    
        pbar = tqdm(enumerate(new_paths), total=len(new_paths),
                    desc="\nTesting model/s", position=0, leave=True)
        
        data_test, _ = self.evaluate_on
        
        final_eers = []
        
        for i, (path) in pbar: 
            
            
            d = {
                "EER": {"holistic": None, "patch": None, "combined": None},
                "AUC": {"holistic": None, "patch": None, "combined": None},
                    }
                
            EERs = pd.DataFrame(d)
            EERs.index.name = "(%)"

            model = self.load_models(path, architecture, device)
            min_per_class = self.minimum_per_class()
            features, labels = self.feature_extraction(model)
            
            
            for name,feature in features.items():
                #sc_features = self.standardize_features(feature)
                sc_features = feature
                protocol = Evaluation_Protocol(data_test, name, self.n_classes_test, min_per_class)
                image_paths = list(self.df.image_id)
                targets, scores = protocol.CalculateScores(image_paths, 
                                                           sc_features, 
                                                           protocol.cosDistance)
            
            
            
                
                fpr, tpr, thresholds = roc_curve(targets, scores)
                fnr = 1-tpr
                EER = 100*fpr[np.nanargmin(np.absolute((fnr - fpr)))]
                EERs['EER'][name] = round(EER,4)
                
                AUC = 100*auc(fpr,tpr)
                EERs['AUC'][name] = round(AUC,4)
                
            final_eers.append(list(EERs['EER'].values))
            
            print(
                  f'\n\nInitial feature shape: {features[name].shape}\n'
                  f'Reordered features shape: {sc_features.shape}\n\n' # sc_features
                  f'{EERs}\n\n',
                  f'Path: {path}\n',
                  f'Minimum samples per class: {min_per_class}\n'
                  f'Reduced classes: {self.n_classes_test}\n'.format(
                  features['combined'], sc_features, EERs, path, min_per_class, self.n_classes_test)
                  )

        return final_eers

def test():
    
    conf = BaseConfig()

    args = conf.ConfigParser()
    evaluate_on = [args.data, args.palm_test]
    load_model_from = [args.data, args.palm_train]
    load_path = input('\nEnter directory: ')

    print('\n-------------------------------------')
    print('Directory name: ', load_path)
    print('Evaluate model on: ', evaluate_on)
    print('Trained model on: ', load_model_from) 
    print('Model type: ', args.model_type)
    print('-------------------------------------')
    
    
    n_models = len(os.listdir(load_path))
    n_last = int(input("\nEnter how many models: "))-1
    
    
    print('\nTotal Number of models: ', n_models)
    print(f'Evaluationg on the last {n_last} models')
    
    ev = EvaluateModel(dir_name = load_path + "/",
                       start_evaluate_from = 0, # n_models - n_last
                       patches = args.patches,
                       evaluate_on = evaluate_on,
                       load_model_from = load_model_from)
                        
                
    pretrained = False                
    warping = False

    print('\nNumber of patches: ', ev.n_patches)
    model = PatchAttentionModel(ev.n_classes_train, ev.n_patches, pretrained, metric_head = 'arc_margin')

    device = get_default_device()   
    EERs = ev.calculate_eer(model, device) 


if __name__ == '__main__':
    test()
