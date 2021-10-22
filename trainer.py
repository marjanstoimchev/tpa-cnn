from model import *
from trainers.train_centerloss import *
from utils import *
from dataset import *
from other_imports import *

class Base_Palmprint_Trainer(BaseConfig):
    
    def __init__(self, backbone, n_classes, patches, n_epochs, num_trainable, metric_head, lr_centers, alpha, device):
        super(Base_Palmprint_Trainer, self).__init__()
        self.backbone = backbone
        self.n_classes = n_classes
        self.patches = patches
        self.n_epochs = n_epochs
        self.num_trainable = num_trainable
        self.metric_head = metric_head
        self.lr_centers = lr_centers
        self.alpha = alpha
        self.device = device

    def training_loop(self, model, fold, 
                      train_dl, train_df, 
                      val_dl, val_df, 
                      main_loss,
                      center_loss,
                      optimizer_main,
                      optimizer_center,
                      scheduler,
                      save_path):
        

        early_stopping = 40
        not_improved_cnt = 0
        best_acc = 0
        best_epoch = 0
        best_loss = 99999
        
        # - - - - - - - Main loop - - - - - - - - -#

        for epoch in range(self.n_epochs):

            print(time.ctime(), 'Fold', '[' + f'{fold+1}' + ']', 'Epoch :', epoch+1, "/", self.n_epochs)



            acc_train, train_loss = train_one_epoch(epoch, model, 
                                                    main_loss,
                                                    center_loss, 
                                                    optimizer_main, 
                                                    optimizer_center,
                                                    train_dl, train_df, 
                                                    self.device, 
                                                    scheduler = scheduler, 
                                                    schd_batch_update=False,
                                                    alpha = self.alpha)

            with torch.no_grad():

                acc_valid, val_loss = valid_one_epoch(epoch, model,
                                                      main_loss, 
                                                      center_loss,
                                                      val_dl, 
                                                      val_df, 
                                                      self.device,
                                                      scheduler=None, 
                                                      schd_loss_update=False, 
                                                      alpha = self.alpha)

                if val_loss < best_loss:
                    
                    best_model = copy.deepcopy(model)
                    best_loss = val_loss
                    #best_acc = acc_valid
                    best_epoch = epoch
                    not_improved_cnt = 0
                    torch.save(best_model.module.state_dict(), f'{save_path}' + f'//model_fold_{fold+1}_epoch_{epoch+1}.pth')
                                                                
                elif early_stopping == not_improved_cnt:
                    print("Met early stopping.")
                    break
                else:
                    not_improved_cnt += 1
                
                print(
                      f'\n\nTrain acc: {acc_train:.4f}\n'
                      f'Train loss: {train_loss:.4f}\n'
                      f'Val acc: {acc_valid:.4f}\n'
                      f'Val loss: {val_loss:.4f}\n'.format(acc_train, train_loss, acc_valid, val_loss)
                      )
                    

        torch.save(best_model.module.state_dict(), f'{save_path}' + f'//Best_model_fold_{fold+1}_epoch_{epoch+1}.pth')
        del model, optimizer_main, optimizer_center
        gc.collect()
        torch.cuda.empty_cache() 


    def percentage_layers_frozen(self, percent, model):
        total_number_of_layers = len([i for i, (name, module) in enumerate(model.named_modules())])
        return round((percent * total_number_of_layers) / 100.0)

    def run_fold(self, df, df_w = None, mode = 'visualize', save_path = None):
        
        #seed_all(self.seed)
        n_patches = calculate_n_patches(df, patches = self.patches)
        print('\nThe total number of patches is: ', n_patches)

        if save_path is not None:
            if not os.path.exists(save_path):
                os.makedirs(save_path)  
            
        folds = df.copy()
        
        skf = StratifiedKFold(n_splits=self.fold_num, shuffle=True, random_state=self.seed)
        skf.get_n_splits(folds, folds['label'])
        
        for fold_num, (train_split, valid_split) in enumerate(skf.split(folds, folds['label'])):
        
            print(f"========== fold: {fold_num} training ==========")
        
            train_df = folds.iloc[train_split].reset_index(drop=True)
            val_df = folds.iloc[valid_split].reset_index(drop=True)
    
            
            if df_w is not None:
                
                warping = True
                folds_w = df_w.copy()
                train_df_w = folds_w.iloc[train_split].reset_index(drop=True)
                val_df_w = folds_w.iloc[valid_split].reset_index(drop=True)
        
            else:
                
                warping = False
                train_df_w = None
                val_df_w = None
                
            dataloaders = DataLoaders(self.device)
            train_dl, val_dl = dataloaders.get_loaders_imgaug(train_df, 
                                                              val_df, 
                                                              train_df_w = train_df_w,
                                                              val_df_w = val_df_w)
            
            
            if mode == 'train':
                    
                # - - - - - - Model definition - - - - - - #

                pretrained = True                      
                model = PatchAttentionModel(self.n_classes, n_patches, pretrained, metric_head = 'arc_margin', kernel = 2)
                num_frozen = self.percentage_layers_frozen(self.num_trainable, model)
   
                model.backbone_holistic = freeze_model(model.backbone_holistic, num_frozen)
                model.backbone_patch = freeze_model(model.backbone_patch, num_frozen) 
                
                model = to_device(model, self.device)
                total_params = count_parameters(model)
                
                my_list = ['att_pool.att.0.weight', 'att_pool.att.0.bias', 
                            'att_pool.att.2.weight', 'att_pool.att.2.bias']

                params_w = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] in my_list, model.named_parameters()))))
                base_params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] not in my_list, model.named_parameters()))))
                base_p = filter(lambda p: p.requires_grad, base_params)   
                
                main_loss = nn.CrossEntropyLoss()
                center_loss = CenterLoss(num_classes=self.n_classes, 
                                         feat_dim=1024, use_gpu=True)

                
                optimizer_main = optim.Adam([{'params': base_p}, 
                                             {'params': params_w, 'lr': 1e-2}], 
                                                             lr=self.lr)
                optimizer_center = optim.Adam(center_loss.parameters(), lr=self.lr_centers)
                
                model = nn.DataParallel(model)
                exp_lr_scheduler= lr_scheduler.StepLR(optimizer_main, step_size=self.step_size, gamma=self.gamma)

                self.training_loop(model, 
                                   fold_num,
                                   train_dl, train_df, 
                                   val_dl, val_df, 
                                   main_loss,
                                   center_loss,
                                   optimizer_main, 
                                   optimizer_center,
                                   exp_lr_scheduler,
                                   save_path = save_path)
                
                
            if fold_num == 0: # getting only a single fold
                break        
