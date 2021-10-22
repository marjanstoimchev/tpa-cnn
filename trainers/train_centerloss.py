from other_imports import *
from config import *

opt = BaseConfig()
scaler = GradScaler()

# Chose normal since the TPS warped images are not available at this time #
def fusion_method(method, model, imgs, patches, labels):
  
  # This method combines the normal augmentations with TPS warped images as additional augmentations #
  if method == 'concat':
    imgs = imgs.flatten(start_dim=0, end_dim=1)
    patches = patches.flatten(start_dim=0, end_dim=1)
    labels = labels.repeat_interleave(2)
    
    logits, features = model(imgs, patches, labels)

  elif method == 'mean_features':
    
    output_metric, features = model(imgs[:,0], patches[:,0], labels)
    output_metric_w, features_w = model(imgs[:,1], patches[:,1], labels)
    
    features = torch.stack((features, features_w), dim = 0).mean(0)
    logits = torch.stack((output_metric, output_metric_w), dim = 0).mean(0)
 
  elif method == 'normal':

    logits, features = model(imgs, patches, labels)


  return features, logits, labels
  

def train_one_epoch(epoch, model, main_loss, center_loss, 
                    optimizer_main, optimizer_center,                             
                    train_loader, train_ds, device, scheduler = None, 
                    schd_batch_update=False, alpha = None):

    model.train()

    t = time.time()
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    
    
    image_preds_all = []
    image_targets_all = []
    train_loss = []
    
    running_loss = 0
    sample_num = 0
    
    for step, (data, labels) in pbar:
        imgs, patches = data
        image_labels = labels.long()

        with autocast():
            
            features, output_metric, image_labels = fusion_method('normal', model, imgs, patches, image_labels)
              
            # predictions after softmax #
            predictions = output_metric.argmax(1).detach()
                                  
            image_preds_all   += [predictions]
            image_targets_all += [image_labels]
            
            # calculate loss #
            loss = main_loss(output_metric, image_labels) # image_labels
            loss = center_loss(features.float(), image_labels) * alpha + loss # image_labels
            
            # calculate running loss # 
            running_loss   += loss.item()*image_labels.shape[0] # image_labels 
            sample_num += image_labels.shape[0] # image_labels 
               
            scaler.scale(loss).backward()

            if ((step + 1) %  opt.n_accumulate == 0) or ((step + 1) == len(train_loader)):
                
                scaler.step(optimizer_main)
                
                if alpha == 0:
                    scaler.update()
                    optimizer_main.zero_grad() 
                    
                else:         
                    for param in center_loss.parameters():
                        param.grad.data *= (1. / alpha)
                    scaler.step(optimizer_center)

                    scaler.update()
                    optimizer_main.zero_grad() 
                    optimizer_center.zero_grad()
                
                if scheduler is not None and schd_batch_update:
                    scheduler.step()
                

            if ((step + 1) % opt.verbose_step == 0) or ((step + 1) == len(train_loader)):
                description = f'epoch {epoch} loss: {running_loss/sample_num:.4f}'
                pbar.set_description(description)
        
            
    image_preds_all = torch.cat(image_preds_all).cpu().numpy()
    image_targets_all = torch.cat(image_targets_all).cpu().numpy()
    acc = (image_preds_all == image_targets_all).mean() 
    
    epoch_loss = running_loss / len(image_preds_all)
    
    if scheduler is not None and not schd_batch_update:
        scheduler.step()


    return acc, epoch_loss
        
def valid_one_epoch(epoch, model, main_loss, center_loss,
                    val_loader, val_ds, device, scheduler=None, 
                    schd_loss_update=False, alpha = None):
    model.eval()

    t = time.time()
    running_loss   = 0
    sample_num = 0
    
    image_preds_all   = []
    image_targets_all = []
    pbar = tqdm(enumerate(val_loader), total=len(val_loader))
    
    for step, (data, labels) in pbar:
        imgs, patches = data
        image_labels = labels.long()
        
        # logits #
        output_metric, features = model(imgs, patches, image_labels)
        
        # predictions after softmax #
        predictions = output_metric.argmax(1).detach()
        
        image_preds_all.append(predictions)
        image_targets_all.append(image_labels)
        
        # calculate loss #
        loss = main_loss(output_metric, image_labels)
        loss = center_loss(features, image_labels) * alpha + loss
        
        # calculate running loss #
        running_loss   += loss.item()*image_labels.shape[0]
        sample_num += image_labels.shape[0]  
                
        if ((step + 1) % opt.verbose_step == 0) or ((step + 1) == len(val_loader)):
            description = f'epoch {epoch} loss: {running_loss/sample_num:.4f}'
            pbar.set_description(description)
    
    image_preds_all = torch.cat(image_preds_all).cpu().numpy()
    image_targets_all = torch.cat(image_targets_all).cpu().numpy()
    acc = (image_preds_all==image_targets_all).mean()
    
    
    epoch_loss = running_loss / len(image_preds_all)
    
    
    if scheduler is not None:
        if schd_loss_update:
            scheduler.step(loss_sum/sample_num)
        else:
            scheduler.step()

    return acc, epoch_loss
