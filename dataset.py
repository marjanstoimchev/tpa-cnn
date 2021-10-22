from other_imports import *

# For IITD dataset
aug = iaa.Sequential([
        
        iaa.Sometimes(0.5, iaa.Affine(rotate=(-10,10), fit_output = False,  mode=["edge"])),
        iaa.Sometimes(0.5, iaa.Crop(percent=(0.05, 0.3),  keep_size=True)),
        iaa.Sometimes(0.5, iaa.Multiply((1, 1.15))),
        
        iaa.OneOf([
            iaa.OneOf([
            iaa.LinearContrast((0.4, 1.6), per_channel=False),
            iaa.BlendAlpha((0.0, 1.0), iaa.HistogramEqualization()),
            iaa.CLAHE(tile_grid_size_px=((3, 100), [3, 5, 7])),
            ]),
    
            iaa.OneOf([
            iaa.Dropout((0.01, 0.1), per_channel=0.5),
            iaa.CoarseDropout((0.01, 0.05), size_px=(2, 20), per_channel=0.2 ),
            iaa.CoarseSaltAndPepper((0.01, 0.05), size_px=(2, 20))]),
           
            iaa.OneOf([
            iaa.GaussianBlur(sigma=(0.0, 3.0)),
            iaa.MotionBlur(k=15),
            iaa.AverageBlur(k=( (5, 11), (1, 3) ) ) ]), 
   
            ]),
        
], random_order=True)

# Uncomment this for CASIA dataset #
#aug = iaa.Sequential([
        
#        iaa.Sometimes(0.5, iaa.Affine(rotate=(-10,10), fit_output = False,  mode=["edge"])),
#        iaa.Sometimes(0.5, iaa.Crop(percent=(0.05, 0.3),  keep_size=True)),
#        iaa.Sometimes(0.5, iaa.Cutout(nb_iterations=(1, 5), fill_mode="gaussian", fill_per_channel=False, size=0.15)),
#        iaa.Sometimes(0.5, iaa.LinearContrast((0.6, 1.1), per_channel=False)),
        
        
#        iaa.OneOf([
#            iaa.LinearContrast((0.4, 1.6), per_channel=False),
#            iaa.BlendAlpha((0.0, 1.0), iaa.HistogramEqualization()),
#            iaa.CLAHE(tile_grid_size_px=((3, 100), [3, 5, 7])) ] ),
            
            
#            ], random_order=True)


        #iaa.OneOf([
        #    iaa.GaussianBlur(sigma=(0.0, 1.5)),
        #    iaa.MotionBlur(k=15),
        #    iaa.AverageBlur(k=( (5, 11), (1, 3) ) ) ]),

        #], random_order=True )

class PalmprintDatasetImgAug(torch.utils.data.Dataset):

    def __init__ (self, df, df_w = None, transform=None, augment = True, patches = []):

        self.df = df
        self.data = [(row['image_id'], row['label']) for _, row in self.df.iterrows()]
        
        self.df_w = df_w
        if self.df_w is not None:
            self.data_warp = [(row['image_id'], row['label']) for _, row in self.df_w.iterrows()]
        
        self.transform = transform
        self.augment = augment
        self.patches = patches        
        

    def check_if_casia_right(self):
        words = ["CASIA", "_r_"]
        one_path = self.df.values[0][0].split('/')
        exist = [b_any(word in x for x in one_path) for word in words]
        return sum(exist)

    def get_patches(self, input):
        """
        Extract patches out of input image according to self.patches [kh, d, p, s]
        :param input: input image Tensor [N, C, H, W]
        :return: output Tensor of patches [N, nPatches, C, kH, kh]
        """
        unfold = torch.nn.Unfold(kernel_size=(self.patches[0], self.patches[0]), dilation=self.patches[1],
                                 padding=self.patches[2], stride=self.patches[3])
        o1, o2 = math.floor(
            (input.shape[2] + 2 * unfold.padding - unfold.dilation * (unfold.kernel_size[0] - 1) - 1) / unfold.stride) + 1, \
                 math.floor(
                     (input.shape[3] + 2 * unfold.padding - unfold.dilation * (unfold.kernel_size[1] - 1) - 1) / unfold.stride) + 1
        output = unfold(input)
        output = output.view(input.shape[0], input.shape[1], self.patches[0], self.patches[0], o1 * o2)
        output = output.permute(0, 4, 1, 2, 3)

        return output.squeeze(0)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__ (self, index):
        
        image_path, label = self.data[index] 
        image = Image.open(image_path).convert('RGB')
        
        if self.check_if_casia_right() == 2:
            image = image.rotate(180)
            image = ImageOps.mirror(image)

        image = np.array(image)
        
        if self.augment:
            image = aug.augment_image(image)/255.0
        else:
            image = image/255.0

        image = image.astype('float32')
        
        if self.transform:
            image = self.transform(image)
        
        if len(self.patches) > 0:
            patches = self.get_patches(image.unsqueeze(0))
        else:
            patches = []
         
        # This is the TPS warping mode , which are concatenated to the basic augmentations #
        if self.df_w is not None:
            
            image_path_w, _ = self.data_warp[index]
            image_path_w = Image.open(image_path_w).convert('RGB')
            
            if self.check_if_casia_right() == 2:
                image_path_w = image_path_w.rotate(180)
                image_path_w = ImageOps.mirror(image_path_w)
                
            image_w = np.array(image_path_w)
            image_w = image_w/255.0
            image_w = image_w.astype('float32')
            
            if self.transform:
                image_w = self.transform(image_w)

            patches_w = self.get_patches(image_w.unsqueeze(0))
            patches = torch.stack((patches, patches_w), dim = 0)
                   
            image = torch.stack((image, image_w), dim = 0)

            
        return (image, patches), label 

