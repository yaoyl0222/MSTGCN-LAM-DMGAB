import sys
import gc
import random
import torch
import torch.optim as optim
from tqdm import tqdm, trange
from cvae_models import VAE
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from util import *




# Training settings
exc_path = sys.path[0]


def loss_fn(recon_x, x, mean, log_var):
    BCE = torch.nn.functional.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return (BCE + KLD) / x.size(0)

def slide_ts(input_tensor,window_size,stride):

    sliced_x = []
    for i in range(0, input_tensor.size(1) - window_size + 1, stride):
        sliced_x.append(input_tensor[:, i:i + window_size])

    result_tensor = torch.cat(sliced_x, dim=0)
    return result_tensor



def generated_generator(args, device, adj_scipy,features):

    x_list, c_list = [], []
    for i in trange(adj_scipy.shape[0]):
        neighbors_index = list(adj_scipy[i].nonzero()[1])  
        x = features[neighbors_index]  # （num_neighb,num_feature)
        c = torch.tile(features[i], (x.shape[0], 1))  # （num_neighb,num_feature)
        x_slided=slide_ts(x,2016,2016)
        c_slided=slide_ts(c,2016,2016)
        x_list.append(x_slided)  # x_list(num_nodes,num_neighb,num_feature)
        c_list.append(c_slided)  # c_list(num_nodes,num_neighb,num_feature)
    features_x = torch.vstack(x_list)
    features_c = torch.vstack(c_list)
    del x_list
    del c_list
    gc.collect() 

    cvae_dataset = TensorDataset(features_x, features_c)
   
    train_size = int(0.7 * len(cvae_dataset))
    val_size = len(cvae_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(cvae_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)  
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size) 
    # cvae_dataset_dataloader = DataLoader(cvae_dataset,  batch_size=args.batch_size)
    
    # Pretrain
    cvae = VAE(encoder_layer_sizes=[int(features_x.shape[1]), 256],
               latent_size=args.latent_size, 
               decoder_layer_sizes=[256,int(features_x.shape[1])],
               conditional=args.conditional, 
               conditional_size=int(features_x.shape[1]),

               ).to(device)
    cvae_optimizer = optim.Adam(cvae.parameters(), lr=args.pretrain_lr)
    # cvae_optimizer=tf.optimizers.Adagrad(learning_rate=args.pretrain_lr, initial_accumulator_value=0.1)

    # Pretrain
    for epoch in trange(100, desc='Run CVAE Train'):
        for batch, (x, c) in enumerate(tqdm(train_loader)):
            cvae.train() 

            x, c = x.to(device), c.to(device)
            if args.conditional:
                recon_x, mean, log_var, _ = cvae(x, c)
            else:
                recon_x, mean, log_var, _ = cvae(x)
            cvae_loss = loss_fn(recon_x, x, mean, log_var)

            cvae_optimizer.zero_grad()
            cvae_loss.backward()
            cvae_optimizer.step()
            # Evaluate the model on the validation set
        cvae.eval()
        with torch.no_grad():
            cvae_loss = 0
            total_samples = 0
            for val_batch_idx, (x, c) in enumerate(val_loader):
                x, c = x.to(device), c.to(device)
                if args.conditional:
                    recon_x, mean, log_var, _ = cvae(x, c)
                else:
                    recon_x, mean, log_var, _ = cvae(x)
                cvae_loss += loss_fn(recon_x, x, mean, log_var)
                total_samples+=1
            average_loss=cvae_loss/total_samples
            print(f'Epoch {epoch + 1}, Validation Loss: {average_loss:.4f}')

        cvae.train()
    return cvae
