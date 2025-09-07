import torch
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from torch import nn
from torch.nn.utils import clip_grad_value_
from torch.utils.data import DataLoader

from gan_layers import Generator, Discriminator
from tokenizer import Tokenizer
from evaluation import evaluation
import numpy as np
import pickle
import os
import random
import numpy
RDLogger.DisableLog('rdApp.*')
from tuning_logger import setup_logging, log_loss
import pandas as pd
import rdkit.Chem as Chem
from rdkit.Chem import PandasTools, QED, Descriptors, rdMolDescriptors
from mol_metrics import *


class FACGAN(nn.Module):

    def __init__(self, data, latent_dim_size=256, lr_critic=2e-4, device='cuda:0', RL_Flag='Pretrain'):

        super().__init__()

        self.device = device

        self.hidden_dim = latent_dim_size

        with open('../Results/Pretrain/2/models/tokenizer.pickle', 'rb') as f:
            self.tokenizer = pickle.load(f)
                            
        print('load tokenizer')

        if not os.path.exists('models'):
            os.makedirs('models')

        with open('models/tokenizer.pickle', 'wb') as f:
            pickle.dump(self.tokenizer, f)




        self.critic = Discriminator(
            latent_dim=latent_dim_size,
            vocab_size=self.tokenizer.vocab_size,
            start_token=self.tokenizer.start_token,
            bidirectional=True
        ).to(device)


        self.critic_optim = torch.optim.Adam(
            self.critic.parameters(), lr=lr_critic, weight_decay=1e-6)


        self.b = 0.  # baseline reward
        
        
        self.RL_Flag = RL_Flag

    def noise_generation(self, batch_size):
        """noise sampling

        Args:
            batch_size 

        Returns:
            torch.Tensor: [batch_size, self.latent_dim]
        """
        return torch.randn(batch_size, self.hidden_dim).to(self.device)

    def discriminator_loss(self, x, y):
        """Discriminator Loss

        Args:
            x (torch.LongTensor): input smiles [batch_size, max_length]
            y (torch.LongTensor): sequence label (zero: generated smiles, ones: real smiles)
                                  [batch_size, max_length]

        Returns:
            loss
        """

        y_prediction, mask = self.discriminator(x).values()

        loss = F.binary_cross_entropy(
            y_prediction, y, reduction='none') * mask

        loss = loss.sum() / mask.sum()

        return loss

    def critic_loss(self, x, values):
        """Critic Loss

        Args:
            x (torch.LongTensor): input smiles [batch_size, max_length]
            values (torch.LongTensor): sequence chemical property values 
                                  [batch_size, max_length]

        Returns:
            loss
        """

        y_prediction_values, mask = self.critic(x).values()

        loss = F.mse_loss(y_prediction_values, values, reduction='none') * mask
        '''
        print('y_prediction_values shape: ', y_prediction_values.shape)
        print('y_prediction_values: ', y_prediction_values)
        '''
        # Sum over the losses and divide by the sum of the mask to calculate the mean loss
        loss = loss.sum() / mask.sum()
        

        return loss


    def critic_pretrain(self, smiles, real_values):
        """One training step

        Args:
           smiles (torch.LongTensor): input smiles
           real_values: real data chemical property values
        """

        batch_size, len_real = smiles.size()
        #print(smiles.size())



        

        



        # create real and fake labels
        x_real = smiles.to(self.device)




        ########
        #Critic#
        ########


        # critic training 
        # critic initialization
        self.critic_optim.zero_grad()


            
        

        #print('reward df:  ', generated_reward_df)

        # Convert real values into a 2D tensor that aligns with the shape of smiles
        real_values = torch.tensor(real_values, dtype=torch.float32).view(-1, 1).expand(-1, smiles.shape[1])
        real_data_values = torch.tensor(real_values).to(self.device)
        
        


        # real critic loss
        real_loss_critic = self.critic_loss(x_real, real_data_values)

        
        # combining real and generated loss
        critic_loss =  real_loss_critic 
        
        critic_loss.backward()

        # clipping trainable parameters
        clip_grad_value_(self.critic.parameters(), 0.1)

        # parameters update
        self.critic_optim.step()



        return {'critic_loss': critic_loss.item()}




    def dataloader_creation(self, data, batch_size=1024, shuffle=True, num_workers=0):

        def b_tokenize(batch):

            smiles = [item['SMILES'] for item in batch]
            qed_values = [item['QED'] for item in batch]
            solubility_values = [item['Solubility'] for item in batch]
            sa_values = [item['SA'] for item in batch]

            tokenized_smiles = self.tokenizer.batch_tokenize(smiles)

            return (tokenized_smiles, qed_values, solubility_values, sa_values)

    

    
        return DataLoader(
            data,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=b_tokenize,
            num_workers=num_workers
        )    
    
    def train_n(self, training_data, train_loader, max_step=60000, evaluate_every=100):
        log_dir = 'logs'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        import logging
        with open('logs/training_log.txt', 'a') as f:
            logging.basicConfig(filename='logs/training_log.txt', level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

        best_model_step = 0
        best_model_critic_loss = 999999999999999


        iter_loader = iter(train_loader)
        
        
        for step in range(max_step):

            try:
                batch = next(iter_loader)
                tokenized_smiles, qed_values, solubility_values, sa_values = batch
            except:
                iter_loader = iter(train_loader)
                batch = next(iter_loader)
                tokenized_smiles, qed_values, solubility_values, sa_values = batch

            # critic pretrain model update
            if self.RL_Flag =='QED':
                critic_loss = self.critic_pretrain(tokenized_smiles, qed_values)
            
            if self.RL_Flag == 'Solubility':
                critic_loss = self.critic_pretrain(tokenized_smiles, solubility_values)

            if self.RL_Flag =='SA':
                critic_loss = self.critic_pretrain(tokenized_smiles, sa_values)

            critic_loss_value = critic_loss['critic_loss']
            print('critic_loss_value: ', critic_loss_value)

            logging.info("Critic Loss Value:      {:}".format(critic_loss_value))

            if (step % evaluate_every == 0) or (step == max_step - 1):
                logging.info(f'step {step}')

                self.eval()
                self.train()

                # Update best scores
                if critic_loss_value < best_model_critic_loss:
                    best_model_step = step
                    best_model_critic_loss = critic_loss_value
                    # Save the best critic model
                    torch.save(self.critic.state_dict(), 'models/best_critic.pth')
                    print('Saving the best critic')

                    logging.info('Saving the best generated data and models')
                    logging.info('\n')  

                                


            '''
            # Check stopping conditions
            if (validity >= 0.95 and uniqueness >= 0.90 and novelty >= 0.90):
                print(f'Early stopping at step {step} with validity {validity:.3f}, uniqueness {uniqueness:.3f}, and novelty {novelty:.3f}')
                # Save the best generator model
                torch.save(self.generator.state_dict(), 'models/best_generator.pth')
                # Save the best discriminator model
                torch.save(self.discriminator.state_dict(), 'models/best_discriminator.pth')
                print('Saving the best models')
                logging.info("Best Validity:     ({:.2f}%)".format(validity*100))
                logging.info("Best Uniqueness:   ({:.2f}%)".format(uniqueness*100))
                logging.info("Best Novelty:      ({:.2f}%)".format(novelty*100))

                break
                '''
        logging.info(f'Best models at step {best_model_step}')
        logging.info("Best Model Critic Loss:       {:}".format(best_model_critic_loss))
        
        # Close the file handle
        logging.shutdown()

        # Save the last critic model
        torch.save(self.critic.state_dict(), 'models/last_critic.pth')

    def get_mapped(self, seq):
        """Transformation of ids to smiles strings

        Args:
            seq (list[int]): input sequence of ids

        Returns:
            str: output smiles strings 
        """
        return ''.join([self.tokenizer.inv_mapping[i] for i in seq])

    @torch.no_grad()
    def generate_n(self, n):

        noise = torch.randn((n, self.hidden_dim)).to(self.device)

        x = self.generator(noise)['x'].cpu()
        

        lenghts = (x > 0).sum(1)

        # l-1 to exclude the end_token
        return [self.get_mapped(x[:l-1].numpy()) for x, l in zip(x, lenghts)]

    def evaluate_n(self, n, training_data):

        generated_data = self.generate_n(n)


        validity, uniqueness, novelty, total, diversity, novel_mean_qed, novel_mean_solubility, novel_mean_sa = evaluation(generated_data, training_data, log_file='logs/training_log.txt')

        return generated_data, validity, uniqueness, novelty, total, diversity, novel_mean_qed, novel_mean_solubility, novel_mean_sa
