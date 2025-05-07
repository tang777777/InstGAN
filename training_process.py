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
import logging


class FACGAN(nn.Module):
    # Create a logger for the class
    logger = logging.getLogger(__name__)
    
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Set up the logging configuration
    logging.basicConfig(filename='logs/training_log.txt', level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s', filemode='a')

    def __init__(self, data, latent_dim_size=256, lr_generator=2e-4, lr_discriminator=2e-4, lr_critic=2e-4, device='cuda:0', EW=1e-3, SRW_D=7.1e-5, SRW_C=7.1e-5, RL_Flag='Pretrain', Alpha_Initial=0.5, CGW=0.5, QED_W=0.5, Logp_W=0.0, SA_W=0.0, DRD2_W=0.5):

        super().__init__()

        self.device = device

        self.hidden_dim = latent_dim_size

        if RL_Flag=='Pretrain':
            self.tokenizer = Tokenizer(data)
            print('create tokenizer')
        else:
            with open('Results/Pretrain/models/tokenizer.pickle', 'rb') as f:
                self.tokenizer = pickle.load(f)
                            
            print('load tokenizer')

        if not os.path.exists('models'):
            os.makedirs('models')

        with open('models/tokenizer.pickle', 'wb') as f:
            pickle.dump(self.tokenizer, f)



        if RL_Flag == 'ALL':
            self.generator = Generator(
                latent_dim=latent_dim_size,
                vocab_size=self.tokenizer.vocab_size - 1,
                start_token=self.tokenizer.start_token - 1,  
                end_token=self.tokenizer.end_token - 1,
            ).to(device)

            self.discriminator = Discriminator(
                latent_dim=latent_dim_size,
                vocab_size=self.tokenizer.vocab_size,
                start_token=self.tokenizer.start_token,
                bidirectional=True
            ).to(device)

            self.critic_qed = Discriminator(
                latent_dim=latent_dim_size,
                vocab_size=self.tokenizer.vocab_size,
                start_token=self.tokenizer.start_token,
                bidirectional=True
            ).to(device)

            self.critic_logp = Discriminator(
                latent_dim=latent_dim_size,
                vocab_size=self.tokenizer.vocab_size,
                start_token=self.tokenizer.start_token,
                bidirectional=True
            ).to(device)

            self.critic_sa = Discriminator(
                latent_dim=latent_dim_size,
                vocab_size=self.tokenizer.vocab_size,
                start_token=self.tokenizer.start_token,
                bidirectional=True
            ).to(device)


            self.generator_optim = torch.optim.Adam(
                self.generator.parameters(), lr=lr_generator)

            self.discriminator_optim = torch.optim.Adam(
                self.discriminator.parameters(), lr=lr_discriminator, weight_decay=1e-6)

            self.critic_qed_optim = torch.optim.Adam(
                self.critic_qed.parameters(), lr=lr_critic, weight_decay=1e-6)

            self.critic_logp_optim = torch.optim.Adam(
                self.critic_logp.parameters(), lr=lr_critic, weight_decay=1e-6)

            self.critic_sa_optim = torch.optim.Adam(
                self.critic_sa.parameters(), lr=lr_critic, weight_decay=1e-6)
            
            # load discriminator
            discriminator_path = "Results/Pretrain/models/best_Total_discriminator.pth"
            self.discriminator.load_state_dict(torch.load(discriminator_path, map_location=device))


            # load generator
            generator_path = "Results/Pretrain/models/best_Total_generator.pth"
            self.generator.load_state_dict(torch.load(generator_path, map_location=device))


            # load critics
            critic_qed_path = f"critic_pretrain/Results/QED/80000/models/best_critic.pth"
            self.critic_qed.load_state_dict(torch.load(critic_qed_path, map_location=device))

            critic_logp_path = f"critic_pretrain/Results/Solubility/80000/models/best_critic.pth"
            self.critic_logp.load_state_dict(torch.load(critic_logp_path, map_location=device))

            critic_sa_path = f"critic_pretrain/Results/SA/80000/models/best_critic.pth"
            self.critic_sa.load_state_dict(torch.load(critic_sa_path, map_location=device))




        elif RL_Flag == 'QED_DRD2':
            self.generator = Generator(
                latent_dim=latent_dim_size,
                vocab_size=self.tokenizer.vocab_size - 1,
                start_token=self.tokenizer.start_token - 1,  
                end_token=self.tokenizer.end_token - 1,
            ).to(device)

            self.discriminator = Discriminator(
                latent_dim=latent_dim_size,
                vocab_size=self.tokenizer.vocab_size,
                start_token=self.tokenizer.start_token,
                bidirectional=True
            ).to(device)

            self.critic_qed = Discriminator(
                latent_dim=latent_dim_size,
                vocab_size=self.tokenizer.vocab_size,
                start_token=self.tokenizer.start_token,
                bidirectional=True
            ).to(device)

            self.critic_drd2 = Discriminator(
                latent_dim=latent_dim_size,
                vocab_size=self.tokenizer.vocab_size,
                start_token=self.tokenizer.start_token,
                bidirectional=True
            ).to(device)



            self.generator_optim = torch.optim.Adam(
                self.generator.parameters(), lr=lr_generator)

            self.discriminator_optim = torch.optim.Adam(
                self.discriminator.parameters(), lr=lr_discriminator, weight_decay=1e-6)

            self.critic_qed_optim = torch.optim.Adam(
                self.critic_qed.parameters(), lr=lr_critic, weight_decay=1e-6)

            self.critic_drd2_optim = torch.optim.Adam(
                self.critic_drd2.parameters(), lr=lr_critic, weight_decay=1e-6)

            
            # load discriminator
            discriminator_path = "Results/Pretrain/models/best_Total_discriminator.pth"
            self.discriminator.load_state_dict(torch.load(discriminator_path, map_location=device))


            # load generator
            generator_path = "Results/Pretrain/models/best_Total_generator.pth"
            self.generator.load_state_dict(torch.load(generator_path, map_location=device))


            # load critics
            critic_qed_path = f"critic_pretrain/Results/QED/80000/models/best_critic.pth"
            self.critic_qed.load_state_dict(torch.load(critic_qed_path, map_location=device))

            critic_drd2_path = f"critic_pretrain/Results/DRD2/80000/models/best_critic.pth"
            self.critic_drd2.load_state_dict(torch.load(critic_drd2_path, map_location=device))





        else:
            self.generator = Generator(
                latent_dim=latent_dim_size,
                vocab_size=self.tokenizer.vocab_size - 1,
                start_token=self.tokenizer.start_token - 1,  
                end_token=self.tokenizer.end_token - 1,
            ).to(device)

            self.discriminator = Discriminator(
                latent_dim=latent_dim_size,
                vocab_size=self.tokenizer.vocab_size,
                start_token=self.tokenizer.start_token,
                bidirectional=True
            ).to(device)

            self.critic = Discriminator(
                latent_dim=latent_dim_size,
                vocab_size=self.tokenizer.vocab_size,
                start_token=self.tokenizer.start_token,
                bidirectional=True
            ).to(device)


            self.generator_optim = torch.optim.Adam(
                self.generator.parameters(), lr=lr_generator)

            self.discriminator_optim = torch.optim.Adam(
                self.discriminator.parameters(), lr=lr_discriminator, weight_decay=1e-6)

            self.critic_optim = torch.optim.Adam(
                self.critic.parameters(), lr=lr_critic, weight_decay=1e-6)


            if RL_Flag!='Pretrain':

                # load discriminator
                discriminator_path = "Results/Pretrain/models/best_Total_discriminator.pth"
                self.discriminator.load_state_dict(torch.load(discriminator_path, map_location=device))


                # load generator
                generator_path = "Results/Pretrain/models/best_Total_generator.pth"
                self.generator.load_state_dict(torch.load(generator_path, map_location=device))


                # load critic
                critic_path = f"critic_pretrain/Results/{RL_Flag}/80000/models/best_critic.pth"
                self.critic.load_state_dict(torch.load(critic_path, map_location=device))


        self.b = 0.  # baseline reward
        
        self.EW = EW
        self.SRW_D = SRW_D
        self.SRW_C = SRW_C
        
        self.RL_Flag = RL_Flag
        self.Alpha_Initial = Alpha_Initial
        self.CGW = CGW
        self.QED_W = QED_W
        self.Logp_W = Logp_W
        self.SA_W = SA_W
        self.DRD2_W = DRD2_W

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







    def critic_loss_qed(self, x, values):
        """Critic Loss

        Args:
            x (torch.LongTensor): input smiles [batch_size, max_length]
            values (torch.LongTensor): sequence chemical property values 
                                  [batch_size, max_length]

        Returns:
            loss
        """

        y_prediction_values, mask = self.critic_qed(x).values()

        loss = F.mse_loss(y_prediction_values, values, reduction='none') * mask
        '''
        print('y_prediction_values shape: ', y_prediction_values.shape)
        print('y_prediction_values: ', y_prediction_values)
        '''
        # Sum over the losses and divide by the sum of the mask to calculate the mean loss
        loss = loss.sum() / mask.sum()
        

        return loss



    def critic_loss_logp(self, x, values):
        """Critic Loss

        Args:
            x (torch.LongTensor): input smiles [batch_size, max_length]
            values (torch.LongTensor): sequence chemical property values 
                                  [batch_size, max_length]

        Returns:
            loss
        """

        y_prediction_values, mask = self.critic_logp(x).values()

        loss = F.mse_loss(y_prediction_values, values, reduction='none') * mask
        '''
        print('y_prediction_values shape: ', y_prediction_values.shape)
        print('y_prediction_values: ', y_prediction_values)
        '''
        # Sum over the losses and divide by the sum of the mask to calculate the mean loss
        loss = loss.sum() / mask.sum()
        

        return loss



    def critic_loss_sa(self, x, values):
        """Critic Loss

        Args:
            x (torch.LongTensor): input smiles [batch_size, max_length]
            values (torch.LongTensor): sequence chemical property values 
                                  [batch_size, max_length]

        Returns:
            loss
        """

        y_prediction_values, mask = self.critic_sa(x).values()

        loss = F.mse_loss(y_prediction_values, values, reduction='none') * mask
        '''
        print('y_prediction_values shape: ', y_prediction_values.shape)
        print('y_prediction_values: ', y_prediction_values)
        '''
        # Sum over the losses and divide by the sum of the mask to calculate the mean loss
        loss = loss.sum() / mask.sum()
        

        return loss

    def critic_loss_drd2(self, x, values):
        """Critic Loss

        Args:
            x (torch.LongTensor): input smiles [batch_size, max_length]
            values (torch.LongTensor): sequence chemical property values 
                                  [batch_size, max_length]

        Returns:
            loss
        """

        y_prediction_values, mask = self.critic_drd2(x).values()

        loss = F.mse_loss(y_prediction_values, values, reduction='none') * mask
        '''
        print('y_prediction_values shape: ', y_prediction_values.shape)
        print('y_prediction_values: ', y_prediction_values)
        '''
        # Sum over the losses and divide by the sum of the mask to calculate the mean loss
        loss = loss.sum() / mask.sum()
        

        return loss





    def pretrain(self, smiles):
        """One training step

        Args:
           smiles (torch.LongTensor): input smiles
        """

        batch_size, len_real = smiles.size()
        #print(smiles.size())

        ###############
        #Discriminator#
        ###############

        # create real and fake labels
        x_real = smiles.to(self.device)
        y_real = torch.ones(batch_size, len_real).to(self.device)

        # sample latent var
        noise = self.noise_generation(batch_size)
        generator_outputs = self.generator.forward(noise, max_length=50)
        x_gen, log_probs, entropies = generator_outputs.values()


        # label for fake data
        _, len_gen = x_gen.size()
        y_gen = torch.zeros(batch_size, len_gen).to(self.device)

        # discriminator training 
        # discriminator initialization
        self.discriminator_optim.zero_grad()

        # fake loss
        fake_loss = self.discriminator_loss(x_gen, y_gen)

        # real loss
        real_loss = self.discriminator_loss(x_real, y_real)

        # combining fake and real loss
        discr_loss = 0.5 * (real_loss + fake_loss)
        discr_loss.backward()

        # clipping trainable parameters
        clip_grad_value_(self.discriminator.parameters(), 0.1)

        # parameters update
        self.discriminator_optim.step()



        ###########
        #Generator#
        ###########

        # generator training
        # generator initialization
        self.generator_optim.zero_grad()

        # prediction for generated smiles
        y_pred, y_pred_mask = self.discriminator(x_gen).values()

        R_sequence = y_pred * y_pred_mask
        #print(R_sequence.shape)

        R_sequence = R_sequence.sum(1)/y_pred_mask.sum(1)
        #print(R_sequence.shape)

        R_sequence = R_sequence.unsqueeze(1)
        #print(R_sequence.shape)
        #print(R_sequence)



        #print(no_mask_R_sequence.shape)

        # global and moment rewards
        R = (2 * y_pred - 1) + self.SRW_D*R_sequence

        # reward length
        lengths = y_pred_mask.sum(1).long()

        # listing these rewards
        list_rewards = [rw[:ln] for rw, ln in zip(R, lengths)]

        # computing - (r - b) log x
        generator_loss = []
        for reward, log_p in zip(list_rewards, log_probs):

            # minus baseline for relative rewards
            reward_baseline = reward - self.b

            generator_loss.append((- reward_baseline * log_p).sum())

        # mean loss and entropy maximization
        generator_loss = torch.stack(generator_loss).mean() - \
            sum(entropies) * self.EW / batch_size

        # exponentional moving average of rewards 
        with torch.no_grad():
            mean_reward = (R * y_pred_mask).sum() / y_pred_mask.sum()
            self.b = 0.9 * self.b + (1 - 0.9) * mean_reward

        generator_loss.backward()

        clip_grad_value_(self.generator.parameters(), 0.1)

        self.generator_optim.step()

        return {'loss_disc': discr_loss.item(), 'mean_reward': mean_reward}


    def property_optimization(self, smiles, real_values):
        """One training step

        Args:
           smiles (torch.LongTensor): input smiles
           real_values: real data chemical property values
        """

        batch_size, len_real = smiles.size()
        #print(smiles.size())



        

        

        ###############
        #Discriminator#
        ###############


        # create real and fake labels
        x_real = smiles.to(self.device)
        y_real = torch.ones(batch_size, len_real).to(self.device)

        # sample latent var
        noise = self.noise_generation(batch_size)
        generator_outputs = self.generator.forward(noise, max_length=50)
        x_gen, log_probs, entropies = generator_outputs.values()



        # label for fake data
        _, len_gen = x_gen.size()
        y_gen = torch.zeros(batch_size, len_gen).to(self.device)

        # discriminator training 
        # discriminator initialization
        self.discriminator_optim.zero_grad()

        # fake discriminator loss
        fake_loss_discriminator = self.discriminator_loss(x_gen, y_gen)

        # real discriminator loss
        real_loss_discriminator = self.discriminator_loss(x_real, y_real)

        # combining fake and real loss
        discr_loss = 0.5 * (real_loss_discriminator + fake_loss_discriminator)
        discr_loss.backward()

        # clipping trainable parameters
        clip_grad_value_(self.discriminator.parameters(), 0.1)

        # parameters update
        self.discriminator_optim.step()




        ########
        #Critic#
        ########


        x_gen1 = x_gen.cpu()
        lenghts_x_gen = (x_gen1 > 0).sum(1)
        decoded_x_gen = [self.get_mapped(x_gen1[:l-1].numpy()) for x_gen1, l in zip(x_gen1, lenghts_x_gen)]
        

        # Pair encoded and decoded SMILES in tuples
        smiles_pairs = list(zip(x_gen1.tolist(), decoded_x_gen))
        
        # Initialize a list to store all property values (valid and invalid)
        rows = []


        # critic training 
        # critic initialization
        self.critic_optim.zero_grad()


        # Create a list of dictionaries where each dictionary represents a row
        for encoded, decoded in smiles_pairs:
            mol = Chem.MolFromSmiles(decoded)
            if mol != None and mol.GetNumAtoms() > 1 and decoded != ' ':
                try:
                    if self.RL_Flag == 'QED':
                        generated_value = QED.qed(mol)
                    elif self.RL_Flag == 'Solubility':
                        generated_value = solubility(mol)
                    elif self.RL_Flag == 'SA':
                        generated_value = SA(mol)
                except:
                    generated_value = 0  # Assign 0 if an error is encountered
            else:
                generated_value = 0  # Assign 0 to invalid SMILES

            rows.append({'Encoded SMILES': encoded, 'SMILES': decoded, self.RL_Flag: generated_value})

        '''
        print('###############################################################################')
        print('########################GENERATED DATA#########################################')
        print('###############################################################################')
        '''
        # Create a DataFrame from rows
        generated_reward_df = pd.DataFrame(rows)
        
        value_column_name = generated_reward_df.columns[-1]
        generated_smiles = torch.tensor(generated_reward_df['Encoded SMILES'].tolist()).to(self.device)
        generated_data_values = torch.tensor(generated_reward_df[value_column_name].values, dtype=torch.float32).view(-1, 1).expand(-1, generated_smiles.shape[1]).to(self.device)

        # generated critic loss 
        generated_loss_critic = self.critic_loss(generated_smiles, generated_data_values)
            
        

        #print('reward df:  ', generated_reward_df)

        # Convert real_values into a 2D tensor that aligns with the shape of smiles
        real_values = torch.tensor(real_values, dtype=torch.float32).view(-1, 1).expand(-1, smiles.shape[1])
        real_data_values = torch.tensor(real_values).to(self.device)
        
        
        #print('real_data_values:  ', real_data_values)

        #print('real_data_values shape:  ', real_data_values.shape)

        '''
        print('##########################################################################')
        print('########################REAL DATA#########################################')
        print('##########################################################################')
        '''


        # real critic loss
        real_loss_critic = self.critic_loss(x_real, real_data_values)

        
        # combining real and generated loss
        critic_loss =  (1 - self.CGW)*real_loss_critic + self.CGW*generated_loss_critic

        critic_loss.backward()

        # clipping trainable parameters
        clip_grad_value_(self.critic.parameters(), 0.1)

        # parameters update
        self.critic_optim.step()


        ###########
        #Generator#
        ###########
  
        # generator training
        # generator initialization
        self.generator_optim.zero_grad()

        # discriminator prediction for generated smiles
        y_discriminator, y_discriminator_mask = self.discriminator(x_gen).values()


        R_sequence_D = y_discriminator * y_discriminator_mask

        R_sequence_D = R_sequence_D.sum(1)/y_discriminator_mask.sum(1)

        R_sequence_D = R_sequence_D.unsqueeze(1)

        # discriminator reward R_discriminator
        R_discriminator = (2 * y_discriminator - 1) + self.SRW_D*R_sequence_D



        # Initialize R_critic with zeros
        R_critic = torch.zeros_like(R_discriminator).to(self.device)
        

        # critic prediction for generated valid smiles
        y_critic, y_critic_mask = self.critic(generated_smiles).values()


        R_sequence_C = y_critic * y_critic_mask

        R_sequence_C = R_sequence_C.sum(1)/y_critic_mask.sum(1)

        R_sequence_C = R_sequence_C.unsqueeze(1)


        # global and moment rewards


        # valid critic reward R_critic
        R_critic = (2 * y_critic - 1) + self.SRW_C*R_sequence_C
        


        R = (1-self.Alpha_Initial)*R_discriminator + self.Alpha_Initial*R_critic
        


        # reward length
        lengths = y_discriminator_mask.sum(1).long()

        # listing these rewards
        list_rewards = [rw[:ln] for rw, ln in zip(R, lengths)]

        # computing - (r - b) log x
        generator_loss = []
        for reward, log_p in zip(list_rewards, log_probs):

            # minus baseline for relative rewards
            reward_baseline = reward - self.b

            generator_loss.append((- reward_baseline * log_p).sum())

        # mean loss and entropy maximization
        generator_loss = torch.stack(generator_loss).mean() - \
            sum(entropies) * self.EW / batch_size

        # exponentional moving average of rewards 
        with torch.no_grad():
            mean_reward = (R * y_discriminator_mask).sum() / y_discriminator_mask.sum()
            self.b = 0.9 * self.b + (1 - 0.9) * mean_reward

        generator_loss.backward()

        clip_grad_value_(self.generator.parameters(), 0.1)

        self.generator_optim.step()

        return {'loss_disc': discr_loss.item(), 'mean_reward': mean_reward}







    def all_property_optimization(self, smiles, real_qed_values, real_logp_values, real_sa_values):
        """One training step

        Args:
           smiles (torch.LongTensor): input smiles
           real_values: real data chemical property values
        """

        batch_size, len_real = smiles.size()
        #print(smiles.size())



        

        

        ###############
        #Discriminator#
        ###############


        # create real and fake labels
        x_real = smiles.to(self.device)
        y_real = torch.ones(batch_size, len_real).to(self.device)

        # sample latent var
        noise = self.noise_generation(batch_size)
        generator_outputs = self.generator.forward(noise, max_length=50)
        x_gen, log_probs, entropies = generator_outputs.values()



        # label for fake data
        _, len_gen = x_gen.size()
        y_gen = torch.zeros(batch_size, len_gen).to(self.device)

        # discriminator training 
        # discriminator initialization
        self.discriminator_optim.zero_grad()

        # fake discriminator loss
        fake_loss_discriminator = self.discriminator_loss(x_gen, y_gen)

        # real discriminator loss
        real_loss_discriminator = self.discriminator_loss(x_real, y_real)

        # combining fake and real loss
        discr_loss = 0.5 * (real_loss_discriminator + fake_loss_discriminator)
        discr_loss.backward()

        # clipping trainable parameters
        clip_grad_value_(self.discriminator.parameters(), 0.1)

        # parameters update
        self.discriminator_optim.step()




        ########
        #Critic#
        ########


        x_gen1 = x_gen.cpu()
        lenghts_x_gen = (x_gen1 > 0).sum(1)
        decoded_x_gen = [self.get_mapped(x_gen1[:l-1].numpy()) for x_gen1, l in zip(x_gen1, lenghts_x_gen)]
        

        # Pair encoded and decoded SMILES in tuples
        smiles_pairs = list(zip(x_gen1.tolist(), decoded_x_gen))
        
        # Initialize a list to store all property values (valid and invalid)
        rows = []

        # critics training 

        # Create a list of dictionaries where each dictionary represents a row
        for encoded, decoded in smiles_pairs:
            mol = Chem.MolFromSmiles(decoded)
            if mol != None and mol.GetNumAtoms() > 1 and decoded != ' ':
                try:
                    generated_qed = QED.qed(mol)
                    generated_logp = solubility(mol)
                    generated_sa = SA(mol)
                except:
                    # Assign 0 if an error is encountered
                    generated_qed = 0
                    generated_logp = 0
                    generated_sa = 0 
            else:
                # Assign 0 to invalid SMILES
                generated_qed = 0
                generated_logp = 0
                generated_sa = 0 


            rows.append({'Encoded SMILES': encoded, 'SMILES': decoded, 'QED': generated_qed, 'Solubility': generated_logp, 'SA': generated_sa})

        '''
        print('###############################################################################')
        print('########################GENERATED DATA#########################################')
        print('###############################################################################')
        '''
        # Create a DataFrame from rows
        generated_reward_df = pd.DataFrame(rows)
        
        generated_smiles = torch.tensor(generated_reward_df['Encoded SMILES'].tolist()).to(self.device)
        generated_data_qed = torch.tensor(generated_reward_df['QED'].values, dtype=torch.float32).view(-1, 1).expand(-1, generated_smiles.shape[1]).to(self.device)
        generated_data_logp = torch.tensor(generated_reward_df['Solubility'].values, dtype=torch.float32).view(-1, 1).expand(-1, generated_smiles.shape[1]).to(self.device)
        generated_data_sa = torch.tensor(generated_reward_df['SA'].values, dtype=torch.float32).view(-1, 1).expand(-1, generated_smiles.shape[1]).to(self.device)

        # generated critics loss 
        generated_loss_critic_qed = self.critic_loss_qed(generated_smiles, generated_data_qed)
        generated_loss_critic_logp = self.critic_loss_logp(generated_smiles, generated_data_logp)
        generated_loss_critic_sa = self.critic_loss_sa(generated_smiles, generated_data_sa)
            
        

        # Convert real_qed, real_logp, real_sa into 2D tensors that aligns with the shape of smiles
        real_qed = torch.tensor(real_qed_values, dtype=torch.float32).view(-1, 1).expand(-1, smiles.shape[1])
        real_data_qed = torch.tensor(real_qed).to(self.device)

        real_logp = torch.tensor(real_logp_values, dtype=torch.float32).view(-1, 1).expand(-1, smiles.shape[1])
        real_data_logp = torch.tensor(real_logp).to(self.device)

        real_sa = torch.tensor(real_sa_values, dtype=torch.float32).view(-1, 1).expand(-1, smiles.shape[1])
        real_data_sa = torch.tensor(real_sa).to(self.device)
        
        

        # real critics loss
        real_loss_critic_qed = self.critic_loss_qed(x_real, real_data_qed)
        real_loss_critic_logp = self.critic_loss_logp(x_real, real_data_logp)
        real_loss_critic_sa = self.critic_loss_sa(x_real, real_data_sa)

        
        # combining real and generated loss
        critic_loss_qed =  (1 - self.CGW)*real_loss_critic_qed + self.CGW*generated_loss_critic_qed
        critic_loss_logp =  (1 - self.CGW)*real_loss_critic_logp + self.CGW*generated_loss_critic_logp
        critic_loss_sa =  (1 - self.CGW)*real_loss_critic_sa + self.CGW*generated_loss_critic_sa


        # QED:
        self.critic_qed_optim.zero_grad()
        critic_loss_qed.backward()
        clip_grad_value_(self.critic_qed.parameters(), 0.1)
        self.critic_qed_optim.step()

        # Solubility:
        self.critic_logp_optim.zero_grad()
        critic_loss_logp.backward()
        clip_grad_value_(self.critic_logp.parameters(), 0.1)
        self.critic_logp_optim.step()

        # SA:
        self.critic_sa_optim.zero_grad()
        critic_loss_sa.backward()
        clip_grad_value_(self.critic_sa.parameters(), 0.1)
        self.critic_sa_optim.step()


        ###########
        #Generator#
        ###########
  
        # generator training
        # generator initialization
        self.generator_optim.zero_grad()

        # discriminator prediction for generated smiles
        y_discriminator, y_discriminator_mask = self.discriminator(x_gen).values()


        R_sequence_D = y_discriminator * y_discriminator_mask

        R_sequence_D = R_sequence_D.sum(1)/y_discriminator_mask.sum(1)

        R_sequence_D = R_sequence_D.unsqueeze(1)



        # discriminator reward R_discriminator
        R_discriminator = (2 * y_discriminator - 1) + self.SRW_D*R_sequence_D




        # QED:
        # Initialize R_critic with zeros
        R_critic_qed = torch.zeros_like(R_discriminator).to(self.device)
        

        # critic prediction for generated valid smiles
        y_critic_qed, y_critic_mask_qed = self.critic_qed(generated_smiles).values()


        R_sequence_C_qed = y_critic_qed * y_critic_mask_qed

        R_sequence_C_qed = R_sequence_C_qed.sum(1)/y_critic_mask_qed.sum(1)

        R_sequence_C_qed = R_sequence_C_qed.unsqueeze(1)


        # global and moment rewards


        # valid critic reward R_critic
        R_critic_qed = (2 * y_critic_qed - 1) + self.SRW_C*R_sequence_C_qed
        
        # Solubility:
        # Initialize R_critic with zeros
        R_critic_logp = torch.zeros_like(R_discriminator).to(self.device)
        

        # critic prediction for generated valid smiles
        y_critic_logp, y_critic_mask_logp = self.critic_logp(generated_smiles).values()


        R_sequence_C_logp = y_critic_logp * y_critic_mask_logp

        R_sequence_C_logp = R_sequence_C_logp.sum(1)/y_critic_mask_logp.sum(1)

        R_sequence_C_logp = R_sequence_C_logp.unsqueeze(1)


        # global and moment rewards


        # valid critic reward R_critic
        R_critic_logp = (2 * y_critic_logp - 1) + self.SRW_C*R_sequence_C_logp
        


        # SA:
        # Initialize R_critic with zeros
        R_critic_sa = torch.zeros_like(R_discriminator).to(self.device)
        

        # critic prediction for generated valid smiles
        y_critic_sa, y_critic_mask_sa = self.critic_sa(generated_smiles).values()


        R_sequence_C_sa = y_critic_sa * y_critic_mask_sa

        R_sequence_C_sa = R_sequence_C_sa.sum(1)/y_critic_mask_sa.sum(1)

        R_sequence_C_sa = R_sequence_C_sa.unsqueeze(1)


        # global and moment rewards


        # valid critic reward R_critic
        R_critic_sa = (2 * y_critic_sa - 1) + self.SRW_C*R_sequence_C_sa
        


        R_critic = self.QED_W*R_critic_qed + self.Logp_W*R_critic_logp + self.SA_W*R_critic_sa


        R = (1-self.Alpha_Initial)*R_discriminator + self.Alpha_Initial*R_critic
        


        # reward length
        lengths = y_discriminator_mask.sum(1).long()

        # listing these rewards
        list_rewards = [rw[:ln] for rw, ln in zip(R, lengths)]

        # computing - (r - b) log x
        generator_loss = []
        for reward, log_p in zip(list_rewards, log_probs):

            # minus baseline for relative rewards
            reward_baseline = reward - self.b

            generator_loss.append((- reward_baseline * log_p).sum())

        # mean loss and entropy maximization
        generator_loss = torch.stack(generator_loss).mean() - \
            sum(entropies) * self.EW / batch_size

        # exponentional moving average of rewards 
        with torch.no_grad():
            mean_reward = (R * y_discriminator_mask).sum() / y_discriminator_mask.sum()
            self.b = 0.9 * self.b + (1 - 0.9) * mean_reward

        generator_loss.backward()

        clip_grad_value_(self.generator.parameters(), 0.1)

        self.generator_optim.step()

        return {'loss_disc': discr_loss.item(), 'mean_reward': mean_reward}



    def qed_drd2_optimization(self, smiles, real_qed_values, real_drd2_values):
        """One training step

        Args:
           smiles (torch.LongTensor): input smiles
           real_values: real data chemical property values
        """

        batch_size, len_real = smiles.size()
        #print(smiles.size())



        

        

        ###############
        #Discriminator#
        ###############


        # create real and fake labels
        x_real = smiles.to(self.device)
        y_real = torch.ones(batch_size, len_real).to(self.device)

        # sample latent var
        noise = self.noise_generation(batch_size)
        generator_outputs = self.generator.forward(noise, max_length=50)
        x_gen, log_probs, entropies = generator_outputs.values()



        # label for fake data
        _, len_gen = x_gen.size()
        y_gen = torch.zeros(batch_size, len_gen).to(self.device)

        # discriminator training 
        # discriminator initialization
        self.discriminator_optim.zero_grad()

        # fake discriminator loss
        fake_loss_discriminator = self.discriminator_loss(x_gen, y_gen)

        # real discriminator loss
        real_loss_discriminator = self.discriminator_loss(x_real, y_real)

        # combining fake and real loss
        discr_loss = 0.5 * (real_loss_discriminator + fake_loss_discriminator)
        discr_loss.backward()

        # clipping trainable parameters
        clip_grad_value_(self.discriminator.parameters(), 0.1)

        # parameters update
        self.discriminator_optim.step()




        ########
        #Critic#
        ########


        x_gen1 = x_gen.cpu()
        lenghts_x_gen = (x_gen1 > 0).sum(1)
        decoded_x_gen = [self.get_mapped(x_gen1[:l-1].numpy()) for x_gen1, l in zip(x_gen1, lenghts_x_gen)]
        

        # Pair encoded and decoded SMILES in tuples
        smiles_pairs = list(zip(x_gen1.tolist(), decoded_x_gen))
        
        # Initialize a list to store all property values (valid and invalid)
        rows = []

        # critics training 


        
        # Create a list of dictionaries where each dictionary represents a row
        for encoded, decoded in smiles_pairs:
            mol = Chem.MolFromSmiles(decoded)
            if mol != None and mol.GetNumAtoms() > 1 and decoded != ' ':
                try:
                    generated_qed = QED.qed(mol)
                    generated_drd2 = DRD2(mol)
                except:
                    # Assign 0 if an error is encountered
                    generated_qed = 0
                    generated_drd2 = 0
            else:
                # Assign 0 to invalid SMILES
                generated_qed = 0
                generated_drd2 = 0


            rows.append({'Encoded SMILES': encoded, 'SMILES': decoded, 'QED': generated_qed, 'DRD2': generated_drd2})

        '''
        print('###############################################################################')
        print('########################GENERATED DATA#########################################')
        print('###############################################################################')
        '''
        # Create a DataFrame from rows
        generated_reward_df = pd.DataFrame(rows)
        
        generated_smiles = torch.tensor(generated_reward_df['Encoded SMILES'].tolist()).to(self.device)
        generated_data_qed = torch.tensor(generated_reward_df['QED'].values, dtype=torch.float32).view(-1, 1).expand(-1, generated_smiles.shape[1]).to(self.device)
        generated_data_drd2 = torch.tensor(generated_reward_df['DRD2'].values, dtype=torch.float32).view(-1, 1).expand(-1, generated_smiles.shape[1]).to(self.device)

        # generated critics loss 
        generated_loss_critic_qed = self.critic_loss_qed(generated_smiles, generated_data_qed)
        generated_loss_critic_drd2 = self.critic_loss_drd2(generated_smiles, generated_data_drd2)
            
        

        # Convert real_qed, real_drd2 into 2D tensors that aligns with the shape of smiles
        real_qed = torch.tensor(real_qed_values, dtype=torch.float32).view(-1, 1).expand(-1, smiles.shape[1])
        real_data_qed = torch.tensor(real_qed).to(self.device)

        real_drd2 = torch.tensor(real_drd2_values, dtype=torch.float32).view(-1, 1).expand(-1, smiles.shape[1])
        real_data_drd2 = torch.tensor(real_drd2).to(self.device)

        
        

        # real critics loss
        real_loss_critic_qed = self.critic_loss_qed(x_real, real_data_qed)
        real_loss_critic_drd2 = self.critic_loss_drd2(x_real, real_data_drd2)

        
        # combining real and generated loss
        critic_loss_qed =  (1 - self.CGW)*real_loss_critic_qed + self.CGW*generated_loss_critic_qed
        critic_loss_drd2 =  (1 - self.CGW)*real_loss_critic_drd2 + self.CGW*generated_loss_critic_drd2


        # QED:
        self.critic_qed_optim.zero_grad()
        critic_loss_qed.backward()
        clip_grad_value_(self.critic_qed.parameters(), 0.1)
        self.critic_qed_optim.step()

        # DRD2:
        self.critic_drd2_optim.zero_grad()
        critic_loss_drd2.backward()
        clip_grad_value_(self.critic_drd2.parameters(), 0.1)
        self.critic_drd2_optim.step()


        ###########
        #Generator#
        ###########
  
        # generator training
        # generator initialization
        self.generator_optim.zero_grad()

        # discriminator prediction for generated smiles
        y_discriminator, y_discriminator_mask = self.discriminator(x_gen).values()


        R_sequence_D = y_discriminator * y_discriminator_mask

        R_sequence_D = R_sequence_D.sum(1)/y_discriminator_mask.sum(1)

        R_sequence_D = R_sequence_D.unsqueeze(1)



        # discriminator reward R_discriminator
        R_discriminator = (2 * y_discriminator - 1) + self.SRW_D*R_sequence_D




        # QED:
        # Initialize R_critic with zeros
        R_critic_qed = torch.zeros_like(R_discriminator).to(self.device)
        

        # critic prediction for generated valid smiles
        y_critic_qed, y_critic_mask_qed = self.critic_qed(generated_smiles).values()


        R_sequence_C_qed = y_critic_qed * y_critic_mask_qed

        R_sequence_C_qed = R_sequence_C_qed.sum(1)/y_critic_mask_qed.sum(1)

        R_sequence_C_qed = R_sequence_C_qed.unsqueeze(1)


        # global and moment rewards


        # valid critic reward R_critic
        R_critic_qed = (2 * y_critic_qed - 1) + self.SRW_C*R_sequence_C_qed
        
        # DRD2:
        # Initialize R_critic with zeros
        R_critic_drd2 = torch.zeros_like(R_discriminator).to(self.device)
        

        # critic prediction for generated valid smiles
        y_critic_drd2, y_critic_mask_drd2 = self.critic_drd2(generated_smiles).values()


        R_sequence_C_drd2 = y_critic_drd2 * y_critic_mask_drd2

        R_sequence_C_drd2 = R_sequence_C_drd2.sum(1)/y_critic_mask_drd2.sum(1)

        R_sequence_C_drd2 = R_sequence_C_drd2.unsqueeze(1)


        # global and moment rewards


        # valid critic reward R_critic
        R_critic_drd2 = (2 * y_critic_drd2 - 1) + self.SRW_C*R_sequence_C_drd2
        




        R_critic = self.QED_W*R_critic_qed + self.DRD2_W*R_critic_drd2


        R = (1-self.Alpha_Initial)*R_discriminator + self.Alpha_Initial*R_critic
        


        # reward length
        lengths = y_discriminator_mask.sum(1).long()

        # listing these rewards
        list_rewards = [rw[:ln] for rw, ln in zip(R, lengths)]

        # computing - (r - b) log x
        generator_loss = []
        for reward, log_p in zip(list_rewards, log_probs):

            # minus baseline for relative rewards
            reward_baseline = reward - self.b

            generator_loss.append((- reward_baseline * log_p).sum())

        # mean loss and entropy maximization
        generator_loss = torch.stack(generator_loss).mean() - \
            sum(entropies) * self.EW / batch_size

        # exponentional moving average of rewards 
        with torch.no_grad():
            mean_reward = (R * y_discriminator_mask).sum() / y_discriminator_mask.sum()
            self.b = 0.9 * self.b + (1 - 0.9) * mean_reward

        generator_loss.backward()

        clip_grad_value_(self.generator.parameters(), 0.1)

        self.generator_optim.step()

        return {'loss_disc': discr_loss.item(), 'mean_reward': mean_reward}






    def dataloader_creation(self, data, batch_size=1024, shuffle=True, num_workers=0):

        
        def b_tokenize(batch):

            smiles = [item['SMILES'] for item in batch]
            qed_values = [item['QED'] for item in batch]
            solubility_values = [item['Solubility'] for item in batch]
            sa_values = [item['SA'] for item in batch]
            drd2_values = [item['DRD2'] for item in batch]

            tokenized_smiles = self.tokenizer.batch_tokenize(smiles)

            return (tokenized_smiles, qed_values, solubility_values, sa_values, drd2_values)

    

    
        return DataLoader(
            data,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=b_tokenize,
            num_workers=num_workers
        )    
    
    def data_model_saving(self, generated_data, step, validity, uniqueness, novelty, total, diversity, novel_mean_qed, novel_mean_solubility, novel_mean_sa, novel_mean_drd2, eva_metric):
        # create a pandas DataFrame from best_generated_data
        df = pd.DataFrame(generated_data, columns=['text'])
        # save the DataFrame as a CSV file
        df.to_csv(f'logs/best_{eva_metric}_generated_data.csv', index=False)

        setattr(self, f'best_{eva_metric}_model_validity', validity)
        setattr(self, f'best_{eva_metric}_model_uniqueness', uniqueness)
        setattr(self, f'best_{eva_metric}_model_novelty', novelty)
        setattr(self, f'best_{eva_metric}_model_total', total)
        setattr(self, f'best_{eva_metric}_model_diversity', diversity)
        setattr(self, f'best_{eva_metric}_model_step', step)

        setattr(self, f'best_{eva_metric}_model_novel_mean_qed', novel_mean_qed)
        setattr(self, f'best_{eva_metric}_model_novel_mean_solubility', novel_mean_solubility)
        setattr(self, f'best_{eva_metric}_model_novel_mean_sa', novel_mean_sa)
        setattr(self, f'best_{eva_metric}_model_novel_mean_all', self.QED_W * novel_mean_qed + self.Logp_W * novel_mean_solubility + self.SA_W * novel_mean_sa)

        setattr(self, f'best_{eva_metric}_model_novel_mean_drd2', novel_mean_drd2)
        setattr(self, f'best_{eva_metric}_model_novel_mean_qed_drd2', self.QED_W*novel_mean_qed + self.DRD2_W*novel_mean_drd2)

        




        # Save the best generator model
        torch.save(self.generator.state_dict(), f'models/best_{eva_metric}_generator.pth')
        # Save the best discriminator model
        torch.save(self.discriminator.state_dict(), f'models/best_{eva_metric}_discriminator.pth')
        # Save the best critic model
        if (self.RL_Flag != 'ALL') and (self.RL_Flag != 'QED_DRD2'):
            if self.RL_Flag != 'Pretrain':
                torch.save(self.critic.state_dict(), f'models/best_{eva_metric}_critic.pth')
        elif self.RL_Flag == 'ALL':
            torch.save(self.critic_qed.state_dict(), f'models/best_{eva_metric}_critic_qed.pth')
            torch.save(self.critic_logp.state_dict(), f'models/best_{eva_metric}_critic_logp.pth')
            torch.save(self.critic_sa.state_dict(), f'models/best_{eva_metric}_critic_sa.pth')
        elif self.RL_Flag == 'QED_DRD2':
            torch.save(self.critic_qed.state_dict(), f'models/best_{eva_metric}_critic_qed.pth')
            torch.save(self.critic_drd2.state_dict(), f'models/best_{eva_metric}_critic_drd2.pth')


        print(f'Saving the best {eva_metric} generated data and models')
        self.logger.info(f'Saving the best {eva_metric} generated data and models')
        self.logger.info('\n')  

        







    def update_score(self, generated_data, step, validity, uniqueness, novelty, total, diversity, novel_mean_qed, novel_mean_solubility, novel_mean_sa, novel_mean_drd2):
      
        all_score = self.QED_W*novel_mean_qed + self.Logp_W*novel_mean_solubility + self.SA_W*novel_mean_sa
        qed_drd2 = self.QED_W*novel_mean_qed + self.DRD2_W*novel_mean_drd2

        metrics = {
            'QED': (novel_mean_qed, self.best_QED_model_novel_mean_qed, self.best_QED_model_total, self.best_Total_model_novel_mean_qed),
            'SA': (novel_mean_sa, self.best_SA_model_novel_mean_sa, self.best_SA_model_total, self.best_Total_model_novel_mean_sa),
            'Solubility': (novel_mean_solubility, self.best_Solubility_model_novel_mean_solubility, self.best_Solubility_model_total, self.best_Total_model_novel_mean_solubility),
            'ALL': (all_score, self.best_ALL_model_novel_mean_all, self.best_ALL_model_total, self.best_Total_model_novel_mean_all),
            'QED_DRD2': (qed_drd2, self.best_QED_DRD2_model_novel_mean_qed_drd2, self.best_QED_DRD2_model_total, self.best_Total_model_novel_mean_qed_drd2)
        }

        # Update best scores
        if total >= 0.8:

            # pretrain:
            if self.RL_Flag == 'Pretrain':
                if total > self.best_Total_model_total:
                    self.data_model_saving(generated_data, step, validity, uniqueness, novelty, total, diversity, novel_mean_qed, novel_mean_solubility, novel_mean_sa, novel_mean_drd2, 'Total')                        
                elif total == self.best_Total_model_total:
                    if novel_mean_qed > self.best_Total_model_novel_mean_qed:
                        self.data_model_saving(generated_data, step, validity, uniqueness, novelty, total, diversity, novel_mean_qed, novel_mean_solubility, novel_mean_sa, novel_mean_drd2, 'Total')

            # chemical property optimization:
            else:
                if metrics[self.RL_Flag][0] > metrics[self.RL_Flag][1]:
                    self.data_model_saving(generated_data, step, validity, uniqueness, novelty, total, diversity, novel_mean_qed, novel_mean_solubility, novel_mean_sa, novel_mean_drd2, self.RL_Flag)
                elif metrics[self.RL_Flag][0] == metrics[self.RL_Flag][1]:
                    if total > metrics[self.RL_Flag][2]:
                        self.data_model_saving(generated_data, step, validity, uniqueness, novelty, total, diversity, novel_mean_qed, novel_mean_solubility, novel_mean_sa, novel_mean_drd2, self.RL_Flag)
                
                if total > self.best_Total_model_total:
                    self.data_model_saving(generated_data, step, validity, uniqueness, novelty, total, diversity, novel_mean_qed, novel_mean_solubility, novel_mean_sa, novel_mean_drd2, 'Total')                       
                elif total == self.best_Total_model_total:
                    if metrics[self.RL_Flag][0] > metrics[self.RL_Flag][3]:
                        self.data_model_saving(generated_data, step, validity, uniqueness, novelty, total, diversity, novel_mean_qed, novel_mean_solubility, novel_mean_sa, novel_mean_drd2, 'Total') 














    
    def train_n(self, training_data, train_loader, max_step=60000, evaluate_every=100):

        model_names = ['Total', 'QED', 'SA', 'Solubility', 'ALL', 'QED_DRD2']
        metric_names = ['validity', 'uniqueness', 'novelty', 'total', 'diversity', 'step',
                'novel_mean_qed', 'novel_mean_solubility', 'novel_mean_sa', 
                'novel_mean_drd2', 'novel_mean_all', 'novel_mean_qed_drd2']

        for model in model_names:
            for metric in metric_names:
                setattr(self, f'best_{model}_model_{metric}', 0)



        iter_loader = iter(train_loader)
        
        
        print(f'{self.RL_Flag}:  ')
        self.logger.info(f'{self.RL_Flag}:  ')
        print('\n')  
        self.logger.info('\n')  


        for step in range(max_step):

            try:
                batch = next(iter_loader)
                tokenized_smiles, qed_values, solubility_values, sa_values, drd2_values = batch
            except:
                iter_loader = iter(train_loader)
                batch = next(iter_loader)
                tokenized_smiles, qed_values, solubility_values, sa_values, drd2_values = batch

            # pretrain model update
            if self.RL_Flag == 'Pretrain':
                self.pretrain(tokenized_smiles)
           
            # rl optimization model update
            # QED
            elif self.RL_Flag == 'QED':
                self.property_optimization(tokenized_smiles, qed_values)


            # Solubility
            elif self.RL_Flag == 'Solubility':
                self.property_optimization(tokenized_smiles, solubility_values)
 
            # SA
            elif self.RL_Flag == 'SA':
                self.property_optimization(tokenized_smiles, sa_values)
            
            # ALL
            elif self.RL_Flag == 'ALL':
                self.all_property_optimization(tokenized_smiles, qed_values, solubility_values, sa_values)
            
            #QED_DRD2:
            elif self.RL_Flag == 'QED_DRD2':
                self.qed_drd2_optimization(tokenized_smiles, qed_values, drd2_values)



            if (step % evaluate_every == 0) or (step == max_step - 1):
                self.logger.info(f'step {step}')

                self.eval()
                generated_data, validity, uniqueness, novelty, total, diversity, novel_mean_qed, novel_mean_solubility, novel_mean_sa, novel_mean_drd2 = self.evaluate_n(10000, training_data)
                self.train()
                
                
                
                # Update best scores
                self.update_score(generated_data, step, validity, uniqueness, novelty, total, diversity, novel_mean_qed, novel_mean_solubility, novel_mean_sa, novel_mean_drd2)


        eva_metric = 'Total'


        self.logger.info(f'Best {eva_metric}: ')

        self.logger.info(f'Best {eva_metric} models at step {getattr(self, f"best_{eva_metric}_model_step")}')
        self.logger.info(f"Best {eva_metric} model Validity:     {getattr(self, f'best_{eva_metric}_model_validity')*100:.2f}%")
        self.logger.info(f"Best {eva_metric} model Uniqueness:   {getattr(self, f'best_{eva_metric}_model_uniqueness')*100:.2f}%")
        self.logger.info(f"Best {eva_metric} model Novelty:      {getattr(self, f'best_{eva_metric}_model_novelty')*100:.2f}%")
        self.logger.info(f"Best {eva_metric} model Total:        {getattr(self, f'best_{eva_metric}_model_total')*100:.2f}%")
        self.logger.info(f"Best {eva_metric} model Diversity:    {getattr(self, f'best_{eva_metric}_model_diversity'):.2f}")

        self.logger.info(f"Best {eva_metric} model Novel Mean QED:      {getattr(self, f'best_{eva_metric}_model_novel_mean_qed'):.2f}")
        self.logger.info(f"Best {eva_metric} model Novel Mean Solubility:{getattr(self, f'best_{eva_metric}_model_novel_mean_solubility'):.2f}")
        self.logger.info(f"Best {eva_metric} model Novel Mean SA:        {getattr(self, f'best_{eva_metric}_model_novel_mean_sa'):.2f}")
        self.logger.info(f"Best {eva_metric} model Novel Mean DRD2:        {getattr(self, f'best_{eva_metric}_model_novel_mean_drd2'):.2f}")
        self.logger.info(f"Best {eva_metric} model Novel Mean ALL:       {getattr(self, f'best_{eva_metric}_model_novel_mean_all'):.2f}")
        self.logger.info(f"Best {eva_metric} model Novel Mean QED_DRD2:       {getattr(self, f'best_{eva_metric}_model_novel_mean_qed_drd2'):.2f}")
        self.logger.info('\n')  

        if self.RL_Flag != 'Pretrain':
            self.logger.info(f'Best {self.RL_Flag}: ')

            self.logger.info(f'Best {self.RL_Flag} models at step {getattr(self, f"best_{self.RL_Flag}_model_step")}')
            self.logger.info(f"Best {self.RL_Flag} model Validity:     {getattr(self, f'best_{self.RL_Flag}_model_validity')*100:.2f}%")
            self.logger.info(f"Best {self.RL_Flag} model Uniqueness:   {getattr(self, f'best_{self.RL_Flag}_model_uniqueness')*100:.2f}%")
            self.logger.info(f"Best {self.RL_Flag} model Novelty:      {getattr(self, f'best_{self.RL_Flag}_model_novelty')*100:.2f}%")
            self.logger.info(f"Best {self.RL_Flag} model Total:        {getattr(self, f'best_{self.RL_Flag}_model_total')*100:.2f}%")
            self.logger.info(f"Best {self.RL_Flag} model Diversity:    {getattr(self, f'best_{self.RL_Flag}_model_diversity'):.2f}")

            self.logger.info(f"Best {self.RL_Flag} model Novel Mean QED:      {getattr(self, f'best_{self.RL_Flag}_model_novel_mean_qed'):.2f}")
            self.logger.info(f"Best {self.RL_Flag} model Novel Mean Solubility:{getattr(self, f'best_{self.RL_Flag}_model_novel_mean_solubility'):.2f}")
            self.logger.info(f"Best {self.RL_Flag} model Novel Mean SA:        {getattr(self, f'best_{self.RL_Flag}_model_novel_mean_sa'):.2f}")
            self.logger.info(f"Best {self.RL_Flag} model Novel Mean DRD2:        {getattr(self, f'best_{self.RL_Flag}_model_novel_mean_drd2'):.2f}")
            self.logger.info(f"Best {self.RL_Flag} model Novel Mean ALL:       {getattr(self, f'best_{self.RL_Flag}_model_novel_mean_all'):.2f}")
            self.logger.info(f"Best {self.RL_Flag} model Novel Mean QED_DRD2:       {getattr(self, f'best_{self.RL_Flag}_model_novel_mean_qed_drd2'):.2f}")
            self.logger.info('\n')  

            




        # Save the last generator model
        torch.save(self.generator.state_dict(), 'models/last_generator.pth')
        # Save the last discriminator model
        torch.save(self.discriminator.state_dict(), 'models/last_discriminator.pth')
        if (self.RL_Flag != 'ALL') and (self.RL_Flag != 'QED_DRD2'):
            if self.RL_Flag != 'Pretrain':
                # Save the last critic model
                torch.save(self.critic.state_dict(), 'models/last_critic.pth')
        elif self.RL_Flag == 'ALL':
            # Save the last critics model
            torch.save(self.critic_qed.state_dict(), 'models/last_critic_qed.pth')
            torch.save(self.critic_logp.state_dict(), 'models/last_critic_logp.pth')
            torch.save(self.critic_sa.state_dict(), 'models/last_critic_sa.pth')


        elif self.RL_Flag == 'QED_DRD2':
            # Save the last critics model
            torch.save(self.critic_qed.state_dict(), 'models/last_critic_qed.pth')
            torch.save(self.critic_drd2.state_dict(), 'models/last_critic_drd2.pth')

        
        




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


        validity, uniqueness, novelty, total, diversity, novel_mean_qed, novel_mean_solubility, novel_mean_sa, novel_mean_drd2 = evaluation(generated_data, training_data, log_file='logs/training_log.txt')

        return generated_data, validity, uniqueness, novelty, total, diversity, novel_mean_qed, novel_mean_solubility, novel_mean_sa, novel_mean_drd2
