from cgi import test
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import random
import numpy as np
import os
from env_2d import Env2D
from utils import *
from rrt2d import RRT
from rrtstar2d import RRTstar
import time
import wandb
import matplotlib.pyplot as plt
from colorama import Fore, Style
import json
from datetime import datetime
from scipy.stats import gaussian_kde
import pickle
import imageio
from moviepy.video.io.bindings import mplfig_to_npimage
from sklearn.neighbors import KernelDensity as KDE2D


class SamplerCVAE(nn.Module):

    def __init__(self, config, env_workspaces, device, data_gather=False, train=False):
        super(SamplerCVAE, self).__init__()
        
        torch.manual_seed(config.getint("training", "seed"))
        torch.cuda.manual_seed(config.getint("training", "seed"))

        # global parameters
        self.device = device
        self.x_lims = (config.getint('data', 'env_x_min'), config.getint('data', 'env_x_max'))
        self.y_lims = (config.getint('data', 'env_y_min'), config.getint('data', 'env_y_max'))
        self.env_dir = config.get("paths", "env_directory") + config.get("paths", "env_name")
        self.env_name = config.get("paths", "env_name")
        self.save_every = config.getint("model", "save_every")
        
        # data collection
        self.time_horizon = config.getint('data_collection', 'time_horizon')
        self.num_generate_data = config.getint('data_collection', 'num_generate_data_loop')
        current_time = datetime.now().strftime('%Y_%m_%d_%H_%M')
        self.data_dir = os.path.join(config['paths']['data_directory'],
                                    '{}/'.format(config['model']['name']), current_time)
        os.makedirs(self.data_dir, exist_ok=True)
        self.data_count = 0
        self.data_buffer = list()

        # training parameters
        self.train_workspace, self.val_workspace, self.test_workspace = env_workspaces
        self.num_epochs = config.getint('training', 'n_epochs')
        if train:
            self.checkpoint_dir = os.path.join(
                config['paths']['checkpoints_directory'],
                '{}/'.format(config['model']['name']),
                current_time)
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            self.img_dir = os.path.join(
                config['paths']['img_directory'],
                '{}/'.format(config['model']['name']),
                current_time)
            os.makedirs(self.img_dir, exist_ok=True)

            train_val_test_ratio = json.loads(config['training']['train_val_test_ratio'])
            self.batch_size = config.getint('training', 'batch_size')
            train_dataset, val_dataset, test_dataset = PreprocessData(train_val_test_ratio, config['paths']['data_path'])
            _train_dataset = DataSet(train_dataset, self.device)
            _val_dataset = DataSet(val_dataset, self.device)
            _test_dataset = DataSet(test_dataset, self.device)
            
            self.trainDataloader = DataLoader(_train_dataset, batch_size=self.batch_size, shuffle=True, \
                                collate_fn=_train_dataset.rollout, drop_last=True)
            self.valDataloader = DataLoader(_val_dataset, batch_size=self.batch_size, shuffle=True, \
                                    collate_fn=_val_dataset.rollout, drop_last=True)    
            self.testDataloader = DataLoader(_test_dataset, batch_size=1, shuffle=True, \
                                    collate_fn=_test_dataset.rollout, drop_last=True)   

        self.extend_len = config.getint('SMP', 'extend_len')
        self.wandb = True if config.getint('mode', 'train') and config.getboolean("log", "wandb") is True else False
        if self.wandb:
            wandb.define_metric("train_step")
            wandb.define_metric("train_loss", step_metric="train_step")
            wandb.define_metric("episode_step")
            wandb.define_metric("episode_train_loss", step_metric="episode_step")
            wandb.define_metric("episode_train_MSE", step_metric="episode_step")
            wandb.define_metric("episode_train_KLD", step_metric="episode_step")
            wandb.define_metric("episode_val_loss", step_metric="episode_step")
            wandb.define_metric("episode_val_MSE", step_metric="episode_step")
            wandb.define_metric("episode_val_KLD", step_metric="episode_step")

        # model parameters
        hidden_size = config.getint("model", "conditional_feature_dimension")
        hidden_size_env1 = 256
        hidden_size_env2 = 128
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=5, stride=3)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=5, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(in_channels=5, out_channels=7, kernel_size=3, stride=1)
        self.env_fc1 = nn.Linear(1183, hidden_size_env1)
        self.env_fc2 = nn.Linear(hidden_size_env1, hidden_size_env2)
        self.env_fc3 = nn.Linear(hidden_size_env2, hidden_size)
        self.relu = nn.ReLU()

        x_dim = 2 # point
        c_dim = hidden_size # environment features (+ start-goal)
        enc_h_dim1 = 256
        enc_h_dim2 = 128
        self.z_dim = config.getint("model", "latent_dimension")
        # encoder part
        self.fc1 = nn.Linear(x_dim + c_dim, enc_h_dim1)
        self.fc2 = nn.Linear(enc_h_dim1, enc_h_dim2)
        self.fc31 = nn.Linear(enc_h_dim2, self.z_dim)
        self.fc32 = nn.Linear(enc_h_dim2, self.z_dim)

        dec_h_dim1 = 128
        dec_h_dim2 = 256
        dec_h_dim3 = 512
        # decoder part
        self.fc4 = nn.Linear(self.z_dim + c_dim, dec_h_dim1)
        self.fc5 = nn.Linear(dec_h_dim1, dec_h_dim2)
        self.fc6 = nn.Linear(dec_h_dim2, dec_h_dim3)
        self.fc7 = nn.Linear(dec_h_dim3, x_dim)

        self.optimizer = torch.optim.Adam(self.parameters(), \
                        lr=config.getfloat('training', 'learning_rate'), \
                        betas=json.loads(config['training']['betas']))
        # self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, \
        #                                 base_lr=0.001, max_lr=0.1,step_size_up=5,mode="triangular2")
        self.mse_loss = torch.nn.MSELoss()
        self.kld_weight = config.getfloat('model', 'KLD_loss_weight')

    def env_encoder(self, env_batch):
        x = self.relu(self.conv1(env_batch))
        x = self.relu(self.conv2(x))
        feature_maps = self.relu(self.conv3(x))
        feature_maps = feature_maps.view(-1, feature_maps.size(1) * feature_maps.size(2) * feature_maps.size(3))
        x = self.relu(self.env_fc1(feature_maps))
        x = self.relu(self.env_fc2(x))
        x = self.relu(self.env_fc3(x))
        return x

    def cvae_encoder(self, label, cond_features):
        concat_input = torch.cat([label, cond_features], 1)
        h = F.relu(self.fc1(concat_input))
        h = F.relu(self.fc2(h))
        return self.fc31(h), self.fc32(h)
    
    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add(mu) # return z sample
    
    def cvae_decoder(self, z, cond_features):
        concat_input = torch.cat([z, cond_features], 1) # detach?? TODO
        h = F.relu(self.fc4(concat_input))
        h = F.relu(self.fc5(h))
        h = F.relu(self.fc6(h))
        
        return self.fc7(h) #self.relu6(self.fc6(h) * 6.0) / 6.0 #torch.sigmoid(self.fc6(h))
    
    def forward(self, env, label):
        env_feature = self.env_encoder(env)
        cond_features = env_feature
        mu, log_var = self.cvae_encoder(label, cond_features)
        z = self.sampling(mu, log_var)
        return self.cvae_decoder(z, cond_features), mu, log_var

    def generate(self, env, label, num=50): #TODO
        env_feature = self.env_encoder(env)
        cond_features = env_feature
        mu, log_var = self.cvae_encoder(label, cond_features)
        z_list = list()
        recon_list = list()
        for _ in range(num):
            z = self.sampling(mu, log_var)
            z_list.append(z[0])
            recon_list.append(self.cvae_decoder(z, cond_features)[0].data.cpu().numpy())
        return z_list, recon_list

    def mixture_sampler(self, planning_env, q_start, q_goal, _lambda=0.5):
        random_sample = tuple()
        if random.uniform(0, 1) > _lambda:
            random_sample = (random.uniform(self.x_lims[0], self.x_lims[1] - 1), 
                             random.uniform(self.y_lims[0], self.y_lims[1] - 1)) if random.uniform(0, 1) > 0.05 else q_goal
        else:
            with torch.no_grad():
                z = torch.randn(1, self.z_dim).to(self.device)
                env_feature = self.env_encoder(torch.from_numpy(np.array([[planning_env.get_env_image()]])).float().to(self.device))
                conditional_features = env_feature
                sample = self.cvae_decoder(z, conditional_features) # it returns sample's cooridnates in [0, 1]
                random_sample = (sample[0].cpu().numpy()[0], sample[0].cpu().numpy()[1])
                random_sample = state_upscaling(random_sample, self.x_lims, self.y_lims)
        return random_sample
    
    def get_data(self): # get data from RRTstar algorithm
        print("Store data from demonstration")
        print("using optimal-variant RRT (RRT*) planner")
        q_start = (3, 3)
        q_goal = (96, 96)
        env_params = {'x_lims': self.x_lims, 'y_lims': self.y_lims}
        for _ in range(self.num_generate_data):
            for i, workspace in enumerate(self.train_workspace):
                # Initialize an environment
                envfile = os.path.abspath(self.env_dir + "/train/" + workspace)
                planning_env = Env2D()
                planning_env.initialize(envfile, env_params)
                planning_env.collision_free(q_start)
                planning_env.collision_free(q_goal)

                # Initialize a planner
                SMP = RRTstar(q_start, planning_env, extend_len=self.extend_len)
                iter = 0
                while iter < self.time_horizon:
                    iter += 1
                    if random.random() > 0.05:
                        random_sample = (random.uniform(self.x_lims[0], self.x_lims[1] - 1), random.uniform(self.y_lims[0], self.y_lims[1] - 1))
                        if SMP.is_collision(random_sample) or SMP.is_contain(random_sample) or random_sample == q_start:
                            continue
                    else: # goal bias
                        random_sample = q_goal
                    q_new = SMP.extend(random_sample)
                    if not q_new:
                        continue

                    SMP.rewire(q_new)
                    if SMP.is_goal_reached(q_new, q_goal, goal_region_radius=3):
                        SMP._q_goal_set.append(q_new)
                
                if SMP._q_goal_set == []:# there is no solution
                    continue
                
                self.data_count += 1
                # Assume that the goal is reached
                SMP.update_best(q_goal) # find best q
                solution = SMP.get_solution_node(SMP._q_best)
                for sample in solution:
                    self.data_buffer.append((planning_env.get_env_image(), state_normalize(sample)))

                if (self.data_count % self.save_every == 0):
                    planning_env.initialize_plot(q_start, q_goal, plot_grid=False)
                    planning_env.plot_path(SMP.reconstruct_path(SMP._q_best), 'solid', 'red', 3)
                    planning_env.plot_states(solution, 'green', alpha=0.8, msize=9)
                    planning_env.plot_save('{}/{}'.format(self.data_dir, str(self.data_count)) )
                print("current_workspace_idx: {}, data_count: {}".format(i, len(self.data_buffer)))

        name = "./" + self.env_name + "_env_sample" + ".pickle"
        with open( name, "wb" ) as file:
            pickle.dump(self.data_buffer, file)

    def fit(self):
        print("Start training....")
        train_step = 0
        q_start = (3, 3)
        q_goal = (96, 96)
        for epoch in range(self.num_epochs):
            self.train()
            _count = 0
            episode_loss_sum = 0
            episode_mse_loss_sum = 0
            episode_kld_loss_sum = 0
            start_time = time.time()
            for data_batch in self.trainDataloader:# TRAINING PHASE           
                env_batch, label_batch = data_batch
                recon_x_batch, mu, log_var = self.forward(env_batch, label_batch)
                loss, mse_loss, kld_loss = self.loss_function(recon_x_batch, label_batch, mu, log_var)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                episode_loss_sum += loss.item()
                episode_mse_loss_sum += mse_loss.item()
                episode_kld_loss_sum += kld_loss.item()
                train_step += 1
                _count += 1
                if self.wandb:
                    wandb.log({"train_step": train_step, "train_loss": loss.item()})
            
            epi_loss = episode_loss_sum / _count
            epi_mse = episode_mse_loss_sum / _count
            epi_kld = episode_kld_loss_sum / _count
            val_loss, val_mse, val_kld = self.evaluate(epoch)
            print("Epoch {} loss [total | MSE | KLD]: {:.5f}, {:.5f}, {:.5f} | KLDweight: {:.4f} | execution time: {:.2f}"
                    .format(epoch, epi_loss, epi_mse, epi_kld, self.kld_weight, time.time() - start_time))
            if self.wandb:
                wandb.log({'epoch_step': epoch, 'epoch_train_loss': epi_loss, 'epoch_val_loss': val_loss,
                            'epoch_train_MSE': epi_mse, 'epoch_train_KLD': epi_kld, 
                            'epoch_val_MSE': val_mse, 'epoch_val_KLD': val_kld})
                
            if epoch % self.save_every == 0:
                self.saveModel(epoch, val_loss)

    def evaluate(self, epoch):
        """Validate a trained-model"""
        self.eval()
        env_params = {'x_lims': self.x_lims, 'y_lims': self.y_lims}
        workspace = random.choice(self.val_workspace)
        envfile = os.path.abspath(self.env_dir + "/validation/" + workspace)
        planning_env = Env2D()
        planning_env.initialize(envfile, env_params)
        q_start = (3, 3)
        q_goal = (96, 96)
        
        # save a picture | get Data from workspace
        SMP = RRTstar(q_start, planning_env, extend_len=self.extend_len)
        with torch.no_grad():
            planning_env.initialize_plot(q_start, q_goal, plot_grid=False)
            rx = list()
            ry = list()
            for _ in range(500): # extract random samples
                random_sample = self.mixture_sampler(planning_env, q_start, q_goal, _lambda=1.0)
                rx.append(random_sample[0])
                ry.append(random_sample[1])
            rx = np.array(rx)
            ry = np.array(ry)
            # Calculate the point density
            rxy = np.vstack([rx, ry])
            rz = gaussian_kde(rxy)(rxy)
            planning_env.plot_kde(SMP.get_rrt(), rx, ry, rz, random_sample)
            planning_env.plot_save('{}/{}'.format(self.img_dir, str(epoch)) )

        # compute a loss
        loss_sum = 0
        mse_loss_sum = 0
        kld_loss_sum = 0
        count = 0
        with torch.no_grad():
            for data_batch in self.valDataloader:# TRAINING PHASE           
                count += 1
                env_batch, label_batch = data_batch
                recon_x_batch, mu, log_var = self.forward(env_batch, label_batch)
                loss, mse_loss, kld_loss = self.loss_function(recon_x_batch, label_batch, mu, log_var)
                loss_sum += loss.item()
                mse_loss_sum += mse_loss.item()
                kld_loss_sum += kld_loss.item()
        return loss_sum / count, mse_loss_sum / count, kld_loss_sum / count


    def test(self, model_path):
        """Test a trained-model"""
        self.eval()
        self.loadModel(model_path)
        SMP = None
        env_params = {'x_lims': self.x_lims, 'y_lims': self.y_lims}
        q_start = (3, 3)
        q_goal = (96, 96)

        iter_ = list()
        time_ = list()
        path_costs = list()

        success_rate = 0

        # for _ in range(100):
        for episode, workspace in enumerate(self.test_workspace):
            envfile = os.path.abspath(self.env_dir + "/test/" + workspace)
            planning_env = Env2D()
            planning_env.initialize(envfile, env_params)
            # q_start, q_goal = planning_env.get_random_start_and_goal()
            SMP = RRTstar(q_start, planning_env, extend_len=self.extend_len)
            
            total_iteration = 0
            solution_path = None
            
            test_imgs = list()
            start_time = time.time()
            path_cost = 0
            for k in range(500):     
                planning_env.initialize_plot(q_start, q_goal, plot_grid=False)
                rx = list()
                ry = list()
                for _ in range(500): # extract random samples
                    random_sample = self.mixture_sampler(planning_env, q_start, q_goal, _lambda=1.0)
                    rx.append(random_sample[0])
                    ry.append(random_sample[1])
                random_sample = self.mixture_sampler(planning_env, q_start, q_goal, _lambda=0.5)
                rx = np.array(rx)
                ry = np.array(ry)
                xx, yy = np.mgrid[0:100:101j, 0:100:101j] # if image_size is 100.
                xy_sample = np.vstack([yy.ravel(), xx.ravel()]).T
                xy_train = np.vstack([ry, rx]).T
                kde_skl = KDE2D(bandwidth=3.0)
                kde_skl.fit(xy_train)
                zz = np.reshape(np.exp(kde_skl.score_samples(xy_sample)), xx.shape)
                planning_env.plot_pcolor(xx, yy, zz, alpha=1.0)
                planning_env.plot_tree(SMP.get_rrt(), 'dashed', 'white', 1)
                planning_env.plot_state(random_sample, color='black', edge_color='white', msize=10)                

                # Calculate the point density
                # rxy = np.vstack([rx, ry])
                # rz = gaussian_kde(rxy)(rxy)
                # planning_env.plot_kde(SMP.get_rrt(), rx, ry, rz, random_sample)
                # planning_env.plot_save( './test_img/{}-{}'.format(str(episode), str(k)) ) 
                planning_env.plot_title("Iteration: {}".format(str(k)))
                data = mplfig_to_npimage(planning_env.figure)  # convert it to a numpy array
                test_imgs.append(data)
                plt.show()
                planning_env.close_plot()
                plt.close()
                
                q_new = SMP.extend(random_sample)
                if not q_new:
                    continue
                
                SMP.rewire(q_new)
                if SMP.is_goal_reached(q_new, q_goal, goal_region_radius=3):
                    SMP._q_goal_set.append(q_new)
                    total_iteration = k
                    solution_path = SMP.reconstruct_path(q_new)
                    # path_cost = SMP.cost(q_new)
                    break

            if solution_path != None:
                planning_env.initialize_plot(q_start, q_goal, plot_grid=False)
                planning_env.plot_title("Total Iteration: {}".format(str(total_iteration)))
                planning_env.plot_tree(SMP.get_rrt(), 'dashed', 'blue', 1)
                planning_env.plot_path(solution_path, 'solid', 'red', 5)
                data = mplfig_to_npimage(planning_env.figure)  # convert it to a numpy array
                # planning_env.plot_save( './test_img/{}-{}'.format(str(episode), str(total_iteration + 1)) )  
                test_imgs.append(data)
            name = './test_img/stationary_{}{}.collcheck{}'.format(str(self.env_name), str(episode), str(total_iteration))
            imageio.mimsave(name + '.gif', test_imgs, fps=3.00)
            img2mp4(test_imgs, name + '.mp4', fps=3)

    def saveModel(self, epoch, loss):
        """Save model parameters under config['model_path']"""
        model_path = '{}/epoch_{}-{}.pt'.format(
            self.checkpoint_dir, str(epoch), str(loss)
        )

        checkpoint = {
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        torch.save(checkpoint, model_path)
        print("Model saved at {}.".format(model_path))
    
    def loadModel(self, model_path):
        """Load model parameters under config['model_path']"""
        try:
            checkpoint = torch.load(model_path)
            print(Fore.YELLOW)
            print("\n[INFO]: model {} loaded, successfully!\n".format(model_path))
            print(Style.RESET_ALL)
        except:
            print(Fore.RED)
            print("\n[INFO]: CAN NOT FIND MODEL AT {}".format(model_path))
            print(Style.RESET_ALL)
            quit()

        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
   
    # return reconstruction error + KL divergence losses
    def loss_function(self, recon_x, x, mu, log_var):
        MSE = self.mse_loss(recon_x, x)
        KLD = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        return MSE + self.kld_weight * KLD, MSE, KLD
