import torch
import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm
from model import MLPnet
from utils import gradient

class Trainer:
    def __init__(self, model: MLPnet, epoch_num):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1, weight_decay=1e-4)
        self.points_batch = 8192
        self.epoch_num = epoch_num
        self.grad_lambda = 0.1
        self.normals_lambda = 1.0
        self.global_sigma = 1.8

        
    def train(self, data):
        print("Trainning")
        sigmas = []
        tree = cKDTree(data)
        for point in np.array_split(data, 100, axis=0):
            d = tree.query(point, 50 + 1)
            sigmas.append(d[0][:, -1])
        sigmas = np.concatenate(sigmas)
        local_sigma = torch.from_numpy(sigmas).float().cuda()

        data = data.to(self.device)
        data.requires_grad_()
        for epoch in tqdm(range(self.epoch_num)):
            # change back to train mode
            self.model.train()
            param_group = self.optimizer.param_groups[0]
            param_group["lr"] = np.maximum(0.005 * (0.5 ** (epoch // 2000)), 5.0e-6)
            # prepare traning data
            indices = torch.tensor(np.random.choice(data.shape[0], self.points_batch, False), dtype=torch.long)
            cur_data = data[indices]
            manifold_pnts = cur_data[:, :3]
            manifold_sigma = local_sigma[indices]
            nonmanifold_pnts = self.get_points(manifold_pnts.unsqueeze(0), manifold_sigma.unsqueeze(0)).squeeze()
            # forward
            manifold_pred = self.model(manifold_pnts)
            nonmanifold_pred = self.model(nonmanifold_pnts)
            manifold_grad = gradient(manifold_pnts, manifold_pred)
            nonmanifold_grad = gradient(nonmanifold_pnts, nonmanifold_pred)
            # manifold loss
            manifold_loss = (manifold_pred.abs()).mean()
            # eikonal loss
            grad_loss = ((nonmanifold_grad.norm(2, dim=-1) - 1) ** 2).mean()
            # normals loss
            normals = cur_data[:, -3:]
            normals_loss = ((manifold_grad - normals).abs()).norm(2, dim=1).mean()
            # loss
            loss = manifold_loss + self.grad_lambda * grad_loss + self.normals_lambda * normals_loss
            # back propagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


    def get_points(self, pc_input, local_sigma):
        batch_size, sample_size, dim = pc_input.shape
        sample_local = pc_input + (torch.randn_like(pc_input) * local_sigma.unsqueeze(-1))
        sample_global = (torch.rand(batch_size, sample_size // 8, dim, device=pc_input.device) * (self.global_sigma * 2)) - self.global_sigma
        sample = torch.cat([sample_local, sample_global], dim=1)
        return sample