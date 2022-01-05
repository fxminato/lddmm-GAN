import torch
import os.path
import torch.nn as nn
import numpy as np
from torch.utils.dlpack import to_dlpack, from_dlpack
from torch.distributions.multivariate_normal import MultivariateNormal
import random
#import matplotlib.pyplot as plt
import math
import time
from scipy import stats
from PIL import Image
from torch.optim import lr_scheduler
from torch.optim import Adam, AdamW
import cupy as cp
import imageio
import math
from math import cos, sin, pi, sqrt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch_to_cupy = lambda x: cp.fromDlpack(to_dlpack(x))
cupy_to_torch = lambda x: from_dlpack(x.toDlpack())
from numpy.fft import fft2, ifft2
import matplotlib.pyplot as plt
from ot_loss import Wasserstein2Loss
N = 10  # N must be an even number because we use Simpson quadrature formula

index = torch.zeros([160 * 160, 2]).float().to(device)

BLOCKSIZE = 1024
BLOCKNUM = lambda x : (x - 1) // BLOCKSIZE + 1

s = 0
for i in range(160):
    for j in range(160):
        index[s, 0] = i
        index[s, 1] = j
        s = s + 1

def Phi(t):
    basis = torch.zeros([N + 1, 1], dtype = torch.float32).to(device)
    t0 = round(t.item() * N)
    basis[t0] = ((t0 + 1) / N - t) * N
    if(t0 < N):
        basis[t0 + 1] = (t - t0 / N) * N
    return basis

class Blocks_v(nn.Module):
    def __init__(self):
        super(Blocks_v, self).__init__()
        self.n1 = nn.Linear(2,20)
        self.n2 = nn.Linear(20, 20)
        self.n3 = nn.Linear(20, 2)
        #self.n1.weight.data.normal_(0, 0.1)
        #nn.init.xavier_normal_(self.n1.weight, 0.1)
        #self.n1.bias.data.fill_(0)
        #self.n2.weight.data.normal_(0, 0.1)
        #nn.init.xavier_normal_(self.n2.weight, 0.1)
        #self.n2.bias.data.fill_(0)
        #self.n3.weight.data.normal_(0, 0.1)
        #nn.init.xavier_normal_(self.n3.weight, 0.1)
        #self.n3.bias.data.fill_(0)
    def forward(self, x):
        y = x
        y = self.n1(y)
        y = nn.functional.relu(y)
        y = self.n2(y)
        #y = torch.tanh(y)
        y = nn.functional.relu(y)
        y = self.n3(y)
        return y

class Resnet(nn.Module):
    def __init__(self):
        super(Resnet, self).__init__()
        self.Linears = nn.ModuleList([Blocks_v() for i in range(N + 1)])

    def forward(self, x):
        for i, l in enumerate(self.Linears):
            x = x + l(x) / N
        return x



class vector(nn.Module):
    def __init__(self):
        super(vector, self).__init__()
        self.Linears = nn.ModuleList([Blocks_v() for i in range(N + 1)])

    def forward(self, x, t):
        y = torch.zeros_like(x)
        for i, l in enumerate(self.Linears):
            y = y + l(x) * Phi(t)[i]
        return y

# Loss_net 输入x,s  输出loss
class lddmmnet(nn.Module):
    def __init__(self):
        super(lddmmnet, self).__init__()
        self.V = vector().to(device)
        #self.V = Blocks_v().to(device)
        self.x, self.Vloss = self.ode_solve()
        self.Dv = None
        #求解的过程只保留在初始化中

    def smooth(self):
        tt = torch.linspace(0, 1, N, device = device).float()
        sum = torch.zeros(1, device = device)
        for n in range(N - 1):  # 0, 2,4,...38
            sum += torch.norm(self.V(index, tt[n + 1]) - self.V(index, tt[n])) ** 2
        self.Dv = sum

    def forward(self, t):
        batch_size = t.shape[0]
        result = torch.zeros([batch_size, 2, 160, 160], dtype = torch.float32).to(device)
        for i in range(batch_size):
            indice = round(t[i].item() * N)
            if (indice == N):
                indice = N - 1
            result[i] = self.x[indice]
        return result

    def ode_solve(self):
        tt = torch.linspace(0, 1, N, device=device).float() #把时间分为2n + 1 份
        loss = torch.zeros(1, device = device)
        X = torch.zeros(N, 2, 160, 160, dtype = torch.float32).to(device)
        X0 = index
        for n in range(N): # 0, 2,4,...38
            X[n] = X0.reshape(160, 160, 2).permute(2, 0, 1)
            V0 = self.V(X0,tt[n])
            #loss = loss + torch.norm(_operator_L(V1))**2
            loss = loss + torch.norm(V0) ** 2
            X0 = X0 + V0 / N
        return X, loss

    def clean(self):
        tt = torch.linspace(0, 1, N, device=device).float() #把时间分为2n + 1 份
        loss = torch.zeros(1, device = device)
        X = torch.zeros(N, 2, 160, 160, dtype = torch.float32).to(device)
        X0 = index
        for n in range(N):  # 0, 2,4,...38
            X[n] = X0.reshape(160, 160, 2).permute(2, 0, 1)
            V0 = self.V(X0, tt[n])
            # loss = loss + torch.norm(_operator_L(V1))**2
            loss = loss + torch.norm(V0) ** 2
            X0 = X0 + V0 / N
        self.x = X
        self.Vloss = loss

    def plot_velocity(self):
        Y, X = np.mgrid[:160, :160]
        plt.subplot(3, 3, 1)
        plt.title('t=0.1')
        vector = self.V(index, torch.tensor([0.1]).to(device))
        vector = vector.reshape(160, 160, 2).permute(2, 0, 1).cpu().detach().numpy()
        plt.streamplot(X, Y, vector[0], vector[1], density=[0.5, 1])
        plt.subplot(3, 3, 2)
        plt.title('t=0.2')
        vector = self.V(index, torch.tensor([0.2]).to(device))
        vector = vector.reshape(160, 160, 2).permute(2, 0, 1).cpu().detach().numpy()
        plt.streamplot(X, Y, vector[0], vector[1], density=[0.5, 1])
        plt.subplot(3, 3, 3)
        plt.title('t=0.3')
        vector = self.V(index, torch.tensor([0.3]).to(device))
        vector = vector.reshape(160, 160, 2).permute(2, 0, 1).cpu().detach().numpy()
        plt.streamplot(X, Y, vector[0], vector[1], density=[0.5, 1])
        plt.subplot(3, 3, 4)
        plt.title('t=0.4')
        vector = self.V(index, torch.tensor([0.4]).to(device))
        vector = vector.reshape(160, 160, 2).permute(2, 0, 1).cpu().detach().numpy()
        plt.streamplot(X, Y, vector[0], vector[1], density=[0.5, 1])
        plt.subplot(3, 3, 5)
        plt.title('t=0.5')
        vector = self.V(index, torch.tensor([0.5]).to(device))
        vector = vector.reshape(160, 160, 2).permute(2, 0, 1).cpu().detach().numpy()
        plt.streamplot(X, Y, vector[0], vector[1], density=[0.5, 1])
        plt.subplot(3, 3, 6)
        plt.title('t=0.6')
        vector = self.V(index, torch.tensor([0.6]).to(device))
        vector = vector.reshape(160, 160, 2).permute(2, 0, 1).cpu().detach().numpy()
        plt.streamplot(X, Y, vector[0], vector[1], density=[0.5, 1])
        plt.subplot(3, 3, 7)
        plt.title('t=0.7')
        vector = self.V(index, torch.tensor([0.7]).to(device))
        vector = vector.reshape(160, 160, 2).permute(2, 0, 1).cpu().detach().numpy()
        plt.streamplot(X, Y, vector[0], vector[1], density=[0.5, 1])
        plt.subplot(3, 3, 8)
        plt.title('t=0.8')
        vector = self.V(index, torch.tensor([0.8]).to(device))
        vector = vector.reshape(160, 160, 2).permute(2, 0, 1).cpu().detach().numpy()
        plt.streamplot(X, Y, vector[0], vector[1], density=[0.5, 1])
        plt.subplot(3, 3, 9)
        plt.title('t=0.9')
        vector = self.V(index, torch.tensor([0.9]).to(device))
        vector = vector.reshape(160, 160, 2).permute(2, 0, 1).cpu().detach().numpy()
        plt.streamplot(X, Y, vector[0], vector[1], density=[0.5, 1])
        plt.savefig('./velocity.png')

class diff(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i0, phi):
        #输入I_0 1 * size * size, phi: batch_size * 2 * size * size
        ctx.phi = phi
        ctx.i0 = i0
        n = i0.shape[-1]
        batch_size = phi.shape[0]
        cpi0 = torch_to_cupy(i0)
        cpphi = torch_to_cupy(phi)
        cpi1 = cp.zeros((batch_size, 1, n, n), dtype = cp.float32)
        for i in range(batch_size):
            ker_warp((BLOCKNUM(n * n), ), (BLOCKSIZE, ), (cpi1[i], cpi0[0], cpphi[i, 0], cpphi[i, 1], n))
        return cupy_to_torch(cpi1)

    @staticmethod
    def backward(ctx, grad_output):
        phi = ctx.phi
        i0 = ctx.i0
        n = i0.shape[-1]
        batch_size = phi.shape[0]
        grad = cp.zeros([batch_size, 2, n, n], dtype=cp.float32)
        cpgrad = torch_to_cupy(grad_output)
        cpi0 = torch_to_cupy(i0)
        cpphi = torch_to_cupy(phi)
        for i in range(batch_size):
            ker_grad((BLOCKNUM(n * n),), (BLOCKSIZE,), (grad[i, 0],grad[i, 1], cpgrad[i,0], cpi0[0], cpphi[i, 0], cpphi[i, 1], n))
        return (None, cupy_to_torch(grad))

ker_warp = cp.RawKernel(r'''extern "C" __global__ void warp(float* y, const float* x, const float* f1, const float* f2, const long n) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < n * n) {
        long i = floor(f1[tid]);
        long j = floor(f2[tid]);
        float di = f1[tid] - i;
        float dj = f2[tid] - j;
        if (i < 0) {
            i = 0;
            di = 0.00001;}
        if (j < 0) {
            j = 0;
            dj = 0.00001;}
        if (i >= n - 1) {
            i = n - 2;
            di = 1 - 0.00001;
        }
        if (j >= n - 1) {
            j = n - 2;
            dj = 1 - 0.00001;
        }

        if (0 <= i && i < n && 0 <= j && j < n)
            atomicAdd(y + i * n + j, x[tid] * (1 - di) * (1 - dj));
        if (0 <= i + 1 && i + 1 < n && 0 <= j && j < n)
            atomicAdd(y + (i + 1) * n + j, x[tid] * di * (1 - dj));
        if (0 <= i && i < n && 0 <= j + 1 && j + 1 < n)
            atomicAdd(y + i * n + j + 1, x[tid] * (1 - di) * dj);
        if (0 <= i + 1 && i + 1 < n && 0 <= j + 1 && j + 1 < n)
            atomicAdd(y + (i + 1) * n + j + 1, x[tid] * di * dj);
    }}''','warp')

ker_grad = cp.RawKernel(r'''extern "C" __global__ void grad(float* v1, float *v2, float* y,const float* x,float* f1,float* f2,const long n)
{   int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < n * n) {
        long i = floor(f1[tid]);
        long j = floor(f2[tid]);
        float di = f1[tid] - i;
        float dj = f2[tid] - j;
        if (i < 0) {
            i = 0;
            di = 0.00001;}
        if (j < 0) {
            j = 0;
            dj = 0.00001;}
        if (i >= n - 1) {
            i = n - 2;
            di = 1 - 0.00001;
        }
        if (j >= n - 1) {
            j = n - 2;
            dj = 1 - 0.00001;
        }

        if (0 <= i && i < n && 0 <= j && j < n)
            v1[tid] = v1[tid] + x[tid] * (dj - 1) * y[i * n + j];
            v2[tid] = v2[tid] + x[tid] * (di - 1) * y[i * n + j];
        if (0 <= i + 1 && i + 1 < n && 0 <= j && j < n)
            v1[tid] = v1[tid] + x[tid] * (1 - dj) * y[(i + 1) * n + j];
            v2[tid] = v2[tid] + x[tid] * (-di) * y[(i + 1) * n + j];
        if (0 <= i && i < n && 0 <= j + 1 && j + 1 < n)
            v1[tid] = v1[tid] + x[tid] * (-dj) * y[i * n + j + 1];
            v2[tid] = v2[tid] + x[tid] * (1 - di) * y[i * n + j + 1];
        if (0 <= i + 1 && i + 1 < n && 0 <= j + 1 && j + 1 < n)
            v1[tid] = v1[tid] + x[tid] * (dj) * y[(i + 1) * n + j + 1];
            v2[tid] = v2[tid] + x[tid] * (di) * y[(i + 1) * n + j + 1];
    }}''','grad')

def tensor2uint(img):
    img = img.data.squeeze().float().clamp_(0, 1).cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    return np.uint8((img*255.0).round())

if __name__ == '__main__':

    i0 = np.array(Image.open('./t5-0.png'))
    i0 = torch.from_numpy(i0 / 255.).float().reshape(1, 160, 160).to(device)
    #model = Resnet().to(device)
    model = lddmmnet().to(device)
    iter = 10000

    fx = torch.zeros_like(index).to(device)

    fx[:, 0] = (index[:, 0] - 80) * cos(pi / 12) - (index[:, 1] - 80) * sin(pi / 12) + 80
    fx[:, 1] = (index[:, 0] - 80) * sin(pi / 12) + (index[:, 1] - 80) * cos(pi / 12) + 80
    #fx[:, 0] = index[:, 0]
    #fx[:, 1] = index[:, 1] + 10
    fx = fx.reshape(160, 160, 2).permute(2, 0, 1)
    #fx = fx.reshape(1, 2, 160, 160)
    true = torch.zeros(1, 2, 160, 160).float().to(device)
    true[0] = fx
    i1 = diff.apply(i0, true)
    #image = tensor2uint(i1[0])
    #yfy = Image.fromarray(image)
    #yfy.save('./hhh.png')
    #fx[:, 0] = (index[:, 0] - 80) * cos(pi / 2) - (index[:, 1] - 80) * sin(pi / 2)
    #fx[:, 1] = (index[:, 0] - 80) * sin(pi / 2) + (index[:, 1] - 80) * cos(pi / 2)
    #fx = fx / 6
    list = [1e+9]
    for lamda in list:
        model = lddmmnet().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr = 1e-2)
        Loss1 = []
        Loss2 = []
        Loss3 = []
        iteration = []
        for epoch in range(iter):
            optimizer.zero_grad()
            t = torch.tensor([1]).float().unsqueeze(1).to(device)
            phi = model(t)
            i11 = diff.apply(i0, phi)
            #loss1 = (i11 - i1).norm() ** 2
            paras = {'eps': 1e-4, 'delta': 1e-6, 'maxiter': 2000,'verbose': 0 }
            #loss1 = torch.norm(i11 - i1) ** 2
            loss1 = Wasserstein2Loss.apply(i11[0,0], i1[0,0], paras)
            loss2 = model.Vloss / lamda
            loss3 = model.Dv / lamda
            loss = loss1 + loss2 + loss3
            loss.backward()
            optimizer.step()
            print(f'epoch {epoch}, loss {loss.item()}, accuracy loss {loss1.item()}, velocity loss {loss2.item()}')
            model.clean()
            model.smooth()
            iteration.append(epoch)
            Loss1.append(loss1)
            Loss2.append(loss2)
            Loss3.append(loss)
        plt.subplot(1, 3, 1)
        plt.plot(iteration, Loss1, 'b', label='accuracy loss')
        plt.xlabel('epoch')
        plt.ylabel('Epoch_Loss')
        plt.legend()
        plt.subplot(1, 3, 2)
        plt.plot(iteration, Loss2, 'r', label='velocity loss')
        plt.xlabel('epoch')
        plt.ylabel('Epoch_Loss')
        plt.legend()
        plt.subplot(1, 3, 3)
        plt.plot(iteration, Loss3, 'y', label='total loss')
        plt.xlabel('epoch')
        plt.ylabel('Epoch_Loss')
        plt.legend()
        plt.savefig('./test{:f}.png'.format(lamda))
        plt.close()
        ts = [k / 10 for k in range(10)]
        t = torch.tensor(ts).float().unsqueeze(1).to(device)
        phi = model(t)
        movie = diff.apply(i0, phi)
        image_list = []
        for i in range(movie.shape[0]):
            frame = tensor2uint(movie[i])
            frame = Image.fromarray(frame)
            image_list.append(frame)
        imageio.mimsave('./result{:f}.gif'.format(lamda), image_list)
