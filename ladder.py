import torch
import numpy as np
import torch.nn as nn
from torch.nn import Parameter
from torch.autograd import Variable
import torch.legacy.nn as legacynn
import torch.functional as F
import torch.optim as optim
import util
from nnCMul import CMul, CAdd

def add_noise(input_data, mean=0, std=1):

	noise = Variable(torch.FloatTensor(input_data.size()).normal_(mean, std))
	return input_data.add(noise)

class LadderNetwork(nn.Module):
	layer_sizes = [784, 1000, 500, 250, 250, 250, 10]

	def __init__(self, layer_sizes=None):
		super(LadderNetwork, self).__init__()
		if layer_sizes:
			self.layer_sizes = layer_sizes
			self.L = layer_sizes-1

		L = 6
		self.L = L

		self.encoder_layers = nn.ModuleList([None]+[nn.Linear(self.layer_sizes[i-1], self.layer_sizes[i]) for i in range(1,self.L+1)])
		self.decoder_layers = nn.ModuleList([None]+[nn.Linear(self.layer_sizes[i], self.layer_sizes[i-1]) for i in range(1, self.L+1)])

		def get_alpha(i):
			return nn.ParameterList([Parameter(torch.FloatTensor(self.layer_sizes[i]).fill_(0).add_(torch.FloatTensor(self.layer_sizes[i]).normal_(0, 0.1))),
									 Parameter(torch.FloatTensor(self.layer_sizes[i]).fill_(1).add_(torch.FloatTensor(self.layer_sizes[i]).normal_(0, 0.1))),
									 Parameter(torch.FloatTensor(self.layer_sizes[i]).fill_(0).add_(torch.FloatTensor(self.layer_sizes[i]).normal_(0, 0.1))),
									 Parameter(torch.FloatTensor(self.layer_sizes[i]).fill_(0).add_(torch.FloatTensor(self.layer_sizes[i]).normal_(0, 0.1))),
									 Parameter(torch.FloatTensor(self.layer_sizes[i]).fill_(0).add_(torch.FloatTensor(self.layer_sizes[i]).normal_(0, 0.1))),
									 Parameter(torch.FloatTensor(self.layer_sizes[i]).fill_(0).add_(torch.FloatTensor(self.layer_sizes[i]).normal_(0, 0.1))),
									 Parameter(torch.FloatTensor(self.layer_sizes[i]).fill_(1).add_(torch.FloatTensor(self.layer_sizes[i]).normal_(0, 0.1))),
									 Parameter(torch.FloatTensor(self.layer_sizes[i]).fill_(0).add_(torch.FloatTensor(self.layer_sizes[i]).normal_(0, 0.1))),
									 Parameter(torch.FloatTensor(self.layer_sizes[i]).fill_(0).add_(torch.FloatTensor(self.layer_sizes[i]).normal_(0, 0.1))),
									 Parameter(torch.FloatTensor(self.layer_sizes[i]).fill_(0).add_(torch.FloatTensor(self.layer_sizes[i]).normal_(0, 0.1)))
									 ])

		self.alpha_layers = nn.ModuleList([get_alpha(i) for i in range(0, self.L+1)])
		self.gamma = Parameter(torch.FloatTensor(self.layer_sizes[self.L]).fill_(1.).add_(torch.FloatTensor(self.layer_sizes[self.L]).normal_(0, 0.1)))
		self.beta = nn.ParameterList([Parameter(torch.FloatTensor(self.layer_sizes[l]).fill_(0.).add_(torch.FloatTensor(self.layer_sizes[l]).normal_(0, 0.1))) for l in range(0, self.L+1)])

		self.means = [None]*(L+1)
		self.stds = [None]*(L+1)
		self.z = [None]*(L+1)
		self.h = [None]*(L+1)
		self.z_noise = [None]*(L+1)
		self.h_noise = [None]*(L+1)
		self.u = [None]*(L+1)
		self.z_hat = [None]*(L+1)
		self.z_hat_bn = [None]*(L+1)
		self.noise_mean = 0.
		self.noise_std = 0.2

		self.denoising_cost = [1000., 10., 0.1, 0.1, 0.1, 0.1, 0.1]

	def encoder(self, x):
		L = self.L
		m = x.size()[0]

		self.z[0] = x.view(-1, self.layer_sizes[0])
		self.h[0] = self.z[0]
		self.z_noise[0] = add_noise(self.z[0])
		self.h_noise[0] = self.z_noise[0]
		# corrupted encoder
		for i in range(1, L+1):
			self.z_noise[i] = self.encoder_layers[i](self.h_noise[i-1])
			self.z_noise[i] = nn.BatchNorm1d(self.layer_sizes[i])(self.z_noise[i])
			self.z_noise[i] = add_noise(self.z_noise[i], self.noise_mean, self.noise_std)
			self.h_noise[i] = Variable(torch.FloatTensor(self.z_noise[i].size()))
			if i==L:
				for j in range(m):
					self.h_noise[L][j] = self.gamma.mul(self.z_noise[L][j].add(self.beta[L]))
			else:
				for j in range(m):
					self.h_noise[i][j] = nn.ReLU()(self.z_noise[i][j]+self.beta[i])

		self.y_noise = self.h_noise[L]

		self.means[0] = self.z[0].mean(0)
		self.stds[0] = Variable(self.z[0].data.std(0))
		self.stds[0].data.add_(torch.FloatTensor(self.stds[0].data.size()).fill_(1e-4))
		# clean encoder
		for i in range(1, L+1):
			# linear transformation
			self.z[i] = self.encoder_layers[i](self.h[i-1])
			# normalization
			self.means[i] = self.z[i].mean(0)
			self.stds[i] = Variable(self.z[i].data.std(0))
			
			self.z[i] = nn.BatchNorm1d(self.layer_sizes[i])(self.z[i])
			self.h[i] = Variable(torch.FloatTensor(self.z[i].size()))
			# non-linearity
			if i==L:
				for j in range(m):
					self.h[L][j] = self.gamma.mul(self.z[L][j].add(self.beta[L]))
			else:
				for j in range(m):
					self.h[i][j] = nn.ReLU()(self.z[i][j]+self.beta[i])

		self.y_noise = nn.LogSoftmax()(self.h_noise[L])
		self.y = nn.LogSoftmax()(self.h[L])
		return self.y, self.y_noise


	def decoder(self, x):
		L = self.L
		# get batch size
		m = x.size()[0]

		for l in range(L, -1, -1):
			self.z_hat[l] = Variable(torch.FloatTensor(m, self.layer_sizes[l]))
			self.z_hat_bn[l] = Variable(torch.FloatTensor(m, self.layer_sizes[l]))
			if l==L:
				self.u[L] = nn.BatchNorm1d(self.layer_sizes[L])(self.h_noise[L])
			else:
				self.u[l] = nn.BatchNorm1d(self.layer_sizes[l])(self.decoder_layers[l+1](self.z_hat[l+1]))

			def g(z_noise, u, l):
				alpha = self.alpha_layers[l]
				m = z_noise.size()[0]
				mu = Variable(torch.FloatTensor(u.size()))
				v = Variable(torch.FloatTensor(u.size()))
				for i in range(m):
					mu[i] = alpha[0]*nn.Sigmoid()(alpha[1]*u[i]+alpha[2]) + alpha[3]*u[i] + alpha[4]
					v[i] = alpha[5]*nn.Sigmoid()(alpha[6]*u[i]+alpha[7]) + alpha[8]*u[i] + alpha[9]
				self.z_hat[l] = (z_noise-mu)*v + mu

					#self.z_hat[l][i] = params[6].add(params[0].mul(z_noise[i])).add(params[2].mul(u[i])).add(params[4].mul(z_noise[i]).mul(u[i])) \
					#	.add(params[8].mul(nn.Sigmoid()(params[7].add(params[1].mul(z_noise[i])).add(params[3].mul(u[i])) \
					#	.add(params[5].mul(z_noise[i]).mul(u[i])))))

			g(self.z_noise[l], self.u[l], l)
			for i in range(m):
				if l==0:
					n = self.layer_sizes[l]
					self.z_hat_bn[l][i] = self.z_hat[l][i]
				else:
					self.z_hat_bn[l][i] = (self.z_hat[l][i] - self.means[l]) / self.stds[l]
		return self.z_hat[0]

	def forward(self, x):
		self.batch_size = x.size()[0]
		y, y_noise = self.encoder(x)
		z_hat = self.decoder(y)
		return y, z_hat

	def unsup_cost(self):
		# unsupervised denoising reconstruction cost
		unsupervised_func = nn.MSELoss()
		CD = 0.
		for l in range(0, self.L+1):
			clean_target = torch.Tensor(self.z[l].size())
			clean_target.copy_(self.z[l].data)
			clean_target = Variable(clean_target)
			#print(unsupervised_func(self.z_hat_bn[l], clean_target))
			CD += self.denoising_cost[l] * unsupervised_func(self.z_hat_bn[l], clean_target)
		return CD

	def sup_cost(self, target):
		# supervised cost
		supervised_func = nn.NLLLoss()
		CC = supervised_func(self.y_noise, target)
		return CC
