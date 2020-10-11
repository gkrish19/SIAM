# Copyright (c) 2017-2020
# School of Electrical, Computer and Energy Engineering, Arizona State University
# PI: Prof. Yu Cao
# All rights reserved.
#
# This source code is part of NeuroSim - a device-circuit-algorithm framework to benchmark
# neuro-inspired architectures with synaptic devices(e.g., SRAM and emerging non-volatile memory).
# Copyright of the model is maintained by the developers, and the model is distributed under
# the terms of the Creative Commons Attribution-NonCommercial 4.0 International Public License
# http://creativecommons.org/licenses/by-nc/4.0/legalcode.
# The source code is free and you can redistribute and/or modify it
# by providing that the following conditions are met:
#
#  1) Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#  2) Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Developer list:
#		Gokul Krishnan Email: gkrish19@asu.edu

# Credits: Prof.Shimeng Yu and his research group for NeuroSim



from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import bitstring
from subprocess import call

def write_matrix_weight(input_matrix,filename):
    # fill matrix to sqaure
    #print(input_matrix.shape)
    height, width = input_matrix.shape
    bigger_dimension = max(height, width)
    fill_dimension = 1
    while bigger_dimension > fill_dimension:
        fill_dimension = 2*fill_dimension
    filled_matrix = np.zeros([fill_dimension,fill_dimension],dtype=np.float32)
    filled_matrix[0:height,0:width] = input_matrix
    #print(filled_matrix.shape)
    np.savetxt(filename, filled_matrix, delimiter=",",fmt='%10.5f')
    return fill_dimension

def write_matrix_activation(input_matrix,fill_dimension,filename):
    # fill matrix to sqaure
    #print(input_matrix.shape)
    height, width = input_matrix.shape
    filled_matrix = np.zeros([fill_dimension,width],dtype=np.float32)
    filled_matrix_b = np.zeros([fill_dimension,width*32],dtype=np.str)
    filled_matrix[0:height,0:width] = input_matrix
    for i in range(fill_dimension):
        for j in range(width):
            f2 = bitstring.BitArray(float=filled_matrix[i,j], length=32)
            x = f2.bin
            for b in range(32):
                filled_matrix_b[i,(j*32+b)] = x[b]

    np.savetxt(filename, filled_matrix_b, delimiter=",",fmt='%s')
    #print(filled_matrix.shape)


def stretch_input(input_matrix,window_size = 5):
    input_shape = input_matrix.shape
    item_num = (input_shape[2] - 5 + 1) * (input_shape[3]-5 + 1)
    output_matrix = np.zeros((input_shape[0],item_num,input_shape[1]*window_size*window_size))
    iter = 0
    for i in range( input_shape[2]-5 + 1 ):
        for j in range( input_shape[3]-5 + 1 ):
            for b in range(input_shape[0]):
                output_matrix[b,iter,:] = input_matrix[b, :, i:i+window_size,j: j+window_size].reshape(input_shape[1]*window_size*window_size)
            iter += 1

    return output_matrix


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def forward_return_activation(self, x):
        x1 = F.relu(F.max_pool2d(self.conv1(x), 2))
        x2 = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x1)), 2))
        x3 = x2.view(-1, 320)
        x4 = F.relu(self.fc1(x3))
        x5 = F.dropout(x4, training=self.training)
        x6 = self.fc2(x5)
        return x1.cpu().data.numpy(),x3.cpu().data.numpy(),x4.cpu().data.numpy()



def get_activation_and_weight(out_size,model,input,iter):
    weight_layer1 = model.conv1.weight.cpu().data.numpy()
    weight_layer2 = model.conv2.weight.cpu().data.numpy()
    weight_layer3 = model.fc1.weight.cpu().data.numpy()
    weight_layer4 = model.fc2.weight.cpu().data.numpy()
    weight_matrix1 = np.transpose(weight_layer1.reshape(weight_layer1.shape[0],-1))
    weight_matrix2 = np.transpose(weight_layer2.reshape(weight_layer2.shape[0],-1))
    weight_matrix3 = np.transpose(weight_layer3.reshape(weight_layer3.shape[0],-1))
    weight_matrix4 = np.transpose(weight_layer4.reshape(weight_layer4.shape[0],-1))
    f1 = write_matrix_weight(weight_matrix1,'./csv_output/epoch_'+str(iter)+'_layer1_weight.csv')
    f2 = write_matrix_weight(weight_matrix2,'./csv_output/epoch_'+str(iter)+'_layer2_weight.csv')
    f3 = write_matrix_weight(weight_matrix3,'./csv_output/epoch_'+str(iter)+'_layer3_weight.csv')
    f4 = write_matrix_weight(weight_matrix4,'./csv_output/epoch_'+str(iter)+'_layer4_weight.csv')

    act1,act2,act3 = model.forward_return_activation(input)
    layer_input1_streched = stretch_input(input, 5)
    layer_input2_streched = stretch_input(act1, 5)
    for batch_iter in range(out_size):
        layer_input1 = np.transpose(layer_input1_streched[batch_iter,:,:])
        layer_input2 = np.transpose(layer_input2_streched[batch_iter,:,:])
        layer_input3 = np.transpose(act2)
        layer_input4 = np.transpose(act3)
        write_matrix_activation(layer_input1,f1,'./csv_output/epoch_'+str(iter)+'_layer1_input'+str(batch_iter)+'.csv')
        write_matrix_activation(layer_input2,f2,'./csv_output/epoch_'+str(iter)+'_layer2_input'+str(batch_iter)+'.csv')
        write_matrix_activation(layer_input3,f3,'./csv_output/epoch_'+str(iter)+'_layer3_input'+str(batch_iter)+'.csv')
        write_matrix_activation(layer_input4,f4,'./csv_output/epoch_'+str(iter)+'_layer4_input'+str(batch_iter)+'.csv')

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))





def test(args, model, device, test_loader,epoch_iter):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            one_sample = data[0:1,:,:,:]
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    get_activation_and_weight(1,model,one_sample,epoch_iter)
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))



def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)


    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    test(args, model, device, test_loader, 0)
    for epoch in range(1, args.epochs + 1):
        #train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader,epoch)
        call(["./main" "./csv_output/epoch_'+str(iter)+'_layer1_input'+str(epoch)+'.csv" "./csv_output/epoch_'+str(epoch)+'_layer1_weight.csv"  "./csv_output/epoch_'+str(epoch-1)+'_layer1_weight.csv" "32" "32" "32" "32"])
if __name__ == '__main__':
    main()
