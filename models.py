import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from snntorch import surrogate, utils
from collections import OrderedDict

class SNN_Vanilla(nn.Module):
    def __init__(self, beta, learn_beta, threshold, learn_threshold):
        super().__init__()

        self.spike_grad = surrogate.atan()
        self.beta = beta
        self.learn_beta = learn_beta
        self.learn_threshold = learn_threshold

        self.net = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(34*34*2, 1000),
                    snn.Leaky(beta=beta, spike_grad=self.spike_grad, init_hidden=True, learn_beta=self.learn_beta),
                    nn.Linear(1000, 10),
                    snn.Leaky(beta=beta, spike_grad=self.spike_grad, init_hidden=True, learn_beta=self.learn_beta, output=True)
                    )


    def forward(self, data):
        spk_rec = []
        mem_rec = []
        utils.reset(self.net)  # resets hidden states for all LIF neurons in net

        #print(data.size(1))
        for step in range(data.size(1)):  # data.size(1) = number of time steps
            spk_out, mem_out = self.net(data[:,step,:,:,:])
            #spk_out, mem_out = self.net(data[step])
            spk_rec.append(spk_out)
            mem_rec.append(mem_out)
            #print(spk_out.size())
            
        return torch.stack(spk_rec), torch.stack(mem_rec)

class SRNN(nn.Module):
    def __init__(self, beta_lsm, beta_lif, learn_beta, threshold, learn_threshold, N, all_to_all, output_size, input_size):
        super().__init__()

        # no need to change these params
        spike_grad = surrogate.atan()
        self.learn_beta = learn_beta
        self.learn_threshold = learn_threshold
        self.threshold = threshold

        self.beta_lsm = beta_lsm
        self.beta_lif = beta_lif
        self.N = N
        self.all_to_all = all_to_all
        self.output_size = output_size
        self.input_size = input_size

        # self.fc1 = nn.Sequential(
        #     OrderedDict([
        #         ("flatten", nn.Flatten()),
        #         ("fc1", nn.Linear(34*34*2, N)),
        #     ]) 
        # )

        # self.lsm = nn.Sequential(
        #     OrderedDict([
        #         ("lsm", snn.RSynaptic(alpha=self.alpha, beta=self.beta, all_to_all=True, spike_grad=spike_grad, learn_beta=self.learn_beta, 
        #                           learn_threshold=self.learn_threshold, linear_features=self.N, threshold=self.threshold)),
        #     ])
        # )

        # self.fc2 = nn.Sequential(
        #     OrderedDict([
        #         ("fc2", nn.Linear(N, 10)),
        #         ("lif1", snn.Leaky(beta=self.beta, spike_grad=spike_grad, learn_beta=self.learn_beta, output=True))
        #     ])
        # )

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self.input_size, N)
        self.lsm = snn.RLeaky(beta=self.beta_lsm, all_to_all=self.all_to_all, spike_grad=spike_grad, learn_beta=self.learn_beta,
                                    learn_threshold=self.learn_threshold, linear_features=N, threshold=self.threshold)
        self.fc2 = nn.Linear(N, self.output_size)
        #self.fc2 = nn.Linear(2312, 10)
        self.lif1 = snn.Leaky(beta=self.beta_lif, spike_grad=spike_grad, learn_beta=self.learn_beta, output=True)
            

    def forward(self, data):

        spk_rec = []
        mem_rec = []

        spk_lsm, mem_lsm = self.lsm.init_rleaky()
        mem_out = self.lif1.init_leaky()

        # do not compute gradient for first 20% time steps

        # data is of size [batch_size, time_steps, 2, 34, 34]
        with torch.no_grad():

            for step in range(int(data.size(1)*0.2)):

                in_curr = self.fc1(self.flatten(data[:,step,...]))
                spk_lsm, mem_lsm = self.lsm(in_curr, spk_lsm, mem_lsm) 
                out_curr = self.fc2(spk_lsm)
                spk_out, mem_out = self.lif1(out_curr, mem_out)

                spk_rec.append(spk_out)
                mem_rec.append(mem_out)

        # compute gradient for remaining 80% time steps
        for step in range(int(data.size(1)*0.2), data.size(1)):

            in_curr = self.fc1(self.flatten(data[:,step,...]))
            spk_lsm, mem_lsm = self.lsm(in_curr, spk_lsm, mem_lsm)
            out_curr = self.fc2(spk_lsm)
            spk_out, mem_out = self.lif1(out_curr, mem_out)

            spk_rec.append(spk_out)
            mem_rec.append(mem_out)


        return torch.stack(spk_rec), torch.stack(mem_rec)

# output is of size [time_steps, batch_size, num_classes]
class SCRNN(nn.Module):

    # input size if either h or w of input

    def __init__(self, beta_lsm, beta_lif, learn_beta, threshold, learn_threshold, all_to_all, input_size, num_classes):
        
        super().__init__()

        # no need to change these params
        spike_grad = surrogate.atan()
        self.learn_beta = learn_beta
        self.learn_threshold = learn_threshold
        self.threshold = threshold

        self.beta_lsm = beta_lsm
        self.beta_lif = beta_lif
        self.all_to_all = all_to_all
        self.num_classes = num_classes
        self.input_size = input_size

        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=1)
        self.mp1 = nn.MaxPool2d(2)
        self.ln1 = nn.LayerNorm([32, input_size//2, input_size//2], elementwise_affine=False)
        self.lif1 = snn.Leaky(beta=self.beta_lif, spike_grad=spike_grad, learn_beta=self.learn_beta, threshold=self.threshold,
                              learn_threshold=self.learn_threshold)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.mp2 = nn.MaxPool2d(2)
        self.ln2 = nn.LayerNorm([64, input_size//4, input_size//4], elementwise_affine=False)
        self.lif2 = snn.Leaky(beta=self.beta_lif, spike_grad=spike_grad, learn_beta=self.learn_beta, threshold=self.threshold,
                              learn_threshold=self.learn_threshold)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.mp3 = nn.MaxPool2d(2)
        self.ln3 = nn.LayerNorm([128, input_size//8, input_size//8], elementwise_affine=False)
        self.lif3 = snn.Leaky(beta=self.beta_lif, spike_grad=spike_grad, learn_beta=self.learn_beta, threshold=self.threshold,
                              learn_threshold=self.learn_threshold)
        
        self.flatten = nn.Flatten()

        self.rsnn = snn.RLeaky(beta=self.beta_lsm, all_to_all=self.all_to_all, spike_grad=spike_grad, learn_beta=self.learn_beta,
                                    learn_threshold=self.learn_threshold, linear_features=128*(input_size//8)*(input_size//8), threshold=self.threshold)
        
        self.fc_out = nn.Linear(128*(input_size//8)*(input_size//8), num_classes)
        self.lif_out = snn.Leaky(beta=self.beta_lif, spike_grad=spike_grad, learn_beta=self.learn_beta, output=True)
            

    def forward(self, data):

        spk_rec = []
        mem_rec = []

        mem_1 = self.lif1.init_leaky()
        mem_2 = self.lif2.init_leaky()
        mem_3 = self.lif3.init_leaky()
    
        spk_rsnn, mem_rsnn = self.rsnn.init_rleaky()
        mem_out = self.lif_out.init_leaky()

        # data is of size [batch_size, time_steps, c, h, w]

        # do not compute gradient for first 20% time steps
        with torch.no_grad():

            for step in range(int(data.size(1)*0.2)):

                print(data.size())

                spk_1, mem_1 = self.lif1(self.ln1(self.mp1(self.conv1(data[:,step,:,:,:]))), mem_1)
                spk_2, mem_2 = self.lif2(self.ln2(self.mp2(self.conv2(spk_1))), mem_2)
                spk_3, mem_3 = self.lif3(self.ln3(self.mp3(self.conv3(spk_2))), mem_3)

                spk_rsnn, mem_rsnn = self.rsnn(self.flatten(spk_3), spk_rsnn, mem_rsnn)

                spk_out, mem_out = self.lif_out(self.fc_out(spk_rsnn), mem_out)

                spk_rec.append(spk_out)
                mem_rec.append(mem_out)

        # compute gradient for remaining 80% time steps
        for step in range(int(data.size(1)*0.2), data.size(1)):

            spk_1, mem_1 = self.lif1(self.ln1(self.mp1(self.conv1(data[:,step,:,:,:]))), mem_1)
            spk_2, mem_2 = self.lif2(self.ln2(self.mp2(self.conv2(spk_1))), mem_2)
            spk_3, mem_3 = self.lif3(self.ln3(self.mp3(self.conv3(spk_2))), mem_3)

            spk_rsnn, mem_rsnn = self.rsnn(self.flatten(spk_3), spk_rsnn, mem_rsnn)

            spk_out, mem_out = self.lif_out(self.fc_out(spk_rsnn), mem_out)

            spk_rec.append(spk_out)
            mem_rec.append(mem_out)

        print(torch.stack(spk_rec).size())
        exit()
        return torch.stack(spk_rec), torch.stack(mem_rec)