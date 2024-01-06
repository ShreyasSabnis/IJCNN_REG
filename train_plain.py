import torch
import snntorch as snn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from snntorch import surrogate
from snntorch import functional as SF
from snntorch import spikeplot as splt
from snntorch import utils
import tonic
import argparse
from models import *
from ray import tune
from ray.air import Checkpoint, session
from ray.tune.schedulers import ASHAScheduler

# import wandb

# wandb.init()
parser = argparse.ArgumentParser(description='Spiking RNN training')


parser.add_argument('--batch_size_train', default=256, type=int, help='batch size for training the linear classifier')
parser.add_argument('--batch_size_test', default=256, type=int, help='batch size for training the linear classifier')
parser.add_argument('--num_workers', default=24, type=int, help='number of workers for loading the data')
parser.add_argument('--lr', default=0.02, type=float, help='learning rate for training the linear classifier')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum for training the linear classifier')
parser.add_argument('--device', default='cuda', type=str, help='device to be used for training the linear classifier')
parser.add_argument('--num_epochs', default=100, type=int, help='number of epochs for training the linear classifier')
parser.add_argument('--learn_threshold', action='store_true', help='learn threshold for the LIF neurons in the backbone')
parser.add_argument('--threshold', default=1.0, type=float, help='threshold for the LIF neurons in the backbone')

parser.add_argument('--beta', default=0.5, type=float, help='beta for the LIF neurons in the backbone')
parser.add_argument('--time_steps', default=150, type=int, help='number of time steps for the SNN')
parser.add_argument('--learn_beta', action='store_true', help='learn beta for the LIF neurons in the backbone')


args = parser.parse_args()

# wandb.config.update(args)

def main():

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = SNN_Vanilla(beta=args.beta, learn_beta=args.learn_beta, threshold=args.threshold, learn_threshold=args.learn_threshold).to(device)

    sensor_size = tonic.datasets.NMNIST.sensor_size

    transform = tonic.transforms.Compose([
            tonic.transforms.Denoise(filter_time=10000),
            tonic.transforms.ToFrame(sensor_size=sensor_size, n_time_bins=args.time_steps)
            ])

    trainset = tonic.datasets.NMNIST(save_to='./data', train=True, transform=transform)
    testset = tonic.datasets.NMNIST(save_to='./data', train=False, transform=transform)

    
    print("trainset: ", len(trainset))
    print("testset: ", len(testset))

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size_train, shuffle=True, num_workers=args.num_workers, drop_last=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size_test, shuffle=False, num_workers=args.num_workers, drop_last=False)

    #implement a learning rate scheduler

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
    criterion = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)

    loss_history = []
    accuracy_history = []


    for epoch in range(args.num_epochs):  # loop over the dataset multiple times

        running_loss = 0.0

        for i, (input, labels) in enumerate(trainloader, 0):
            
            labels = labels.type(torch.LongTensor)

            input = input.to(device)
            labels = labels.to(device)

            input = input.type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor)

            model.train()

            spk_rec, mem_rec = model(input)

            loss = criterion(spk_rec, labels)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            # print statistics
            running_loss += loss.item()

        #scheduler.step()

        #print epoch loss
        loss_history.append(running_loss / len(trainloader))
        print('[%d] loss: %.3f' % (epoch + 1, running_loss / len(trainloader)))

        total = 0
        running_acc = 0.0

        with torch.no_grad():

            for i, (images, labels) in enumerate(testloader, 0):
                
                labels = labels.type(torch.LongTensor)

                images = images.to(device)
                labels = labels.to(device)

                images = images.type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor)

                model.eval()

                spk_rec, mem_rec = model(images)

                acc = SF.accuracy_rate(spk_rec, labels)*100

                total += labels.size(0)
                running_acc += acc

            accuracy = running_acc / len(testloader)
            accuracy_history.append(accuracy)

            print("Total: {}, Accuracy: {}".format(total, accuracy))
          
if __name__ == '__main__':
    main()