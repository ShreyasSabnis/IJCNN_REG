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

import wandb

wandb.init(project='ijcnn reg')

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

parser.add_argument('--N', default=125, type=int, help='number of neurons in LSM')
parser.add_argument('--beta_lsm', default=0.5, type=float, help='beta for the LIF neurons in the linear classifier')
parser.add_argument('--beta_lif', default=0.5, type=float, help='beta for the LIF neurons in the backbone')
parser.add_argument('--time_steps', default=150, type=int, help='number of time steps for the SNN')
parser.add_argument("--all_to_all", action="store_true", help="Use all-to-all connectivity in the LSM layer")
parser.add_argument("--use_reg_loss", action="store_true", help="Use l1 loss for the lsm layer")
parser.add_argument("--lambda_l1", default=0.01, type=float, help="lambda for l1 loss")
parser.add_argument('--learn_beta', action='store_true', help='learn beta for the LIF neurons in the backbone')
parser.add_argument('--dataset', default='NMNIST', type=str, help='dataset to be used for training the linear classifier')

args = parser.parse_args()

wandb.config.update(args)

def main():

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    SRNN_model = SCRNN(beta_lsm=args.beta_lsm, beta_lif=args.beta_lif, learn_beta=args.learn_beta, threshold=args.threshold, 
                       learn_threshold=args.learn_threshold, all_to_all=args.all_to_all, num_classes=10, input_size=34).to(device)

    # dataloading
    if args.dataset == "NMNIST":

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

    optimizer = torch.optim.Adam(SRNN_model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    criterion = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)

    loss_history = []
    loss_total_history = []
    accuracy_test_history = []

    SRNN_model.train()

    for epoch in range(args.num_epochs):  # loop over the dataset multiple times
        
        # running loss is normal loss
        # running loss total is normal + reg loss

        running_loss = 0.0
        running_loss_total = 0.0

        for i, (input, labels) in enumerate(trainloader, 0):
            
            labels = labels.type(torch.LongTensor)

            input = input.to(device)
            labels = labels.to(device)

            input = input.type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor)

            spk_rec, mem_rec = SRNN_model(input)

            loss = criterion(spk_rec, labels)

            # extract weights of the lsm layer in the SRNN model
            lsm_weights = SRNN_model.lsm.recurrent.weight
            
            if args.use_reg_loss:
                total_loss = loss
            else:
                total_loss = loss
          
            optimizer.zero_grad()

            total_loss.backward()

            optimizer.step()

            # print statistics
            running_loss += loss.detach().cpu.item()
            running_loss_total += total_loss.detach().cpu().item()

            wandb.log({"loss": loss.detach().cpu().item(), "total_loss": total_loss.detach().cpu().item()})

        scheduler.step()

        # calculate % of non-zero weights in the lsm layer and print sparsity
        lsm_weights = SRNN_model.lsm.recurrent.weight
        non_zero = torch.nonzero(lsm_weights)
        non_zero = len(non_zero)
        total = lsm_weights.numel()
        sparsity = 100*(1 - (non_zero/total))

        print("Sparsity: ", sparsity)

        #print epoch loss and total loss
        loss_history.append(running_loss / len(trainloader))
        loss_total_history.append(running_loss_total / len(trainloader))
        print('[%d] loss: %.3f' % (epoch + 1, running_loss / len(trainloader)))
        print('[%d] loss_total: %.3f' % (epoch + 1, running_loss_total / len(trainloader)))

        SRNN_model.eval()

        with torch.no_grad():

            correct = 0
            total = 0

            for i, (images, labels) in enumerate(testloader, 0):
                
                labels = labels.type(torch.LongTensor)

                images = images.to(device)
                labels = labels.to(device)

                images = images.type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor)

                spk_rec, mem_rec = SRNN_model(images)

                # calculate number of correct predictions
                _, idx = spk_rec.sum(dim=0).max(1)
                correct += idx.eq(labels).sum().item()

                total += labels.size(0)

            accuracy_test = 100. * correct / total
            accuracy_test_history.append(accuracy_test)
            max_accuracy = max(accuracy_test_history)

            wandb.log({"test_acc": accuracy_test, "max_test_acc": max_accuracy})

            print('Test accuracy: %.3f %%' % (accuracy_test))
            print('Max test accuracy: %.3f %%' % (max_accuracy))
          
if __name__ == '__main__':
    main()