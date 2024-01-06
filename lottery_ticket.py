from copy import deepcopy
from typing import Tuple
import torch.nn.utils.prune as prune
from tqdm import trange
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


parser = argparse.ArgumentParser(description='Spiking RNN training')

parser.add_argument('--batch_size_train', default=256, type=int, help='batch size for training the linear classifier')
parser.add_argument('--batch_size_test', default=256, type=int, help='batch size for training the linear classifier')
parser.add_argument('--num_workers', default=24, type=int, help='number of workers for loading the data')
parser.add_argument('--lr', default=0.02, type=float, help='learning rate for training the linear classifier')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum for training the linear classifier')
parser.add_argument('--device', default='cuda', type=str, help='device to be used for training the linear classifier')
parser.add_argument('--learn_threshold', action='store_true', help='learn threshold for the LIF neurons in the backbone')
parser.add_argument('--threshold', default=1.0, type=float, help='threshold for the LIF neurons in the backbone')

parser.add_argument('--dataset', default='nmnist', type=str, help='dataset to be used for training the linear classifier')
parser.add_argument('--num_epochs', default=5, type=int, help='number of epochs for training the linear classifier')
parser.add_argument('--N', default=1000, type=int, help='number of neurons in LSM')
parser.add_argument('--beta_lsm', default=0.5, type=float, help='beta for the LIF neurons in the linear classifier')
parser.add_argument('--beta_lif', default=0.5, type=float, help='alpha for the LIF neurons in the linear classifier')
parser.add_argument('--time_steps', default=150, type=int, help='number of time steps for the SNN')
parser.add_argument('--prune_pc', default=0.8, type=float, help='percentage of weights to prune')
parser.add_argument('--prune_rounds', default=5, type=int, help='number of pruning rounds')
parser.add_argument("--use_l1_loss", action="store_true", help="Use l1 loss for the lsm layer")
parser.add_argument("--lambda_l1", default=0.01, type=float, help="lambda for l1 loss")
parser.add_argument('--learn_beta', action='store_true', help='learn beta for the LIF neurons in the backbone')
parser.add_argument('--all_to_all', action='store_true', help='use all to all connectivity in the lsm layer')

args = parser.parse_args()

def _train_model(model, optimizer, l1_lambda, trainloader, testloader, num_epochs, device):
       
    loss_history = []
    accuracy_history = []
    loss_total_history = []

    criterion = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)

    for epoch in range(args.num_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        running_loss_total = 0.0

        print(enumerate(trainloader, 0))


        for i, (input, labels) in enumerate(trainloader, 0):
            
            labels = labels.type(torch.LongTensor)

            input = input.to(device)
            labels = labels.to(device)

            input = input.type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor)

            # # print size
            # print(input.size())
            # print(labels.size())
            
            # exit()
            model.train()

            spk_rec, mem_rec = model(input)

            loss = criterion(spk_rec, labels)

            # extract weights of the lsm layer in the SRNN model
            lsm_weights = model.lsm.recurrent.weight
            
            if args.use_l1_loss:
                # add l1 loss of the lsm weights which are in the form of a matrix
                l1_reg = torch.norm(lsm_weights, 1)

                total_loss = loss + args.lambda_l1*l1_reg
            else:
                total_loss = loss
          
            optimizer.zero_grad()

            total_loss.backward()

            optimizer.step()

            # print statistics
            running_loss += loss.item()
            running_loss_total += total_loss.item()

        #scheduler.step()

        # calculate % of non-zero weights in the lsm layer
        lsm_weights = model.lsm.recurrent.weight
        non_zero = torch.nonzero(lsm_weights)
        non_zero = len(non_zero)
        total = lsm_weights.numel()
        sparsity = 100*(1 - (non_zero/total))

        #print epoch loss
        loss_history.append(running_loss / len(trainloader))
        loss_total_history.append(running_loss_total / len(trainloader))
        print('[%d] loss: %.3f' % (epoch + 1, running_loss / len(trainloader)))
        print('[%d] loss_total: %.3f' % (epoch + 1, running_loss_total / len(trainloader)))

        # print the new value of beta in the lsm layer
        print("Beta of lsm: ", model.lsm.beta.item())

        # print the new value of beta in the lif layer
        print("Beta of lif: ", model.lif1.beta.item())

        print("Sparsity: ", sparsity)

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

    return model, accuracy  

def prune_model(model, prune_pc):

    # # module of the lsm layer in the model
    parameters_to_prune = (
        (model.lsm.recurrent, "weight"),
    )

    prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=prune_pc)


    # calculate sparsity in lsm
    lsm_weights = model.lsm.recurrent.weight
    non_zero = torch.nonzero(lsm_weights)
    non_zero = len(non_zero)
    total = lsm_weights.numel()
    sparsity = 100*(1 - (non_zero/total))

    return model, sparsity

def _reinitilise_model(model, initial_weights):
    pruned_state_dict = model.state_dict()

    for parameter_name, parameter_values in initial_weights.items():
        # Pruned weights are called <parameter_name>_orig
        augmented_parameter_name = parameter_name + "_orig"

        if augmented_parameter_name in pruned_state_dict:
            pruned_state_dict[augmented_parameter_name] = parameter_values
        else:
            # Parameter name has not changed
            # e.g. bias or weights from non-pruned layer
            pruned_state_dict[parameter_name] = parameter_values

    model.load_state_dict(pruned_state_dict)

    return model

def main():

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.dataset == 'nmnist':
        model =  SRNN(beta_lsm=args.beta_lsm, beta_lif=args.beta_lif, learn_beta=args.learn_beta, threshold=args.threshold, learn_threshold=args.learn_threshold, 
                            N=args.N, all_to_all=args.all_to_all, output_size=10, input_size=34*34*2).to(device)
    elif args.dataset == 'shd':
        model = SRNN(beta_lsm=args.beta_lsm, beta_lif=args.beta_lif, learn_beta=args.learn_beta, threshold=args.threshold, learn_threshold=args.learn_threshold,
                            N=args.N, all_to_all=args.all_to_all, output_size=20, input_size=700).to(device)
    
    if args.dataset == 'nmnist':

        sensor_size = tonic.datasets.NMNIST.sensor_size

        transform = tonic.transforms.Compose([
                tonic.transforms.Denoise(filter_time=10000),
                tonic.transforms.ToFrame(sensor_size=sensor_size, n_time_bins=args.time_steps)
                ])

        trainset = tonic.datasets.NMNIST(save_to='./data', train=True, transform=transform)
        testset = tonic.datasets.NMNIST(save_to='./data', train=False, transform=transform)

    elif args.dataset == 'shd':

        sensor_size = tonic.datasets.SHD.sensor_size

        transform = tonic.transforms.Compose([
                #tonic.transforms.Denoise(filter_time=10000),
                tonic.transforms.ToFrame(sensor_size=sensor_size, n_time_bins=args.time_steps)
                ])
        
        trainset = tonic.datasets.SHD(save_to='./data', train=True, transform=transform)
        testset = tonic.datasets.SHD(save_to='./data', train=False, transform=transform)

    
    print("trainset: ", len(trainset))
    print("testset: ", len(testset))

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size_train, shuffle=True, num_workers=args.num_workers, drop_last=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size_test, shuffle=False, num_workers=args.num_workers, drop_last=False)

    # save initial weights of the model
    initial_weights = deepcopy(model.state_dict())

    # calculate number of weights in the lsm layer
    lsm_weights = model.lsm.recurrent.weight
    total_parameters = lsm_weights.numel()

    # print names parameters of the model
    for name, param in model.named_parameters():
        print(name)

    print(model.lsm.recurrent.weight.shape)


    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

    prune_pc_per_round = 1 - (1-args.prune_pc) ** (1 / args.prune_rounds)

    for round in range(args.prune_rounds):
        print(f"\nPruning round {round} of {args.prune_rounds}")

        # Train modell
        model, accuracy = _train_model(model, optimizer, args.lambda_l1, trainloader, testloader, args.num_epochs, device)

        # Prune model
        model, sparsity = prune_model(model, prune_pc_per_round)

        print(f"Model accuracy: {accuracy:.3f}%")
        print(f"Sparsity: {sparsity:.3f}%")

        # Reset model
        model = _reinitilise_model(model, initial_weights)


    # Train final model
    model, accuracy = _train_model(model, optimizer, args.lambda_l1, trainloader, testloader, args.num_epochs, device)

    print(f"Final model accuracy: {accuracy:.3f}%")
    print(f"Final sparsity: {sparsity:.3f}%")
          
if __name__ == '__main__':
    main()