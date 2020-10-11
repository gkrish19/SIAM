import argparse
from model_NN import NN
from data_providers.utils import get_data_provider_by_name
from VGG16 import *
from VGG19 import *
from DenseNet import *
from ResNet import *
from config import *
from utils import *
from pact_dorefa import *
import math

train_params_cifar = {
    'batch_size': 128,
    'n_epochs': 200,
    'initial_learning_rate': 0.1,
    'reduce_lr_epoch_1': 100,  # epochs * 0.5
    'reduce_lr_epoch_2': 150,  # epochs * 0.75
    'validation_set': True,
    'validation_split': None,  # None or float
    'shuffle': 'every_epoch',  # None, once_prior_train, every_epoch
    'normalization': 'by_chanels',  # None, divide_256, divide_255, by_chanels
}

train_params_svhn = {
    'batch_size': 64,
    'n_epochs': 40,
    'initial_learning_rate': 0.1,
    'reduce_lr_epoch_1': 20,
    'reduce_lr_epoch_2': 30,
    'validation_set': True,
    'validation_split': None,  # you may set it 6000 as in the paper
    'shuffle': True,  # shuffle dataset every epoch or not
    'normalization': 'divide_255',
}


def get_train_params_by_name(name):
    if name in ['C10', 'C10+', 'C100', 'C100+']:
        return train_params_cifar
    if name == 'SVHN':
        return train_params_svhn


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train', action='store_true',
        help='Train the model')
    parser.add_argument(
        '--test', action='store_true',
        help='Test model for required dataset if pre-trained model exists.'
             'If provided together with `--train` flag testing will be'
             'performed right after training.')
    parser.add_argument(
        '--distr', action='store_true',
        help='Plot the model')
    parser.add_argument(
        '--model_type', '-m', type=str,
        choices=['VGG16', 'VGG19', 'NiN', 'SqueezeNet', 'ResNet32', 'ResNet20', 'ResNet56', 'ResNet110', 'DenseNet', 'DenseNet-BC'],
        default='VGG16',
        help='What type of model to use')
    parser.add_argument(
        '--dataset', '-ds', type=str,
        choices=['C10', 'C10+', 'C100', 'C100+', 'SVHN'],
        default='C10',
        help='What dataset should be used')

    parser.add_argument(
        '--iterations', '-it', type=float, default=200, metavar='',
        help="iterations")
    parser.add_argument(
        '--initial_learning_rate', '-inr', type=float, default=0.1, metavar='',
        help="initial_learning_rate")
    parser.add_argument(
        '--momentum_rate', '-mrr', type=float, default=0.9, metavar='',
        help="momentum_rate")
    parser.add_argument(
        '--dropout_rate', '-drr', type=float, default=0.5, metavar='',
        help="dropout_rate")
    parser.add_argument(
        '--keep_prob', '-kp', type=float, default=1.0, metavar='',
        help="Keep probability for dropout.")
    parser.add_argument(
        '--stddevVar', '-stdvar', type=float, default=0.5, metavar='',
        help="Variation Standard deviation for the RRAM values")
    parser.add_argument(
        '--weight_decay', '-wd', type=float, default=5e-4, metavar='',
        help='Weight decay for optimizer (default: %(default)s)')
    parser.add_argument(
        '--nesterov_momentum', '-nm', type=float, default=0.9, metavar='',
        help='Nesterov momentum (default: %(default)s)')
    parser.add_argument(
        '--logs', dest='should_save_logs', action='store_true',
        help='Write tensorflow logs')
    parser.add_argument(
        '--no-logs', dest='should_save_logs', action='store_false',
        help='Do not write tensorflow logs')
    parser.set_defaults(should_save_logs=True)
    parser.add_argument(
        '--saves', dest='should_save_model', action='store_true',
        help='Save model during training')
    parser.add_argument(
        '--vat', dest='vat', action='store_true',
        help='Perform VAT')
    parser.add_argument(
        '--quant', dest='quant', action='store_true',
        help='Perform Fixed-point quantization')
    parser.add_argument(
        '--rram', dest='rram', action='store_true',
        help='Use custom Conv files')
    parser.add_argument(
        '--no-saves', dest='should_save_model', action='store_false',
        help='Do not save model during training')
    parser.set_defaults(should_save_model=True)
    # should_reload_model

    parser.add_argument(
        '--renew-logs', dest='renew_logs', action='store_true',
        help='Erase previous logs for model if exists.')
    parser.add_argument(
        '--not-renew-logs', dest='renew_logs', action='store_false',
        help='Do not erase previous logs for model if exists.')

    parser.add_argument(
        '--num_inter_threads', '-inter', type=int, default=1, metavar='',
        help='number of inter threads for inference / test')
    parser.add_argument(
        '--num_intra_threads', '-intra', type=int, default=128, metavar='',
        help='number of intra threads for inference / test')

    parser.add_argument(
        '--act_width', '-actw', type=int, default=8, metavar='',
        help="Activation Bitwidth")
    parser.add_argument(
        '--wgt_width', '-wgtw', type=int, default=8, metavar='',
        help="Weights Bitwidth")
    parser.add_argument(
        '--xbar_size', '-xbar_size', type=int, default=256, metavar='',
        help="Activation Bitwidth")
    parser.add_argument(
        '--adc_bits', '-adc_bits', type=int, default=4, metavar='',
        help="Activation Bitwidth")

    # DenseNet Args
    parser.add_argument(
        '--growth_rate', '-k', type=int, choices=[12, 24, 40],
        default=12,
        help='Grows rate for every layer, '
             'choices were restricted to used in paper')
    parser.add_argument(
        '--depth', '-d', type=int, choices=[40, 100, 190, 250],
        default=40,
        help='Depth of whole network, restricted to paper choices')
    parser.add_argument(
        '--total_blocks', '-tb', type=int, default=3, metavar='',
        help='Total blocks of layers stack (default: %(default)s)')
    parser.add_argument(
        '--reduction', '-red', type=float, default=0.5, metavar='',
        help='reduction Theta at transition layer for DenseNets-BC models')

    parser.set_defaults(renew_logs=True)
    args = parser.parse_args()
    if not args.keep_prob:
        if args.dataset in ['C10', 'C100', 'SVHN']:
            args.keep_prob = 0.8
        else:
            args.keep_prob = 1.0
    if args.model_type == 'DenseNet-BC':
        args.bc_mode = True
    else:
        args.bc_mode = False
        args.reduction = 1.0

    model_params = vars(args)
    if not args.train and not args.test and not args.distr:
        print("You should train or test your network. Please check params.")
        exit()

    # some default params dataset/architecture related
    train_params = get_train_params_by_name(args.dataset)
    print("Params:")
    for k, v in model_params.items():
        print("\t%s: %s" % (k, v))
    print("Train params:")
    for k, v in train_params.items():
        print("\t%s: %s" % (k, v))

    print("Prepare training data...")
    data_provider = get_data_provider_by_name(args.dataset, train_params)
    print("Initialize the model..")
    model = NN(data_provider=data_provider, **model_params)
    if args.train:
        print("Data provider train images: ", data_provider.train.num_examples)
        model.train_all_epochs(train_params)
    if args.test:
        if not args.train:
            if args.quant:
                if args.rram:
                    model.rram_load_model()
                elif args.vat:
                    model.vat_load_model()
                else:
                    model.q_load_model()
            else:
                model.load_model()
        print("Data provider test images: ", data_provider.test.num_examples)
        print("Testing...")
        if ((args.vat)):
            loss, accuracy = model.test(data_provider.test, batch_size=200, vat=True)
        else:
            loss, accuracy = model.test(data_provider.test, batch_size=200, vat=False)
        print("mean cross_entropy: %f, mean accuracy: %f" % (loss, accuracy))
    if args.distr:
        if not args.train:
            if args.quant:
                if args.rram:
                    model.rram_load_model()
                elif args.vat:
                    model.vat_load_model()
                else:
                    model.q_load_model()
            else:
                model.load_model()
        print("Data provider test images: ", data_provider.test.num_examples)
        print("Testing...")
        if ((args.vat)):
            loss, accuracy = model.distr(data_provider.test, batch_size=200)
        else:
            loss, accuracy = model.distr(data_provider.test, batch_size=200)
        print("mean cross_entropy: %f, mean accuracy: %f" % (loss, accuracy))
