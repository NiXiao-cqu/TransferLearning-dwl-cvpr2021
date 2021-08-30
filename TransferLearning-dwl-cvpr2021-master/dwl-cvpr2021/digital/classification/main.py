from __future__ import print_function
import argparse
import torch
from solver import Solver
import os

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MCD Implementation')
parser.add_argument('--all_use', type=str, default='no', metavar='N',
                    help='use all training data? in usps adaptation')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoint', metavar='N',
                    help='source only or not')
parser.add_argument('--eval_only', action='store_true', default=False,
                    help='evaluation only option')
parser.add_argument('--lr', type=float, default=0.0002, metavar='LR',
                    help='learning rate (default: 0.0002)')
parser.add_argument('--max_epoch', type=int, default=200, metavar='N',
                    help='how many epochs')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--num_k', type=int, default=4, metavar='N',
                    help='hyper paremeter for generator update')
parser.add_argument('--one_step', action='store_true', default=False,
                    help='one step training with gradient reversal layer')
parser.add_argument('--optimizer', type=str, default='adam', metavar='N', help='which optimizer')
parser.add_argument('--resume_epoch', type=int, default=100, metavar='N',
                    help='epoch to resume')
parser.add_argument('--save_epoch', type=int, default=10, metavar='N',
                    help='when to restore the model')
parser.add_argument('--save_model', action='store_true', default=False,
                    help='save_model or not')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--source', type=str, default='svhn', metavar='N',
                    help='source dataset')
parser.add_argument('--target', type=str, default='mnist', metavar='N', help='target dataset')
parser.add_argument('--use_abs_diff', action='store_true', default=False,
                    help='use absolute difference value as a measurement')
parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
print(args)


def main():
    solver = Solver(args, source=args.source, target=args.target, learning_rate=args.lr, batch_size=args.batch_size,
                    optimizer=args.optimizer, num_k=args.num_k, all_use=args.all_use,
                    checkpoint_dir=args.checkpoint_dir,
                    save_epoch=args.save_epoch)
    record_num = 1
    if args.source == 'usps' or args.target == 'usps':

        record_train = 'record/%s_%s_k_%s_alluse_%s_onestep_%s_%s.txt' % (
            args.source, args.target, args.num_k, args.all_use, args.one_step, record_num)
        record_test = 'record/%s_%s_k_%s_alluse_%s_onestep_%s_%s_test.txt' % (
            args.source, args.target, args.num_k, args.all_use, args.one_step, record_num)
        while os.path.exists(record_train):
            record_num += 1
            record_train = 'record/%s_%s_k_%s_alluse_%s_onestep_%s_%s.txt' % (
                args.source, args.target, args.num_k, args.all_use, args.one_step, record_num)
            record_test = 'record/%s_%s_k_%s_alluse_%s_onestep_%s_%s_test.txt' % (
                args.source, args.target, args.num_k, args.all_use, args.one_step, record_num)
    else:
        record_train = 'record/%s_%s_k_%s_onestep_%s_%s.txt' % (
            args.source, args.target, args.num_k, args.one_step, record_num)
        record_test = 'record/%s_%s_k_%s_onestep_%s_%s_test.txt' % (
            args.source, args.target, args.num_k, args.one_step, record_num)
        while os.path.exists(record_train):
            record_num += 1
            record_train = 'record/%s_%s_k_%s_onestep_%s_%s.txt' % (
                args.source, args.target, args.num_k, args.one_step, record_num)
            record_test = 'record/%s_%s_k_%s_onestep_%s_%s_test.txt' % (
                args.source, args.target, args.num_k, args.one_step, record_num)

    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)
    if not os.path.exists('record'):
        os.mkdir('record')
    if args.eval_only:
        solver.test(0)
    else:
        count = 0 
        A_st_n = 0.5
        J_w_n = 0.5
        max_Jw = 1
        min_Jw =1
        Ast_max = 0
        Ast_min = 0
        acc_m = 0
        for t in range(args.max_epoch):
            A_st_norm = A_st_n
            J_w_norm = J_w_n
            A_st_max= Ast_max
            A_st_min= Ast_min
            max_J_w = max_Jw
            min_J_w = min_Jw
            if not args.one_step:
                Ast_min ,Ast_max, min_Jw,max_Jw,A_st_n,J_w_n,num = solver.train(A_st_min,A_st_max, min_J_w,max_J_w,A_st_norm, J_w_norm,t, record_file=record_train)
                if t==0:
                    min_Jw = max_Jw
            else:
                num = solver.train_onestep(t, record_file=record_train)
            count += num
            
            acc_max = acc_m
            if t % 1 == 0:
                acc_m = solver.test(acc_max,t, record_file=record_test, save_model=args.save_model)
            if count >= 20000:
                break
            #if args.source == 'svhn' and t>=7:
            	#break
if __name__ == '__main__':
    main()
