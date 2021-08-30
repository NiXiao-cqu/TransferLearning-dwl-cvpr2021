from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
from utils import *
from taskcv_loader import CVDataLoader
from basenet import *
import torch.nn.functional as F
import os
from torch.nn.parallel import DataParallel
import scipy.io as sio 

#from aligned_reid.utils.utils import set_devices

# Training settings
parser = argparse.ArgumentParser(description='Visda Classification')
parser.add_argument('-d', '--sys_device_ids', type=eval, default=(0,))
parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=8, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--optimizer', type=str, default='momentum', metavar='OP',
                    help='the name of optimizer')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--num_k', type=int, default=1, metavar='K',
                    help='how many steps to repeat the generator update')
parser.add_argument('--num-layer', type=int, default=2, metavar='K',
                    help='how many layers for classifier')
parser.add_argument('--name', type=str, default='board', metavar='B',
                    help='board dir')
parser.add_argument('--save', type=str, default='/home/ps/xiaoni/DETD/Image/save_model/', metavar='B',
                    help='board dir')

parser.add_argument('--train_path', type=str, default='/home/ps/xiaoni/datasets/image_clef/i', metavar='B',
                    help='directory of source datasets')
parser.add_argument('--val_path', type=str, default='/home/ps/xiaoni/datasets/image_clef/p', metavar='B',
                    help='directory of target datasets')
parser.add_argument('--resnet', type=str, default='50', metavar='B',help='which resnet 18,50,101,152,200')
parser.add_argument('--task_name', type=str, default='i-p', metavar='B',help='domain1-domain2')
parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
#-------
#i-p-c

class TransferVarTensor(object):
  """Return a copy of the input Variable or Tensor on specified device."""

  def __init__(self, device_id=-1):
    self.device_id = device_id

  def __call__(self, var_or_tensor):
    return var_or_tensor.cpu() if self.device_id == -1 \
      else var_or_tensor.cuda(self.device_id)


class TransferModulesOptims(object):
  """Transfer optimizers/modules to cpu or specified gpu."""

  def __init__(self, device_id=-1):
    self.device_id = device_id

  def __call__(self, modules_and_or_optims):
    may_transfer_modules_optims(modules_and_or_optims, self.device_id)
def set_devices(sys_device_ids):
  """
  It sets some GPUs to be visible and returns some wrappers to transferring
  Variables/Tensors and Modules/Optimizers.
  Args:
    sys_device_ids: a tuple; which GPUs to use
      e.g.  sys_device_ids = (), only use cpu
            sys_device_ids = (3,), use the 4th gpu
            sys_device_ids = (0, 1, 2, 3,), use first 4 gpus
            sys_device_ids = (0, 2, 4,), use the 1st, 3rd and 5th gpus
  Returns:
    TVT: a `TransferVarTensor` callable
    TMO: a `TransferModulesOptims` callable
  """
  # Set the CUDA_VISIBLE_DEVICES environment variable
  import os
  visible_devices = '0'
  for i in sys_device_ids:
    visible_devices += '{}, '.format(i)
  os.environ['CUDA_VISIBLE_DEVICES'] = visible_devices
  # Return wrappers.
  # Models and user defined Variables/Tensors would be transferred to the
  # first device.
  device_id = 0 if len(sys_device_ids) > 0 else -1
  print(len(sys_device_ids))
  TVT = TransferVarTensor(device_id)
  TMO = TransferModulesOptims(device_id)
  return TVT, TMO


args = parser.parse_args()
sys_device_ids = args.sys_device_ids
TVT, TMO = set_devices(args.sys_device_ids)

os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_id
args.cuda = not args.no_cuda and torch.cuda.is_available()
train_path = args.train_path
val_path = args.val_path
num_k = args.num_k
num_layer = args.num_layer
batch_size = args.batch_size
# print('batch_size:',batch_size)
save_path = args.save+'_'+str(args.num_k)
cross_domain = args.task_name
class_num= 12
num_k1 = 8
num_k2 = 1
num_k3 = 8
num_k4 = 1
offset =0.1
output_cr_t_C_label = np.zeros(batch_size)

data_transforms = {
    train_path: transforms.Compose([
        transforms.Scale(256),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    val_path: transforms.Compose([
        transforms.Scale(256),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
dsets = {x: datasets.ImageFolder(os.path.join(x), data_transforms[x]) for x in [train_path,val_path]}
#dset_s_sizes = {x: len(dsets[x]) for x in [train_path]}
dset_s_sizes_dic = {x: len(dsets[x]) for x in [train_path]}
dset_t_sizes_dic = {x: len(dsets[x]) for x in [val_path]}
dset_s_sizes = list(dset_s_sizes_dic.values())[0]
dset_t_sizes = list(dset_t_sizes_dic.values())[0]
print("source_num")
print(dset_s_sizes)
print("target_num")
print(dset_t_sizes)
dset_classes = dsets[train_path].classes
print ('classes'+str(dset_classes))
use_gpu = torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
train_loader = CVDataLoader()
train_loader.initialize(dsets[train_path],dsets[val_path],batch_size)
dataset = train_loader.load_data()
test_loader = CVDataLoader()
opt= args
test_loader.initialize(dsets[train_path],dsets[val_path],batch_size,shuffle=True)
dataset_test = test_loader.load_data()
option = 'resnet'+args.resnet

G = ResNet_all(option)

#G_w = DataParallel(G)

C = Predictor()#(num_layer=num_layer)
C1 = Predictor()#(num_layer=num_layer)
C2 = Predictor()#(num_layer=num_layer)
D = AdversarialNetwork(2048)
gpus = args.gpu_id.split(',')
if len(gpus) > 1:
    G = nn.DataParallel(G, device_ids=[int(i) for i in gpus])
    C = nn.DataParallel(C, device_ids=[int(i) for i in gpus])
    C1 = nn.DataParallel(C1, device_ids=[int(i) for i in gpus])
    C2 = nn.DataParallel(C2, device_ids=[int(i) for i in gpus])
    D = nn.DataParallel(D, device_ids=[int(i) for i in gpus])
#D = Domain_discriminator()
C.apply(weights_init)
C1.apply(weights_init)
C2.apply(weights_init)
#D.apply(weights_init)
lr = args.lr


if args.cuda:
    G.cuda()
    C.cuda()
    C1.cuda()
    C2.cuda()
    D.cuda()
if args.optimizer == 'momentum':
    opt_g = optim.SGD(list(G.features.parameters()), lr=args.lr,weight_decay=0.0005)
    opt_c = optim.SGD(list(C.parameters()),momentum=0.9,lr=args.lr,weight_decay=0.0005)
    opt_c1c2 = optim.SGD(list(C1.parameters())+list(C2.parameters()),momentum=0.9,lr=args.lr,weight_decay=0.0005)
    opt_d = optim.SGD(list(D.parameters()),momentum=0.9,lr=args.lr,weight_decay=0.0005)
    
elif args.optimizer == 'adam':
    opt_g = optim.Adam(G.features.parameters(), lr=args.lr,weight_decay=0.0005)
    opt_c = optim.Adam(list(C.parameters()),lr=args.lr,weight_decay=0.0005)
    opt_c1c2 = optim.Adam(list(C1.parameters())+list(C2.parameters()), lr=args.lr,weight_decay=0.0005)
    opt_d = optim.Adam(list(D.parameters()),lr=args.lr,weight_decay=0.0005)

else:
    opt_g_cr = optim.Adadelta(G.features.parameters(), lr=args.lr,weight_decay=0.0005)
    opt_c = optim.Adadelta(list(C.parameters()),lr=args.lr,weight_decay=0.0005)
    opt_c1c2 = optim.Adadelta(list(C1.parameters())+list(C2.parameters()),lr=args.lr,weight_decay=0.0005) 
    opt_d = optim.Adadelta(list(D.parameters()),lr=args.lr,weight_decay=0.0005)
    

def reset_grad():
        opt_g.zero_grad()
        opt_c.zero_grad()
        opt_c1c2.zero_grad()
        opt_d.zero_grad()

def discrepancy(out1, out2):
        return torch.mean(torch.abs(F.softmax(out1) - F.softmax(out2)))

def props_to_onehot(props):
       props = props.cpu().detach().numpy()
       if isinstance(props, list):
          props = np.array(props)
       a = np.argmax(props, axis=1)
       b = np.zeros((len(a), props.shape[1]))
       b[np.arange(len(a)), a] = 1
       return torch.from_numpy(b)

def ent(output):
    out = -torch.mean(F.softmax(output.cuda() + 1e-6)*torch.log(F.softmax(output.cuda() + 1e-6)))
    out = Variable(out,requires_grad=True)
    return out

def linear_mmd(f_of_X, f_of_Y):
    loss = 0.0
    delta = f_of_X - f_of_Y
    loss = torch.mean(torch.mm(delta, torch.transpose(delta, 0, 1)))
    return loss


def train(num_epoch):
    criterion = nn.CrossEntropyLoss().cuda()
    adv_loss = nn.BCEWithLogitsLoss().cuda()
    #############balance+norm initial###############
    A_st_min = 0 
    A_st_max = 0
    min_J_w =1 #1e2
    max_J_w =1 # 0
    A_st_norm = 0.5
    J_w_norm = 0.5
    max_value =0
    acc = [0,0,0,0,0]
    ###############################################
    for ep in range(1,num_epoch):
        if ep ==2:
            min_J_w = max_J_w
        G.train()
        C.train()
        C1.train()
        C2.train()
        D.train()
        fea_for_LDA = np.empty(shape=(0,2048))
        fea_s_for_LDA = np.empty(shape=(0,2048)) 
        label_for_LDA = np.empty(shape=(0,1)) 
        label_s_for_LDA = []
        label_t_for_tSNE = []
  
        #################### A-DISTANCE initial end ###########################
        loss_mmd_all=0


        for batch_idx, data in enumerate(dataset):
            #if batch_idx * batch_size > 30000:
                #break
            if args.cuda:
                data1 = data['S']
                target1 = data['S_label']
                data2  = data['T']
                target2 = data['T_label']
                data1, target1 = data1.cuda(), target1.cuda()
                data2 = data2.cuda()#, target2.cuda()
            # when pretraining network source only
            eta = 1.0
            data = Variable(TVT(torch.cat((data1,data2),0)))
            target1 = Variable(TVT(target1))

##########################compute the T #############
            T_complex = A_st_norm/(A_st_norm+(1.0-J_w_norm))
            T = T_complex.real
        
################### source domain is discriminative.###################
            for i1 in range(num_k1):
                feat_cr_s = G(data1)
                output_cr_s_C = C(feat_cr_s.cuda())
            
                loss_cr_s = criterion(output_cr_s_C, target1)
            
                loss_1 = loss_cr_s 
                loss_1.backward()
                opt_g.step()
                opt_c.step()
                reset_grad()

#################### transferability ########################
            for i2 in range(num_k2):
                feat_cr_s = G(data1)
                feat_cr_t = G(data2)

                output_cr_s_D = D(feat_cr_s.cuda())
                output_cr_t_D = D(feat_cr_t.cuda())

                loss_D = nn.BCELoss()(output_cr_s_D, output_cr_t_D.detach())
                loss_2 = 0.1*loss_D
                loss_2.backward()
                opt_d.step()
                reset_grad()

##################### discriminablity ##################
                feat_cr_s = G(data1)
                feat_cr_t = G(data2)
            
                output_cr_s_C =  C(feat_cr_s.cuda())
                output_cr_s_C1 = C1(feat_cr_s.cuda())
                output_cr_s_C2 = C2(feat_cr_s.cuda()) 
                output_cr_t_C1 = C1(feat_cr_t.cuda())
                output_cr_t_C2 = C2(feat_cr_t.cuda())

                loss_cr_s = criterion(output_cr_s_C1, target1) + criterion(output_cr_s_C2, target1) + criterion(output_cr_s_C, target1)
                loss_dis1_t = -discrepancy(output_cr_t_C1, output_cr_t_C2)
                loss_3 = loss_cr_s + loss_dis1_t 
                loss_3.backward()
                opt_c1c2.step()
                opt_c.step()
                reset_grad()

################ the balance of transferability and discriminability ###########
            for i3 in range(num_k3):
                feat_di_s = G(data1)
                feat_di_t = G(data2)
                output_di_s_D = D(feat_di_s.cuda())
                output_di_t_D = D(feat_di_t.cuda())
                loss_bce = -nn.BCELoss()(output_di_s_D , output_di_t_D.detach()) 
                loss_1 = 0.2*loss_bce
                feat_cr_t = G(data2)
                output_cr_t_C = C(feat_cr_t.cuda())
                output_cr_t_C1 = C1(feat_cr_t.cuda())
                output_cr_t_C2 = C2(feat_cr_t.cuda())

                loss_41 = discrepancy(output_cr_t_C1, output_cr_t_C2)
                loss_42 = discrepancy(output_cr_t_C, output_cr_t_C1)
                loss_43 = discrepancy(output_cr_t_C, output_cr_t_C2)
                loss_4 = loss_41 + loss_42 + loss_43

                loss_balance = T*loss_1 + (1.0-T)*loss_4 #+ offset*(loss_1 + loss_4)
                loss_all = loss_balance
                loss_all.backward()
                opt_g.step()
                reset_grad()  
############ re-weiting based on the uncertainity of pseudo-label ###############
            # for i4 in range(num_k4):
            #     ############# H_emp
            #     feat_cr_t = G(data2)
            #     output_cr_t_C = C(feat_cr_t.cuda())
            #     output_cr_t_C_de = output_cr_t_C.detach()
            #
            #     for ii in range(batch_size):
            #         output_cr_t_C_label[ii] = np.argmax(output_cr_t_C_de[ii].cpu().numpy())
            #
            #     #print(self.output_cr_t_C_label)
            #     #H_emp = criterion(output_cr_t_C, output_cr_t_C.detach())
            #     output_cr_t_C_labels = torch.from_numpy(output_cr_t_C_label).cuda().long()
            #     #print('wwwwwwwwww')
            #     #print(output_cr_t_C_labels.dtype)
            #     Ly_ce_t = criterion(output_cr_t_C, output_cr_t_C_labels)
            #     H_emp = ent(output_cr_t_C)
            #     ############# weight coefficient mu
            #     mu = (torch.exp(-H_emp)-1.0/class_num)/(1-1.0/class_num)
            #     #Ly_loss_ce_t = mu*criterion(output_cr_t_C, output_cr_t_C.detach())
            #     Ly_loss = 0.5*(mu*Ly_ce_t+(1-mu)*H_emp) #+ 0.1*(Ly_ce_t + H_emp)
            #     Ly_loss.backward()
            #     opt_g.step()
            #     opt_c.step()
            #     reset_grad()

            reset_grad()
            
            feat_s = G(data1)
            feat_t = G(data2)
            #label_s
            label_predi = C(feat_t)
            ############## add for test tSNE  ###################
            #----------------------s-----------------------#
            label_s_test = target1
            feat_s_test = feat_s

            label_s_test_np = label_s_test.cpu().detach().numpy() 
            feat_s_test_np = feat_s_test.cpu().detach().numpy()
 
            label_s_for_LDA = np.concatenate((label_s_for_LDA,label_s_test_np),axis=0)
            fea_s_for_LDA = np.vstack((fea_s_for_LDA,feat_s_test_np))

            #-------------------------t --------------------#
            feat_test = feat_t
            feat_test_np = feat_test.cpu().detach().numpy()
            fea_for_LDA = np.vstack((fea_for_LDA,feat_test_np))
            #------------------------t_label-----------------#
            #-----target label------#
            label_test = label_predi

            label_t = torch.max(label_test,1)[1]
            #print(label_t1.shape)
            label_test_np = label_t.cpu().detach().numpy()

            label_test_np = label_test_np.reshape(batch_size,1)
            label_for_LDA = np.vstack((label_for_LDA,label_test_np))
            ###################### end    ####################

            if batch_idx == 1 and ep >0:  ############
                max_value,acc = test(ep,max_value,acc)
                G.train()
                C.train()
                C1.train()
                C2.train()
                D.train()

        ################## A_st ##############################
            if max(dset_s_sizes,dset_t_sizes)>= 2000:
                loss_mmd = linear_mmd(feat_s ,feat_t )
                A_st = loss_mmd.cpu().detach().numpy()
                A_st_max = max(abs(A_st_max),abs(A_st))
                A_st_min = min(abs(A_st_min),abs(A_st))
                A_st_norm = abs(A_st-A_st_min)/(A_st_max-A_st_min+1e-6)
        if max(dset_s_sizes,dset_t_sizes)<2000:
            f_of_X = torch.from_numpy(fea_s_for_LDA)
            f_of_Y = torch.from_numpy(fea_for_LDA)
            loss_mmd = linear_mmd(f_of_X ,f_of_Y )
            A_st = loss_mmd.cpu().detach().numpy()
            A_st_max = max(abs(A_st_max),abs(A_st))
            A_st_min = min(abs(A_st_min),abs(A_st))
            A_st_norm = abs(A_st-A_st_min)/(A_st_max-A_st_min+1e-6)

        ###################### J_w_s ####################################       
        n_dim = class_num-1
        clusters1 = np.unique(label_s_for_LDA)

        if n_dim > len(clusters1)-1:
           print("K is too much")
        Sw1 = np.zeros((fea_s_for_LDA.shape[1],fea_s_for_LDA.shape[1]))
        for i in clusters1:
    	    datai1 = fea_s_for_LDA[label_s_for_LDA.reshape(-1) == i]
    	    datai1 = datai1-datai1.mean(0)
    	    Swi1 = np.mat(datai1).T*np.mat(datai1)
    	    Sw1 += Swi1

	#between_class scatter matrix
        SB1 = np.zeros((fea_s_for_LDA.shape[1],fea_s_for_LDA.shape[1]))
        u1 = fea_s_for_LDA.mean(0)  #所有样本的平均值

        for i in clusters1:
    	    Ni1 = fea_s_for_LDA[label_s_for_LDA.reshape(-1) == i].shape[0]
    	    ui1 = fea_s_for_LDA[label_s_for_LDA.reshape(-1) == i].mean(0)  #某个类别的平均值
    	    SBi1 = Ni1*np.mat(ui1 - u1).T*np.mat(ui1 - u1)
    	    SB1 += SBi1

        S1= np.linalg.inv(Sw1+(1e-6*np.eye(Sw1.shape[0])))*SB1
        eigVals1,eigVects1 = np.linalg.eig(S1)  #求特征值，特征向量
        eigValInd1 = np.argsort(eigVals1)
        eigValInd1 = eigValInd1[:(-n_dim-1):-1]
        J_max1 = 0
        for i in range(n_dim):
    	    J_max1 = J_max1 + eigVals1[eigValInd1[i]]

        J_w_s = J_max1/len(clusters1)
        max_J_w = max(max_J_w,J_w_s)
        min_J_w = min(min_J_w,J_w_s)
        ###################### J_w_t ####################################       
        n_dim = class_num-1
        clusters = np.unique(label_for_LDA)

        if n_dim > len(clusters)-1:
            print("K is too much")
            
        Sw = np.zeros((fea_for_LDA.shape[1],fea_for_LDA.shape[1]))
        for i in clusters:
    	    datai = fea_for_LDA[label_for_LDA.reshape(-1) == i]
    	    datai = datai-datai.mean(0)
    	    Swi = np.mat(datai).T*np.mat(datai)
    	    Sw += Swi

	#between_class scatter matrix
        SB = np.zeros((fea_for_LDA.shape[1],fea_for_LDA.shape[1]))
        u = fea_for_LDA.mean(0)  #所有样本的平均值

        for i in clusters:
    	    Ni = fea_for_LDA[label_for_LDA.reshape(-1) == i].shape[0]
    	    ui = fea_for_LDA[label_for_LDA.reshape(-1) == i].mean(0)  #某个类别的平均值
    	    SBi = Ni*np.mat(ui - u).T*np.mat(ui - u)
    	    SB += SBi

        S = np.linalg.inv(Sw+(1e-6*np.eye(Sw.shape[0])))*SB
        eigVals,eigVects = np.linalg.eig(S)  #求特征值，特征向量
        eigValInd = np.argsort(eigVals)
        eigValInd = eigValInd[:(-n_dim-1):-1]
        J_max = 0
        for i in range(n_dim):
    	    J_max = J_max + eigVals[eigValInd[i]]

        J_w_t = J_max/len(clusters)
        max_J_w = max(max_J_w,J_w_t)
        min_J_w = min(min_J_w,J_w_t)
        J_w = min(J_w_s,J_w_t)
        J_w_norm = (J_w -min_J_w)/(max_J_w-min_J_w+1e-6)
        ################################################
        #return A_st_min, A_st_max, min_Jw_s_1,max_J_w_1, A_st_norm, J_w_1_norm, batch_idx

        #-----------------end----------------------#         





def test(epoch,max_value,acc):
    G.eval()
    C.eval()
    C1.eval()
    C2.eval()
    test_loss1 = 0
    test_loss2 = 0
    correct1 = 0
    correct2 = 0
    correct3 = 0
    correct4 = 0
    correct5 = 0
    size = 0

    for batch_idx, data in enumerate(dataset_test):
        #if batch_idx*batch_size > 5000:
            #break
        if args.cuda:
            data2  = data['T']
            target2 = data['T_label']
            #if val:
                #data2  = data['S']
                #target2 = data['S_label']
            data2, target2 = data2.cuda(), target2.cuda()
        data1, target1 = Variable(data2, volatile=True), Variable(target2)
        feat = G(data1)
        output1 = C(feat)
        output21 = C1(feat)
        output22 = C2(feat)
        test_loss1 += F.nll_loss(output1, target1).item()
        test_loss2 += F.nll_loss(output21, target1).item()
        output_ensemble_c1c2 = output21 + output22
        output_ensemble_cc1c2 = output1 + output21 + output22
        pred1 = output1.data.max(1)[1]
        pred21 = output21.data.max(1)[1]
        pred22 = output22.data.max(1)[1]
        pred_ensemble_c1c2 = output_ensemble_c1c2.data.max(1)[1]
        pred_ensemble_cc1c2 = output_ensemble_cc1c2.data.max(1)[1]
        k = target1.data.size()[0]
        correct1 += pred1.eq(target1.data).cpu().sum()
        correct2 += pred21.eq(target1.data).cpu().sum()
        correct3 += pred22.eq(target1.data).cpu().sum()
        correct4 += pred_ensemble_c1c2.eq(target1.data).cpu().sum()
        correct5 += pred_ensemble_cc1c2.eq(target1.data).cpu().sum()
        size += k
    test_loss1 = test_loss1 / size
    test_loss2 = test_loss2 / size
    acc1 = 100. * float(correct1) / float(size)
    acc2 = 100. * float(correct2) / float(size)
    acc3 = 100. * float(correct3) / float(size)
    acc4 = 100. * float(correct4) / float(size)
    acc5 = 100. * float(correct5) / float(size)

    value = max(acc1,acc2,acc3,acc4,acc5)
    if value>max_value:
       max_value = value
       acc[0],acc[1]= acc1, value
       print( '\n Epoch: {}, Accuracy C: {}/{} ({:.1f}%), Max Accuracy: ({:.1f}%)\n'.format(epoch, correct1, size, acc1, value))
    else:
       print( '\n Epoch: {}, Accuracy C: ({:.1f}%), Max Accuracy: ({:.1f}%)\n'.format(epoch, acc[0],acc[1]))
       
    #if not val and value > 60:
    # if value > 98:
    #     torch.save(G.state_dict(), save_path+'_'+cross_domain+'_'+str(epoch)+'_'+str(value10)+'_'+'G.pth')
    #     torch.save(C.state_dict(), save_path+'_'+cross_domain+'_'+str(epoch)+'_'+str(value10)+'_'+'C.pth')
    #     torch.save(C1.state_dict(), save_path+'_'+cross_domain+'_'+str(epoch)+'_'+str(value10)+'_'+'C1.pth')
    #     torch.save(C2.state_dict(), save_path+'_'+cross_domain+'_'+str(epoch)+'_'+str(value10)+'_'+'C2.pth')
    #     torch.save(D.state_dict(), save_path+'_'+cross_domain+'_'+str(epoch)+'_'+str(value10)+'_'+'D.pth')
    #
    return max_value,acc
#for epoch in range(1, args.epochs + 1):
train(args.epochs+1)

