import argparse

import torch.optim as optim
import torch.utils.data.sampler as sampler

from create_network import *
from create_dataset import *
from utils import *

parser = argparse.ArgumentParser(description='Single-task: Split')
parser.add_argument('--mode', default='none', type=str)
parser.add_argument('--port', default='none', type=str)

parser.add_argument('--gpu', default=0, type=int, help='gpu ID')
parser.add_argument('--seed', default=0, type=int, help='gpu ID')
parser.add_argument('--subset_id', default=0, type=int, help='mtan')

opt = parser.parse_args()

torch.manual_seed(opt.seed)
np.random.seed(opt.seed)
random.seed(opt.seed)

# create logging folder to store training weights and losses
if not os.path.exists('logging'):
    os.makedirs('logging')

# define model, optimiser and scheduler
device = torch.device("cuda:{}".format(opt.gpu) if torch.cuda.is_available() else "cpu")
model = MTLVGG16(num_tasks=1).to(device)
train_tasks = {'class_{}'.format(opt.subset_id): 5}

total_epoch = 200
optimizer = optim.SGD(model.parameters(), lr=0.1, weight_decay=5e-4, momentum=0.9)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, total_epoch)

# define dataset
trans_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]),
])

trans_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]),
])

train_set = CIFAR100MTL(root='dataset', train=True, transform=trans_train, subset_id=opt.subset_id)
test_set = CIFAR100MTL(root='dataset', train=False, transform=trans_test, subset_id=opt.subset_id)

batch_size = 32

train_loader = torch.utils.data.DataLoader(
    dataset=train_set,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_set,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4
)


# Train and evaluate multi-task network
print('Training Task: CIFAR-100 - {} in Single Task Learning Mode with VGG-16'.format(train_set.subset_class.title()))

train_batch = len(train_loader)
test_batch = len(test_loader)
train_metric = TaskMetric(train_tasks, train_tasks, batch_size, total_epoch, 'cifar100')
test_metric = TaskMetric(train_tasks, train_tasks, batch_size, total_epoch, 'cifar100')
for index in range(total_epoch):

    # evaluating train data
    model.train()
    train_dataset = iter(train_loader)
    for k in range(train_batch):
        train_data, train_target = train_dataset.next()
        train_data = train_data.to(device)
        train_target = train_target.to(device)

        train_pred = model(train_data, 0)

        optimizer.zero_grad()
        train_loss = F.cross_entropy(train_pred, train_target)
        train_loss.backward()
        optimizer.step()

        train_metric.update_metric([train_pred], {'class_{}'.format(opt.subset_id): train_target}, [train_loss])

    train_str = train_metric.compute_metric()
    train_metric.reset()

    # evaluating test data
    model.eval()
    with torch.no_grad():
        test_dataset = iter(test_loader)
        for k in range(test_batch):
            test_data, test_target = test_dataset.next()
            test_data = test_data.to(device)
            test_target = test_target.to(device)

            test_pred = model(test_data, 0)
            test_loss = F.cross_entropy(test_pred, test_target)

            test_metric.update_metric([test_pred], {'class_{}'.format(opt.subset_id): test_target}, [test_loss])

    test_str = test_metric.compute_metric()
    test_metric.reset()

    scheduler.step()

    print('Epoch {:04d} | TRAIN:{} || TEST:{} | Best: {} {:.4f}'
          .format(index, train_str, test_str, train_set.subset_class.title(),
                  test_metric.get_best_performance('class_{}'.format(opt.subset_id))))

    task_dict = {'train_loss': train_metric.metric, 'test_loss': test_metric.metric}
    np.save('logging/stl_cifar_{}_{}.npy'.format(opt.subset_id, opt.seed), task_dict)



