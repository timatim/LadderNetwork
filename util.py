import numpy as np
import pickle
import torch
from torch.autograd import Variable

def add_noise(variable, mean=0, std=1):
	noise = Variable(torch.FloatTensor(variable.size()).normal_(mean, std))
	return variable.add(noise)

def pseudo_labeling(model):
	unlabeled_import = pickle.load(open("../data/train_unlabeled.p", 'rb'))
	# fill train labels with empty tensor for iterator
	unlabeled_import.train_labels = torch.ByteTensor(unlabeled_import.train_data.size()[0])

	unlabeled_loader = torch.utils.data.DataLoader(unlabeled_import, batch_size=64, shuffle=False)

	pseudolabels = np.array([])

	for data, target in unlabeled_loader:
		model.eval()
		data = Variable(data, volatile=True)
		output = model(data)
		temp = output.data.max(1)[1].numpy().reshape(-1)
		pseudolabels = np.concatenate((pseudolabels, temp))

	# add pseudolabels to unlabeled dataset
	unlabeled_import.train_labels = torch.from_numpy(pseudolabels).long()

	# dump to pickle
	pickle.dump(unlabeled_import, open("../data/train_unlabeled_pseudolabel1.p", 'wb'))

def test(epoch, valid_loader):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in valid_loader:

        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss /= len(valid_loader) # loss function already averages over batch size
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(valid_loader.dataset),
        100. * correct / len(valid_loader.dataset)))