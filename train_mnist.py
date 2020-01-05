import numpy as np
import struct
from PIL import Image

train_labels_filename = 'train-labels-idx1-ubyte'
test_labels_filename = 't10k-labels-idx1-ubyte'
train_images_filename = 'train-images-idx3-ubyte'
test_images_filename = 't10k-images-idx3-ubyte'
def read_labels_file(filename):
	with open(filename, 'rb') as labels_file:
		magic = struct.unpack('>i', labels_file.read(4))[0]
		assert 0x00000801 == magic, "magic number check failed, got {:08X}".format(magic)
		num_items = struct.unpack('>i', labels_file.read(4))[0]
		labels = np.fromfile(labels_file, dtype='B', count=num_items)

	return labels


def read_images_file(filename):
	with open(filename, 'rb') as images_file:
		magic = struct.unpack('>i', images_file.read(4))[0]
		assert 0x00000803 == magic, "magic number check failed, got {:08X}".format(magic)
		num_items = struct.unpack('>i', images_file.read(4))[0]
		rows = struct.unpack('>i', images_file.read(4))[0]
		cols = struct.unpack('>i', images_file.read(4))[0]
		images = np.fromfile(images_file, dtype='B', count=num_items*rows*cols).reshape(num_items, rows*cols)

	return images

def load_mnist():
	train_labels = read_labels_file(train_labels_filename)
	test_labels = read_labels_file(test_labels_filename)
	train_images = read_images_file(train_images_filename)
	test_images = read_images_file(test_images_filename)

	max_pixel_value = 255.0
	train_images = train_images / max_pixel_value
	test_images = test_images / max_pixel_value

	return (train_labels, train_images, test_labels, test_images)


def show_image(data):
	image = Image.fromarray(data, 'L')
	image.show()


def softmax(x):
	"""
	Expect a vector of logits, return a vector of same size normalized to a probability distribution.
	Assumes NxHW packing, sums axis 1
	"""
	expx = np.exp(x - np.max(x, axis=1, keepdims=True))
	y = expx / np.sum(expx, axis=1, keepdims=True)
	return y

def cross_entropy(y, y_hat, epsilon=0.000001):
	"""
	sum[ y_hat * log(y) ]
	returns large negative number when y is very small and y_hat is 1.
	TODO: negative sign?
	"""
	# y = np.amax(y, axis=(), initial=epsilon)
	# return np.sum(y_hat * np.log(y), axis=1)
	bsz = y_hat.shape[0]
	y_hat_argmax = y_hat.argmax(axis=1)
	log_probs = -np.log(y[range(bsz), y_hat_argmax])
	loss = np.sum(log_probs) / bsz

	return loss

def cross_entropy_softmax_backward(y, y_hat):
	"""
	Given vector of error per training example (size bsz)
	Compute partial derivative for each input logit
	input 'error' shape N
	return shape N, Logits
	"""
	# return y - y_hat
	bsz = y_hat.shape[0]
	grad = y
	grad[range(bsz), y_hat.argmax(axis=1)] -= 1
	grad = grad / bsz
	return grad

"""
TODOs and Debug steps
- *error in weight update?
- sum errors across batch at the top or sum dWs before update? (is this the same?)
- gradient check (numerical finite difference)
- check magnitue of weights, updates (ratio ~1e-3)
- check chance loss at the beginning
- data prep- mean subtract, scale?

Tried disabling update to inspect for chance loss.  got 3.2 != 2.3==-log(0.1).  Tried xavier initialization, made it worse (loss now 1000+).  
Scaled images /= 255.0, fixed chance loss.
Noticed with Xavier init the loss bounces around 2.26-2.35ish, with fixed 0.01 scale init, loss steadier around 2.30; sticking with xavier

"""
class Model(object):
	def __init__(self, hidden_size=100, logits=10, lr=0.1, bsz=128):
		input_size = 28*28
		self.lr = lr
		self.bsz = bsz
		self.W1 = np.empty((hidden_size, input_size), dtype=float)
		# self.b1 = np.empty((hidden_size), dtype=float)
		self.W2 = np.empty((logits, hidden_size), dtype=float)
		# self.b2 = np.empty((logits), dtype=float)


	def init(self, mean=0):
		"""
		TODO what is the best initalization to use?
		"""
		self.W1[:] = np.random.normal(loc=mean, scale=np.sqrt(2/np.sum(self.W1.shape)), size=self.W1.shape)
		# self.b1[:] = np.random.normal(loc=mean, scale=var, size=self.b1.shape)
		self.W2[:] = np.random.normal(loc=mean, scale=np.sqrt(2/np.sum(self.W2.shape)), size=self.W2.shape)
		# self.b2[:] = np.random.normal(loc=mean, scale=var, size=self.b2.shape)


	def forward(self, x):
		"""

		"""
		self.x = x
		self.actW1 = np.tensordot(self.W1 , x, axes=([1],[1])).transpose(1,0)
		# y = y + self.b1
		self.actRelu1 = np.amax(self.actW1, axis=(), initial=0)

		self.actW2 = np.tensordot(self.W2, self.actRelu1, axes=([1],[1])).transpose(1,0)
		# y = y + self.b2
		y = softmax(self.actW2)

		return y

	def gradcheck(self, y_hat):
		# check softmax, CE grad first
		x = self.actW2
		y = cross_entropy(softmax(x), y_hat)
		dx = cross_entropy_softmax_backward(y, y_hat)

		step = 1e-5
		for point in [(0,0), (5,12)]:
			import ipdb; ipdb.set_trace()



	def backward(self, y, y_hat):

		grad = cross_entropy_softmax_backward(y, y_hat)
		# grad shape (N, logits)
		# actRelu1 shape (N, hidden_size)
		# dW2 shape (N, hidden_size, logits)
		self.dW2 = np.einsum("nj,nk->jk", grad, self.actRelu1)
		
		dRelu1 = np.einsum("lh,nl->nh", self.W2, grad)
		dActW1 = np.greater(self.actRelu1, 0) * dRelu1
		self.dW1 = np.einsum("nj,nk->jk", dActW1, self.x)


	def update(self, error):
		"""
		Run SGD, apply update
		"""
		self.W2 -= self.lr * error * self.dW2
		self.W1 -= self.lr * error * self.dW1

def train_loader(train_data, train_labels, bsz):
	"""
	TODO neglects randomness per epoch
		and drops last partial batch
	"""
	for i in range(len(train_data)//bsz):
		start = bsz * i
		stop = start + bsz
		yield (train_data[start:stop, :], train_labels[start:stop])


def train_loop(epochs=10):
	(train_labels, train_images, test_labels, test_images) = load_mnist()

	model = Model()
	model.init()

	# debug for overfitting capacity
	# epochs = 10000
	# train_images = train_images[:128]
	# train_labels = train_labels[:128]
	for epoch in range(epochs):
		minibatch = 0
		for data, labels in train_loader(train_images, train_labels, bsz=model.bsz):
			y_hat = np.eye(10)[labels]

			y = model.forward(data)

			# model.gradcheck(y_hat)

			loss = cross_entropy(y, y_hat)
			print("Batch {}, sum** error {}".format(minibatch, loss))
			model.backward(y, y_hat)
			model.update(loss)
			minibatch += 1


if __name__ == "__main__":
	train_loop()