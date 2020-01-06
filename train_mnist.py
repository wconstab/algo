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


def numerical_grad_check(x, dx_analytical, f_x, fail_tolerance=1e-3, warn_tolerance=1e-4, step=1e-5):
	"""
	http://cs231n.github.io/neural-networks-3/#gradcheck
	pass in f_x, a lambda that just takes x.
	"""
	x_plus = np.array(x, dtype=np.float64)
	x_minus = np.array(x, dtype=np.float64)
	for point in [(0, 0), (5, 2), (13, 3), (25, 5), (127, 9)]:
		x_plus[point] += step
		x_minus[point] -= step
		df_x_plus = f_x(x_plus)
		df_x_minus = f_x(x_minus)
		dx_numerical = (df_x_plus - df_x_minus) / (2 * step)
		relative_diff = (dx_numerical - dx_analytical[point]) / max(abs(dx_numerical), abs(dx_analytical[point]))
		assert relative_diff < fail_tolerance, "Failed gradient check at point {}, dx_numerical {}, dx {}, relative_diff {}".format(point, dx_numerical, dx_analytical[point], relative_diff)
		if relative_diff > warn_tolerance:
			print("Warning: point {}, dx_numerical {}, dx {}, relative_diff {}".format(point, dx_numerical, dx_analytical[point], relative_diff))
		x_plus[point] = x[point]
		x_minus[point] = x[point]


"""
TODOs and Debug steps
- *error in weight update?
- sum errors across batch at the top or sum dWs before update? (is this the same?)
- gradient check (numerical finite difference)
- check magnitue of weights, updates (ratio ~1e-3)
- check chance loss at the beginning
- data prep- mean subtract, scale?
- summing over batch dim of grads at random places during backward() - something is probably wrong there

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
		self.b1 = np.zeros((hidden_size), dtype=float)
		self.W2 = np.empty((logits, hidden_size), dtype=float)
		self.b2 = np.zeros((logits), dtype=float)


	def init(self, mean=0):
		"""
		Using Glorot initialization
		"""
		self.W1[:] = np.random.normal(loc=mean, scale=np.sqrt(2/np.sum(self.W1.shape)), size=self.W1.shape)
		self.W2[:] = np.random.normal(loc=mean, scale=np.sqrt(2/np.sum(self.W2.shape)), size=self.W2.shape)


	def forward(self, x):
		"""
		"""
		self.x = x
		self.actW1 = np.tensordot(self.W1 , x, axes=([1],[1])).transpose(1,0)
		self.actb1 = self.actW1 + self.b1
		self.actRelu1 = np.amax(self.actb1, axis=(), initial=0)

		self.actW2 = np.tensordot(self.W2, self.actRelu1, axes=([1],[1])).transpose(1,0)
		self.actb2 = self.actW2 + self.b2
		y = softmax(self.actb2)

		return y





	def gradcheck_softmax_ce(self, y_hat, fail_tolerance=1e-3, warn_tolerance=1e-4):
		# check softmax, CE grad first
		x = self.actW2.astype(np.float64)
		y = softmax(x)
		# loss = cross_entropy(y, y_hat)
		dx = cross_entropy_softmax_backward(y, y_hat)

		f_x = lambda _x : cross_entropy(softmax(_x), y_hat)

		numerical_grad_check(x, dx, f_x)
		# step = 1e-5
		# for point in [(0, 0), (5, 2), (13, 3), (25, 5), (127, 9)]:
		# 	x_plus = np.array(x)
		# 	x_minus = np.array(x)
		# 	x_plus[point] += step
		# 	x_minus[point] -= step
		# 	df_x_plus = cross_entropy(softmax(x_plus), y_hat)
		# 	df_x_minus = cross_entropy(softmax(x_minus), y_hat)
		# 	dx_numerical = (df_x_plus - df_x_minus) / (2 * step)
		# 	relative_diff = (dx_numerical - dx[point]) / max(abs(dx_numerical), abs(dx[point]))
		# 	assert relative_diff < fail_tolerance, "Failed gradient check at point {}, dx_numerical {}, dx {}, relative_diff {}".format(point, dx_numerical, dx[point], relative_diff)
		# 	if relative_diff > warn_tolerance:
		# 		print("Warning: point {}, dx_numerical {}, dx {}, relative_diff {}".format(point, dx_numerical, dx[point], relative_diff))

	def backward(self, y, y_hat):

		grad = cross_entropy_softmax_backward(y, y_hat)
		# grad shape (N, logits)
		# actRelu1 shape (N, hidden_size)
		# dW2 shape (N, hidden_size, logits)
		self.dW2 = np.einsum("nj,nk->jk", grad, self.actRelu1)
		self.db2 = np.sum(grad, axis=0)
		
		dRelu1 = np.einsum("lh,nl->nh", self.W2, grad)
		dActW1 = np.greater(self.actRelu1, 0) * dRelu1
		self.dW1 = np.einsum("nj,nk->jk", dActW1, self.x)
		self.db1 = np.sum(dActW1, axis=0)


	def update(self, error):
		"""
		Run SGD, apply update
		"""
		self.W2 -= self.lr * error * self.dW2
		# self.b2 -= self.lr * error * self.db2
		self.W1 -= self.lr * error * self.dW1
		# self.b1 -= self.lr * error * self.db1


def train_loader(train_data, train_labels, bsz):
	"""
	TODO neglects randomness per epoch
		and drops last partial batch
	"""
	for i in range(len(train_data)//bsz):
		start = bsz * i
		stop = start + bsz
		yield (train_data[start:stop, :], train_labels[start:stop])


def evaluate(model, test_data, test_labels):
	predictions = model.forward(test_data)
	correct = np.argmax(predictions, axis=1) == test_labels
	return np.sum(correct) / test_labels.shape[0]


def train_loop(epochs=10):
	(train_labels, train_images, test_labels, test_images) = load_mnist()

	model = Model()
	model.init()

	# debug for overfitting capacity
	# epochs = 10000
	# train_images = train_images[:128]
	# train_labels = train_labels[:128]

	accuracy = evaluate(model, test_images, test_labels)
	print("random init test accuracy {}".format(accuracy))

	for epoch in range(epochs):
		minibatch = 0
		for data, labels in train_loader(train_images, train_labels, bsz=model.bsz):
			y_hat = np.eye(10)[labels]

			y = model.forward(data)

			model.gradcheck_softmax_ce(y_hat)

			loss = cross_entropy(y, y_hat)
			# print("Batch {}, sum** error {}".format(minibatch, loss))
			model.backward(y, y_hat)
			model.update(loss)
			minibatch += 1

		accuracy = evaluate(model, test_images, test_labels)
		print("Epoch {}, test accuracy {}".format(epoch, accuracy))
	import pdb; pdb.set_trace()


if __name__ == "__main__":
	train_loop()