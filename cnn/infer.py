from model import *

import math
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

test_file = '../dataset/test.txt'
infer_dir = './vis'

# test_file = '../dataset/test_1.txt'
# infer_dir = './vis_1'

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # ignore warning messages from tf package

def calculate_accuracy(gt, pred):
	bin_size = 32.
	height = len(gt)
	width = len(gt[1])

	total_num = height * width
	sum_corrects = 0


	for i in range(width):
		for j in range(height):
			color_gt = gt[j][i]
			color_pred = pred[j][i]
			con_1 = math.floor(round(color_gt[0] * 255) / bin_size) == math.floor(round(color_pred[0] * 255) / bin_size)
			con_2 = math.floor(round(color_gt[1] * 255) / bin_size) == math.floor(round(color_pred[1] * 255) / bin_size)
			con_3 = math.floor(round(color_gt[2] * 255) / bin_size) == math.floor(round(color_pred[2] * 255) / bin_size)
			if con_1 and con_2 and con_3:
				sum_corrects = sum_corrects + 1
			else:
				continue

	return sum_corrects * 10. / total_num / 10.

def main(_):
	if not os.path.exists(infer_dir):
		os.mkdir(infer_dir)

	model = Model()
	model.setup_inference()

	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	with tf.Session(config=config) as sess:
		sess.run(tf.group(tf.global_variables_initializer(),
						tf.local_variables_initializer()))	
		# restore parameters
		saver = tf.train.Saver([var for var in tf.trainable_variables()])							
		saver.restore(sess, save_path = tf.train.latest_checkpoint(model.logdir))

		# create input image paths
		paths = open(test_file, 'r').read().splitlines()
		im_paths = [p.split('\t')[0] for p in paths] # image 
		gt_paths = [p.split('\t')[1] for p in paths] # gt
 		save_paths = [p.split('/')[-1] for p in im_paths]
		vis_paths = [os.path.join(infer_dir, p.split('.png')[0]+'_legend_pred.png') for p in save_paths]
		paths_ID = [p.split('.png')[0] for p in save_paths]

		print paths_ID

		accuracy_array = []

		# infer loop
		n = len(im_paths)
		for i in xrange(n):
			# im = imread(im_paths[i], mode='RGB') / 255.
			im = np.genfromtxt(im_paths[i], delimiter=",")
			# im[0] = 0.0
			# im[-1] = 0.0
			# maxV = max(im)
			# minV = min(im)
			# diffV = (maxV - minV) * 1.0
			# for v in range(len(im)):
			# 	im[v] = (im[v] - minV) / diffV

			gt = imread(gt_paths[i], mode='RGB') / 255.

			pred = model.inference(im, sess)

			accuracy = calculate_accuracy(gt, pred)
			accuracy_array.append(accuracy)

			plt.clf()
			plt.subplot(211)
			plt.imshow(pred)
			plt.axis('off')
			plt.title('predict')
			plt.subplot(212)
			plt.imshow(gt)
			plt.axis('off')
			plt.title('ground truth')

			print 'Saving to {}'.format(vis_paths[i])
			# imsave(vis_paths[i], pred)
			plt.savefig(vis_paths[i])

		print accuracy_array




if __name__ == '__main__':
	tf.app.run()