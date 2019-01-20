from model import *

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import math


infer_dir = './vis'

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # ignore warning messages from tf package


def calculate_accuracy(gt, pred):
	bin_size = 32.
	channel = 3

	total_num = len(gt) / channel
	sum_corrects = 0

	for i in range(0, total_num, channel):
			con_1 = math.floor(round(gt[i] * 255) / bin_size) == math.floor(round(pred[i] * 255) / bin_size)
			con_2 = math.floor(round(gt[i+1] * 255) / bin_size) == math.floor(round(pred[i+1] * 255) / bin_size)
			con_3 = math.floor(round(gt[i+2] * 255) / bin_size) == math.floor(round(pred[i+2] * 255) / bin_size)
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
		vis_paths = [os.path.join(infer_dir, p.split('.png')[0]+'_legend_pred.csv') for p in save_paths]

		# infer loop
		accuracy_array = []
		n = len(im_paths)
		for i in xrange(n):
			# im = imread(im_paths[i], mode='RGB') / 255.
			im = imread(im_paths[i], mode='RGB') / 255.
			gt0 = np.genfromtxt(gt_paths[i], delimiter=",")

			gt = gt0 / 255.
			pred = model.inference(im, sess)

			accuracy_array.append(calculate_accuracy(gt, pred))

			for j in range(len(pred)):
				pred[j] = round(pred[j] * 255)

			res = open(vis_paths[i],'w')
			a = np.array([gt0,pred],dtype=np.int32)
			np.savetxt(res, a, delimiter=",", fmt="%d")


			# print >> res, gt0
			# print >> res,pred
			# res.close()
			#
			#
			#
			#
			# plt.clf()
			# plt.subplot(211)
			# plt.imshow(pred)
			# plt.axis('off')
			# plt.title('predict')
			# plt.subplot(212)
			# plt.imshow(gt)
			# plt.axis('off')
			# plt.title('ground truth')

			print 'Saving to {}'.format(vis_paths[i])
			# imsave(vis_paths[i], pred)
			# plt.savefig(vis_paths[i])
		print accuracy_array

if __name__ == '__main__':
	tf.app.run()