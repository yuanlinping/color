from model import *

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

test_file = '../dataset/test.txt'
infer_dir = './vis'

# test_file = '../dataset/test_1.txt'
# infer_dir = './vis_1'

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # ignore warning messages from tf package

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

		# infer loop
		n = len(im_paths)
		for i in xrange(n):
			# im = imread(im_paths[i], mode='RGB') / 255.
			im = np.genfromtxt(im_paths[i], delimiter=",")
			im[0] = 0.0
			im[-1] = 0.0
			maxV = max(im)
			minV = min(im)
			diffV = (maxV - minV) * 1.0
			for v in range(len(im)):
				im[v] = (im[v] - minV) / diffV

			gt = imread(gt_paths[i], mode='RGB') / 255.

			pred = model.inference(im, sess)

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

if __name__ == '__main__':
	tf.app.run()