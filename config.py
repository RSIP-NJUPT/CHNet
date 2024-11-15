#coding:utf-8
import warnings
warnings.filterwarnings('ignore')
import scipy.io as sio

class DefaultConfig(object):
	dataset = 3 #1-muufl 2-trento 3-huston 4-as  5-cut
	lr=1e-3
	module = "Rsnet"
	pca_components = 30
	epoches = 100		
	n = 5
	patch_size = 11
	RANK_ATOMS = 1
	NUM_CLUSTER = 1024
	BETA =  0.001
	note = "cls_params/"+str(pca_components)+"_"+str(NUM_CLUSTER)+"_"+str(patch_size)+"_"+str(lr)
	note_1 = "cls_params/"+str(pca_components)+"_"+str(NUM_CLUSTER)+"_"+str(patch_size)+"_"+str(lr)
	down_chennel = 67

	if dataset == 4:
		TOTAL_SIZE = 78294   # augsburg
		ALL_SIZE = 28983600
		class_num = 7
		data_HSI = sio.loadmat('data/Augsburg/data_HS_LR.mat')['data_HS_LR']
		data_lidar = sio.loadmat('data/Augsburg/data_DSM.mat')['data_DSM']
		labels = sio.loadmat('data/Augsburg/label_a.mat')['label']
		amount = [ 20, 20, 20, 20, 20, 20, 20]
		target_names = ['Forest','Residential area','Industrial area','Low plants','Allotment','Commercial area','Water']

opt = DefaultConfig()