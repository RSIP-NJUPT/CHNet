#coding:utf-8
import warnings
warnings.filterwarnings('ignore')
import scipy.io as sio

class DefaultConfig(object):
	dataset = 3 #1-muufl 2-trento 3-huston 4-as
	lr = 3e-4
	module = "Rsnet"
	pca_components = 30
	epoches = 100		
	n = 3
	oa = 0
	topk = 0.9
	patch_size = 11
	RANK_ATOMS = 1
	NUM_CLUSTER = 1024
	BETA =  0.001
	note = "cls_result/"+str(dataset)+"_"+str(pca_components)+"_"+str(NUM_CLUSTER)+"_"+str(patch_size)+"_"+str(lr)+"_"+str(topk)
	note_1 = "cls_result/"+str(dataset)+"_"+str(pca_components)+"_"+str(NUM_CLUSTER)+"_"+str(patch_size)+"_"+str(lr)+"_"+str(topk)
	down_chennel = 67
	if dataset == 1:
		TOTAL_SIZE = 53687 #muufl
		ALL_SIZE = 71500
		class_num = 11
		data_HSI = sio.loadmat('/home/xyn/Documents/Trento_train/data/HSI.mat')['HSI'] 
		data_lidar = sio.loadmat('/home/xyn/Documents/Trento_train/data/LiDAR_DEM1.mat')['LiDAR']
		labels = sio.loadmat('/home/xyn/Documents/Trento_train/data/gt.mat')['GT']
		amount = [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20]
		# amount = [60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60]
		target_names = ['Trees', 'Mostly grass', 'Mixed ground surface', 'Dirt and sand', 'Road', 'Water','Building shadow','Building','Sidewalk','Yellow curb','Cloth panel']
	elif dataset == 2:
		TOTAL_SIZE = 30214  # trento
		ALL_SIZE = 99600
		class_num = 6
		data_HSI = sio.loadmat('/home/xyn/Documents/Trento_train/data/HSI_Trento.mat')['hsi_trento']
		data_lidar = sio.loadmat('/home/xyn/Documents/Trento_train/data/Lidar1_Trento.mat')['lidar1_trento']
		labels = sio.loadmat('/home/xyn/Documents/Trento_train/data/GT_Trento.mat')['gt_trento']
		amount = [20, 20, 20, 20, 20, 20]
		target_names = ['Apple Tree', 'Building', 'Ground', 'Wood', 'Vineyard', 'Roads']
	elif dataset == 3:
		TOTAL_SIZE = 15029   # huston
		ALL_SIZE = 664845
		class_num = 16
		data_HSI = sio.loadmat('/home/xyn/Documents/Trento_train/data/Houston2013/HSI.mat')['HSI']
		data_lidar = sio.loadmat('/home/xyn/Documents/Trento_train/data/Houston2013/LiDAR.mat')['LiDAR']
		labels = sio.loadmat('/home/xyn/Documents/Trento_train/data/Houston2013/gt.mat')['gt']
		amount = [ 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20 , 20, 20, 20, 20]
		# amount = [ 198, 190, 192, 188, 186, 182, 196, 191, 193, 191, 181 , 192, 184, 181, 187]
		target_names = ['Healthy grass', 'Streesed grass', 'Synthetic grass', 'Trees', 'Soil', 'Water','Residential','Commercial','Road','Highway','Railway','Park lot 1','Park lot 2','Tennis court','Running track']
	elif dataset == 4:
		TOTAL_SIZE = 78294   # augsburg
		ALL_SIZE = 28983600
		class_num = 7
		data_HSI = sio.loadmat('/home/xyn/Documents/Trento_train/data/Augsburg/data_HS_LR.mat')['data_HS_LR']
		data_lidar = sio.loadmat('/home/xyn/Documents/Trento_train/data/Augsburg/data_DSM.mat')['data_DSM']
		labels = sio.loadmat('/home/xyn/Documents/Trento_train/data/Augsburg/label_a.mat')['label']
		amount = [ 20, 20, 20, 20, 20, 20, 20]
		target_names = ['Forest','Residential area','Industrial area','Low plants','Allotment','Commercial area','Water']
opt = DefaultConfig()