Code for Paper:

G. Mai, Kai Cao, P. C. Yuen and Anil K. Jain, "On the Reconstruction of Face Images from Deep Face Templates", 
IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2019

Please cite our paper if you find the code useful. 

Clone the project from the github with 

git clone git@github.com:csgcmai/NBNet.git

Install the required packages, the usage of the conda enviornment is recommended.
Run the following commands

conda create -n nbnet python=2.7 anaconda
pip install --upgrade pip

Piror to the next step, please refer https://www.tensorflow.org/install/gpu 
and http://mxnet.io/install/index.html for installing the necessary packages 
for installing the tensorflow and mxnet with GPU support


cd NBNet
pip install -r requirement.txt

cd src
python train_of2img_mae.py --gpus 0,1,2,3

export PYTHONPATH=/home/liangcong/NBNet/src/util:$PYTHONPATH
python train_of2img_mae.py --gpus 0 --LRPPN_path /home/liangcong/dataset/Privacycheckpoints/new_FERG16_base/net_epoch_36_id_En_h.pth --data_dir /home/liangcong/dataset/FERG/ --model-save-prefix /home/liangcong/dataset/NBNet_checkpoint/HR_FERG/net --model-load-prefix /home/liangcong/dataset/NBNet_checkpoint/HR_FERG/net --is_HR --is_train




