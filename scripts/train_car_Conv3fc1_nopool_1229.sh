python ./tools/demo.py \
	--train_batch_size 32 \
	--val_batch_size 8 \
	--img_size 112 \
	--learning_rate 1e-2 \
	--num_epoches 50 \
	--step_size 10 \
	--learning_rate_dec 0.5 \
	--model_name Conv3fc1_nopool \
	--loss_name CrossEntropyLoss \
	--alg_name SGD \
	--data_root /home/zkyang/Workspace/task/Pytorch_task/Pytorch_train/data/car/Data_hand \
	--train_list_file /home/zkyang/Workspace/task/Pytorch_task/Pytorch_train/data/car/train.txt \
	--val_list_file /home/zkyang/Workspace/task/Pytorch_task/Pytorch_train/data/car/val.txt \
	--label_list_file /home/zkyang/Workspace/task/Pytorch_task/Pytorch_train/data/car/label.txt \
	--save_model_step 10 \
	--save_model_path /home/zkyang/Workspace/task/Pytorch_task/Pytorch_train/out \
	--save_model_name car_Conv3fc1_nopool_1229 \
	--save_train_loss 1 \
	--save_train_path /home/zkyang/Workspace/task/Pytorch_task/Pytorch_train/result 
#	--finetune 1 \
#	--finetune_model /home/zkyang/Workspace/task/Pytorch_task/Pytorch_train/net/mobilenet.pth \
#	--finetune_layer_num 2 \
