python ./tools/demo.py \
	--train_batch_size 8 \
	--val_batch_size 4 \
	--img_size 112 \
	--learning_rate 1e-2 \
	--num_epoches 20 \
	--step_size 5 \
	--learning_rate_dec 0.5 \
	--model_name Conv2fc1_avg_16 \
	--loss_name CrossEntropyLoss \
	--alg_name SGD \
	--data_root ./data/car/Data_hand \
	--train_list_file ./data/car/train.txt \
	--val_list_file ./data/car/val.txt \
	--label_list_file ./data/car/label.txt \
	--save_model_step 5 \
	--save_model_path ./out \
	--save_model_name car_Conv3fc2_1229 \
	--save_train_loss 1 \
	--save_train_path ./result 
#	--finetune 1 \
#	--finetune_model ./net/mobilenet.pth \
#	--finetune_layer_num 2 \
