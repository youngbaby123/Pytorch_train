python ./tools/train_demo.py \
	--train_batch_size 64 \
	--val_batch_size 32 \
	--img_size 112 \
	--learning_rate 1e-2 \
	--num_epoches 40 \
	--step_size 8 \
	--learning_rate_dec 0.5 \
	--model_name Conv2fc1_avg_16 \
	--loss_name CrossEntropyLoss \
	--alg_name SGD \
	--data_root ./data/car/Data_hand \
	--train_list_file ./data/car/train.txt \
	--val_list_file ./data/car/val.txt \
	--label_list_file ./data/car/label.txt \
	--save_model_step 8 \
	--save_model_path ./out \
	--save_model_name car_Conv3fc2_1229 \
	--save_train_loss 1 \
	--save_train_path ./result
#	--finetune_model ./net/mobilenet.pth \
#	--finetune_layer_num 2 \
#	--finetune 1 \


# 如果不需要预训练模型则将 --finetune_model --finetune_layer_num --finetune 注释掉
# 如果需要预训练模型初始化参数则 --finetune 1 (1表示使用预训练模型初始化) --finetune_layer_num 2 (2表示最后两层梯度更新，前面都不更新) --finetune_model 为预训练模型路径