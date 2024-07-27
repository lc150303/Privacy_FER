# data organization
Save HR and downsampled images in the same dir.
```
>FERG
    >imgs
        >aia_anger_1.png
        >aia_anger_1_LR16_.png
        >aia_anger_2.png
        >...
    >train_ids_16.csv
    >test_ids_16.csv
```

In meta `csv` file:
```text
HR_image,exp,id,LR_image
```


# training 
Example:
```bash
python3 train.py --checkpoints_dir ./checkpoints --data_dir ./dataset/FERG --ids_file_suffix _16.csv --gpu_ids 0 --save_features 0 --save_model_freq 32 --batch_size 16 --n_threads_train 4 --n_threads_test 2 --expression_type 7 --subject_type 6 --HR_image_size 256 --nepochs_no_decay 12 --nepochs_decay 24 --lr_En 0.0005 --lr_C 0.0005 --lr_De 0.0005 --use_scheduler --L_cross 0.001 --L_adv 0.00000 --L_cls_sim 0.0001 --L_lir 0.1 --load_epoch 0 --name FERG_res_Gu --train_Gu_SC --train_Gu_LIR --resnet
```

# testing
Train the NBNet model to predict images (human faces) from the frozen En_l of our model.
Example:
```bash
python3 NBNet\src\train_of2img_mae.py --gpus 0 --model-save-prefix ./NBNet_checkpoint/128_MUG16/first --model-load-prefix ./NBNet_checkpoint/128_MUG16/first --batch-size 128 --LRPPN_path ./Privacycheckpoints/MUG16_full/net_epoch_48_id_En_l.pth --model-load-epoch 80 --data_dir ./MUG
```