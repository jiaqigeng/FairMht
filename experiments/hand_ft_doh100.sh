cd src
python train.py hand --exp_id hand-detector-doh-full --gpus 0 --batch_size 8 --load_model '../exp/hand/hand-detector-doh-full/model_last.pth' --num_epochs 5 --lr 1e-5 --lr_step '15' --data_cfg '../src/lib/cfg/doh100.json' 
cd ..