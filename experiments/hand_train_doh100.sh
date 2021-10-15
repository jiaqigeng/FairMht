cd src
python train.py hand --exp_id person-detector-doh-full --gpus 0 --batch_size 8 --load_model '../models/ctdet_coco_dla_2x.pth' --num_epochs 10 --lr_step '50' --data_cfg '../src/lib/cfg/doh100.json' 
cd ..