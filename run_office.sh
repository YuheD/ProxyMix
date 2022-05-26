python train_source.py --dset office --s 0 --max_epoch 50
python train_source.py --dset office --s 1 --max_epoch 50
python train_source.py --dset office --s 2 --max_epoch 50


python train_target.py --dset office --easynum 5 --output test --gpu_id 7 --s 0 --t 1
python train_target.py --dset office --easynum 5 --output test --gpu_id 7 --s 0 --t 2
python train_target.py --dset office --easynum 5 --output test --gpu_id 7 --s 1 --t 0
python train_target.py --dset office --easynum 5 --output test --gpu_id 7 --s 2 --t 0
