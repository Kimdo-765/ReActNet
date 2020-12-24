clear
mkdir models
cp ../1_step1/models/checkpoint.pth.tar ./models/checkpoint_ba.pth.tar
mkdir log
python train.py  --batch_size=256 --learning_rate=1e-2 --epochs=256 --weight_decay=0 --workers=16 | tee -a log/training.txt
