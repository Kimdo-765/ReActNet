clear
python train.py  --batch_size=512 --learning_rate=1e-2 --epochs=256 --weight_decay=1e-5 --workers 16 | tee -a log/training.txt
