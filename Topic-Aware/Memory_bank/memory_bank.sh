export CUDA_VISIBLE_DEVICES=1

data_path="/home/MSC/"
model_path="/home/data/public/bert-base-uncased/"
checkpoint_path="/home/data/public/topic_detector/checkpoint/"
output_dir="/home/MSC/processed/"

python main.py --data_path ${data_path} --model_path ${model_path} --checkpoint_path ${checkpoint_path} --output_dir ${output_dir}