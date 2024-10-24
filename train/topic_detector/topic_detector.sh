export CUDA_VISIBLE_DEVICES=1

train_dir="/home/liudongshuo/topic_shift/train_dataset.json"
test_dir="/home/liudongshuo/topic_shift/test_dataset.json"
epoches=2
lr=2e-5
model_path="/home/data/public/bert-base-uncased/"
output_dir="/home/liudongshuo/topic_shift/checkpoint"

python bert_cls.py --train_dir ${train_dir} --test_dir ${test_dir} --epoches ${epoches} --lr ${lr} --model_path ${model_path} --output_dir ${output_dir}