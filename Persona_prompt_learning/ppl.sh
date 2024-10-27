export CUDA_VISIBLE_DEVICES=1
backbone="llama" # choose llama or chatglm
model_path="/home/data/public/llama2"
checkpoint_path=""
log_name=${backbone} + "_ppl.txt"
save_path="/home/data/public"
data_path="" # after running memory bank
epoches=5
batch_size=32
vae_input_dim=4096
vae_h_dim=2048
vae_z_dim=4096
in_channels=4096
hidden_channels=2048
out_channels=4096
n_layers=2

if [ "$backbone" == "llama" ]; then
    python train_ppl_llama.py --model_path ${model_path} --checkpoint_path ${checkpoint_path} --log_name ${log_name} --save_path ${save_path} --data_path ${data_path} --epoches ${epoches} --batch_size ${batch_size} --vae_input_dim ${vae_input_dim} --vae_h_dim ${vae_h_dim} --vae_z_dim ${vae_z_dim} --in_channels ${in_channels} --hidden_channels ${hidden_channels} --out_channels ${out_channels}
elif [ "$backbone" == "chatglm" ]; then
    python train_ppl_chatglm.py --model_path ${model_path} --checkpoint_path ${checkpoint_path} --log_name ${log_name} --save_path ${save_path} --data_path ${data_path} --epoches ${epoches} --batch_size ${batch_size} --vae_input_dim ${vae_input_dim} --vae_h_dim ${vae_h_dim} --vae_z_dim ${vae_z_dim} --in_channels ${in_channels} --hidden_channels ${hidden_channels} --out_channels ${out_channels}
else
    echo "unsupported backbone"
fi