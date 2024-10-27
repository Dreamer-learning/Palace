export CUDA_VISIBLE_DEVICES=1
backbone="llama" # choose llama or chatglm
model_path="/home/data/public/llama2"
log_name=${backbone} + "_vae_lora.txt"
save_path="/home/data/public"
data_path="" # after running memory bank
epoches=5
batch_size=32
vae_input_dim=4096
vae_h_dim=2048
vae_z_dim=4096

if [ "$backbone" == "llama" ]; then
    python train_vae_lora_llama.py --model_path ${model_path} --log_name ${log_name} --save_path ${save_path} --data_path ${data_path} --epoches ${epoches} --batch_size ${batch_size} --vae_input_dim ${vae_input_dim} --vae_h_dim ${vae_h_dim} --vae_z_dim ${vae_z_dim}
elif [ "$backbone" == "chatglm" ]; then
    python train_vae_lora_chatglm.py --model_path ${model_path} --log_name ${log_name} --save_path ${save_path} --data_path ${data_path} --epoches ${epoches} --batch_size ${batch_size} --vae_input_dim ${vae_input_dim} --vae_h_dim ${vae_h_dim} --vae_z_dim ${vae_z_dim}
else
    echo "unsupported backbone"
fi