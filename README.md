# Palace: A Persona-Aware LLM-Enhanced Framework for Multi-Session Personalized Dialogue Generation

Thank you for your interest in our work.

1. To reproduce our work, you first need to download the original dataset files and perform data processing, which can be found in the `/data` folder.

2. Then you need to run the `/Topic-Aware folder` to obtain the historical information from the dataset.

3. Next, perform VAE-LoRA training on different backbones, as detailed in the `VAE-LoRA` folder.

4. Finally, we train the persona prompt learning, which can be found in folder `\Persona_prompt_learning` 

For evaluation, you can use the `.generate` method in the code to generate the corresponding responses, and we use `nlg_eval` for assessment [Link](https://github.com/Maluuba/nlg-eval).




