# LongWriter-V: Enabling Ultra-Long and High-Fidelity Generation in Vision-Language Models

<p align="center">
    🤗 <a href="https://huggingface.co/datasets/THU-KEG/LongWriter-V-22K" target="_blank">HF Repo</a> • 📃 <a href="https://arxiv.org/abs/2408.07055" target="_blank">Paper</a>
</p>

## 🔍 Table of Contents
- [⚙️ LongWriter-V Deployment](#deployment)
- [🤖️ AgentWrite](#agentwrite)
- [🖥️ Model Training](#longwriter-v-training)
- [📊 Evaluation](#evaluation)
- [👀 Cases](#case)
- [📝 Citation](#citation)

<a name="deployment"></a>
## ⚙️ LongWriter-V Deployment

**Environmental Setup**:
To inference Qwen2.5-VL based models, you may need to install transformers from source. Refer to this [issue](https://github.com/QwenLM/Qwen2.5-VL/issues/706) for more details.

We open-source three models: [LongWriter-V-7B](https://huggingface.co/THU-KEG/LongWriter-V-7B) and [LongWriter-V-7B-DPO](https://huggingface.co/THU-KEG/LongWriter-V-7B-DPO), trained based on [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) and [LongWriter-V-72B](https://huggingface.co/THU-KEG/LongWriter-V-72B), trained based on [Qwen2.5-VL-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct). 

<a name="agentwrite"></a>
## 🤖️ AgentWrite

We are also open-sourcing AgentWrite under `agentwrite/`, our automated ultra-long output data construction pipeline. Run `plan.py` and then `write.py` to obtain the final data. Please configure your API key in `config.json`.

<a name="longwriter-v-training"></a>
## 🖥️ Model Training

You can download and save the **LongWriter-V-22K** data through the Hugging Face datasets ([🤗 HF Repo](https://huggingface.co/datasets/THU-KEG/LongWriter-V-22K)).

You can train the model with [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), we used the [official Qwen2_VL training script](https://github.com/hiyouga/LLaMA-Factory/blob/main/examples/train_full/qwen2vl_full_sft.yaml) for training.

<a name="evaluation"></a>
## 📊 Evaluation
We introduce two evaluation benchmarks: **MMLongBench-Write** and **LongWrite-V-Ruler**. **MMLongBench-Write** focuses more on measuring the long output quality as well as the output length, while **LongWrite-V-Ruler** is designed as a light-weight stress test of the model's maximum output length.
We provide our evaluation data and code under `eval/`. Run
```bash
python -m eval.{mmlongbench_write, longwrite_v_ruler} --model {model_name} --method {vlm, caption_llm}
```
to get evaluation resuts. Remember to configure your OpenAI API key in `config.json` since we adopt GPT-4o as the judge.

Here are the evaluation results on **MMLongBench-Write**:
Here are the evaluation results on **LongWrite-V-Ruler**:


<a name="case"></a>
## 👀 Cases
Here are LongWriter-V-7B-DPO's outputs to random test prompts.

*User: Write a tragic love story about a lord's daughter falling in love with a servant, 5000 words.*
<details>
<summary>Assistant: (6176 words)</summary>
<div style="max-height: 200px; overflow-y: auto; padding: 10px; border: 1px solid #e1e4e8; border-radius: 6px;">

</div>
</details>
<br/>

*User: 写一篇10000字的中国旅游指南*
<details>
<summary>Assistant: (10691字)</summary>
<div style="max-height: 200px; overflow-y: auto; padding: 10px; border: 1px solid #e1e4e8; border-radius: 6px;">
</div>
</details>

<a name="citation"></a>
## 📝 Citation

If you find our work useful, please kindly cite:
