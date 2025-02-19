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

We are also open-sourcing AgentWrite under `agentwrite/`, our automated ultra-long output data construction pipeline. Run `plan.py` and then `write.py` to obtain the final data. Please configure your API key in `config.py`.

<a name="longwriter-v-training"></a>
## 🖥️ Model Training

You can download and save the **LongWriter-V-22K** data through the Hugging Face datasets ([🤗 HF Repo](https://huggingface.co/datasets/THU-KEG/LongWriter-V-22K)).

You can train the model with [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), we used the [official Qwen2_VL training script](https://github.com/hiyouga/LLaMA-Factory/blob/main/examples/train_full/qwen2vl_full_sft.yaml) for training.

<a name="evaluation"></a>
## 📊 Evaluation
We introduce two evaluation benchmarks: **MMLongBench-Write** and **LongWrite-V-Ruler**. **MMLongBench-Write** focuses more on measuring the long output quality as well as the output length, while **LongWrite-V-Ruler** is designed as a light-weight stress test of the model's maximum output length.
We provide our evaluation data and code under `eval/`. Run
```bash
python -m eval.mmlongbench_write --model {model_name} --method {vlm, caption_llm}
python -m eval.longwrite_v_ruler --model {model_name}
```
to get evaluation resuts. Remember to configure your OpenAI API key in `config.py` since we adopt GPT-4o as the judge.

Here are the evaluation results on **MMLongBench-Write**:
![image](https://github.com/user-attachments/assets/d4c7cce5-c48b-4bd0-9e9a-77cad06eae62)

Here are the evaluation results on **LongWrite-V-Ruler**:
![image](https://github.com/user-attachments/assets/f529b324-3ad5-4ddb-9c81-cbad59d1813b)


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

*User:
这张图片是一个经典电影中的场景，请分析这个经典场景，写一个此电影的长篇评论，要求1600字。*
<details>
<summary>Assistant: (1628字)</summary>
<div style="max-height: 200px; overflow-y: auto; padding: 10px; border: 1px solid #e1e4e8; border-radius: 6px;">
在探讨电影《当幸福来敲门》时，我们首先被其深刻的主题所吸引。影片聚焦于一个在困境中坚持追求梦想的普通人的故事，这不仅是一个关于成功的叙事，更是一段关于坚韧不拔和自我超越的旅程。影片通过主角的遭遇，探讨了关于成功、奋斗与坚持的意义，引发观众对自身目标和生活的深刻思考。

影片的背景设定在动荡的经济环境中，进一步增强了故事的现实感和紧迫感。在这样的背景下，主角面临着巨大的挑战，但他凭借无比的毅力和智慧，逐步实现了自己的梦想。这种设定不仅展示了人性在困境中的光辉，也激励我们在逆境中找到自己的方向。

影片中，角色的动机和目标非常明确，即通过金融市场的成功来改善家庭的经济状况和生活质量。这一目标贯穿整个故事，推动情节发展，使角色在追求过程中经历了一系列考验和成长。影片通过这种动机的实现，传递了关于成功与幸福的深刻见解。

总体而言，《当幸福来敲门》不仅是一部关于成功的电影，更是一次关于人性、梦想和坚持的深刻探讨。它提醒我们，真正的成功来源于不懈的努力和对梦想的执着追求。

在《当幸福来敲门》中，导演运用了多种技巧来增强影片的感染力。影片通过细腻的情感描绘和紧凑的剪辑节奏，成功地将观众带入主角的生活。导演巧妙地运用了近景和中景的切换，使得角色的情感变化更加直观。例如，在与儿子的对话场景中，镜头常常拉近到角色的面部，捕捉他们细微的表情变化，从而增强了观众的代入感。

影片中的音乐选择同样至关重要。背景音乐在关键时刻起到了画龙点睛的作用，尤其是在角色面临挑战时，音乐的激昂和紧张相呼应，增强了情绪的戏剧性。在一些情感深刻的画面中，音乐的停顿更是给观众带来了心灵上的冲击，使得情感表达更加深刻。

这些技巧的运用使得影片不仅仅是一个故事的讲述，而是一种情感的体验。观众在观影过程中，仿佛亲身经历了角色的喜怒哀乐，这正是影片成功的关键所在。

在《当幸福来敲门》的经典场景中，角色之间的对话成为了一种情感和智慧的较量。影片通过这种交流，不仅推动了情节的发展，还深刻揭示了人物的内心世界和价值观。在这一场景中，角色的对话充满了力量与决心，每一句话都仿佛在挑战命运的不公，表达对未来的坚定信念。

首先，角色的对话展现了他们之间的深厚情感。通过互相鼓励和支持，他们共同面对生活的艰难。这种情感纽带是影片的一大亮点，使观众感受到温暖和力量。其次，对话中蕴含的智慧令人深思。角色通过言辞传递出对梦想的坚持和对未来的希望，这种积极的人生态度感染了每一个观众。

此外，影片通过对话揭示了角色的内心冲突与成长。在面对挫折时，角色的选择和反应展现了他们的勇气和智慧，使人物形象更加立体和真实。这种情感与智慧的交织，不仅提升了影片的戏剧性，也为观众提供了深刻的思考。因此，这些对话不仅是情节的推动器，更是影片成功的关键所在。

在《当幸福来敲门》中，角色的成长与变化是影片的核心。影片通过细腻的情感描绘和紧凑的剪辑，展现了角色在追求梦想过程中的心理和行为转变。最初，角色被生活的重担压得喘不过气，但随着情节的发展，他逐渐找到了生活的意义和目标。这种成长不仅体现在外部行动上，更深入到内心深处的情感变化。

影片中的对话成为推动角色成长的重要力量。通过与他人的交流，角色逐渐认识到梦想实现的不易，以及坚持不懈的重要性。这种外部压力和内部反思的结合，促使角色在面对困难时展现出更加坚定和成熟的态度。

此外，影片还通过环境和情节的变化来衬托角色的成长。从最初的迷茫与无助到后来的自信与坚定，角色的成长与变化在每个细节中都得到了体现。这种多层次的成长描绘，不仅让观众感受到角色的转变，也引发了对自身梦想与生活的深刻思考。

在《当幸福来敲门》中，影片的结局设计既在意料之外，又在情理之中，完美地呼应了前面的故事情节。主角经过不懈的努力和坚持，终于实现了自己的梦想，这不仅是个人的胜利，更是对所有为梦想奋斗的人的鼓舞。影片通过这一结局，传递了关于努力、信念和成功的深刻信息。

影片的最后，角色站在梦想的顶峰，回顾这一路的艰辛与坚持，这一设计不仅令人动容，还引发了观众对自身目标和生活的思考。影片通过这样的结局，不仅完成了故事的闭环，还赋予了影片更深层次的意义和价值。

总体而言，《当幸福来敲门》以感人的结局为影片画上了圆满的句号，同时留给观众无限的思考和启发。这种结局不仅提升了影片的艺术感染力，也为观众提供了积极的人生启示。
</div>
</details>

<a name="citation"></a>
## 📝 Citation

If you find our work useful, please kindly cite:
