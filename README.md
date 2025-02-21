# LongWriter-V: Enabling Ultra-Long and High-Fidelity Generation in Vision-Language Models

<p align="center">
    🤗 <a href="https://huggingface.co/datasets/THU-KEG/LongWriter-V-22K" target="_blank">HF Repo</a> • 📃 <a href="https://arxiv.org/abs/2502.14834" target="_blank">Paper</a>
</p>

## 🔍 Table of Contents
- [⚙️ LongWriter-V Deployment](#deployment)
- [🤖️ LongWriter-Agent-V](#agentwrite)
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
## 🤖️ LongWriter-Agent-V

We are also open-sourcing LongWriter-Agent-V under `agentwrite/`, our automated ultra-long output data construction pipeline. Run `outline_vlm.py` to obtain the final data. Please configure your API key in `config.py`.

<a name="longwriter-v-training"></a>
## 🖥️ Model Training

You can download and save the **LongWriter-V-22K** data through the Hugging Face datasets ([🤗 HF Repo](https://huggingface.co/datasets/THU-KEG/LongWriter-V-22K)).

You can train the model with [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), we used the [official Qwen2_VL training script](https://github.com/hiyouga/LLaMA-Factory/blob/main/examples/train_full/qwen2vl_full_sft.yaml) for training.

<a name="evaluation"></a>
## 📊 Evaluation
We introduce two evaluation benchmarks: [**MMLongBench-Write**](https://huggingface.co/datasets/THU-KEG/MMLongBench-Write) and [**LongWrite-V-Ruler**](https://huggingface.co/datasets/THU-KEG/LongWrite-V-Ruler). **MMLongBench-Write** focuses more on measuring the long output quality as well as the output length, while **LongWrite-V-Ruler** is designed as a light-weight stress test of the model's maximum output length.
We provide our evaluation code under `eval/`. Run
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
Here are LongWriter-V-7B's outputs to random test prompts.

*User: Write a 3200-word lecture script based on these ppt images. The script can be read aloud by a teacher in class and should not contain any unreadable content. The script should be consistent with the teaching goal of the ppt and not simply repeat the content on the ppt. It should be clear, structured, and have educational and inspirational value. Please output the script directly, without any extra content like subtitles or questions, and please again ensure that the script is readable.*

<details>    
    <summary>Images: (16 pages)</summary>
    <div style="max-height: 200px; overflow-y: auto; padding: 10px; border: 1px solid #e1e4e8; border-radius: 6px;">
        <img src="https://github.com/user-attachments/assets/ce3e5d7c-8671-4a4f-bb22-672306f3c4b3" alt="image" style="max-width: 100%; height: auto;">
        <img src="https://github.com/user-attachments/assets/5786371c-366f-4f49-aa28-1b3714d9acd7" alt="image" style="max-width: 100%; height: auto;">
        <img src="https://github.com/user-attachments/assets/a48c099e-ab22-4f33-acaf-5c9a78750907" alt="image" style="max-width: 100%; height: auto;">
        <img src="https://github.com/user-attachments/assets/db9be326-be66-4592-bc35-4743c1594c8c" alt="image" style="max-width: 100%; height: auto;">
        <img src="https://github.com/user-attachments/assets/3d07ad36-3581-4d66-a33d-6c440fa51054" alt="image" style="max-width: 100%; height: auto;">
        <img src="https://github.com/user-attachments/assets/cbf54b7e-a624-4696-9cb6-45bf814ffa16" alt="image" style="max-width: 100%; height: auto;">
        <img src="https://github.com/user-attachments/assets/15ebfb91-b6c2-4bf9-90f4-f5e980596a6b" alt="image" style="max-width: 100%; height: auto;">
        <img src="https://github.com/user-attachments/assets/edc5920c-c035-4a34-9616-f7b916c2e5ce" alt="image" style="max-width: 100%; height: auto;">
        <img src="https://github.com/user-attachments/assets/c5a50780-ab69-4d71-8e44-518404bb1b07" alt="image" style="max-width: 100%; height: auto;">
        <img src="https://github.com/user-attachments/assets/4d1f7df7-9bb4-431a-a08e-4b76a98e40f6" alt="image" style="max-width: 100%; height: auto;">
        <img src="https://github.com/user-attachments/assets/1106d05a-acd4-431f-a0cd-5074e94c726d" alt="image" style="max-width: 100%; height: auto;">
        <img src="https://github.com/user-attachments/assets/597c57a4-2a73-4873-87d0-f4cad4a8d21b" alt="image" style="max-width: 100%; height: auto;">
        <img src="https://github.com/user-attachments/assets/f1eeea6c-5703-45c7-a783-5d5b541c3e29" alt="image" style="max-width: 100%; height: auto;">
        <img src="https://github.com/user-attachments/assets/9934a09e-a253-4ad3-8ece-c6b96cbfa06f" alt="image" style="max-width: 100%; height: auto;">
        <img src="https://github.com/user-attachments/assets/9f15a9b5-e4e9-4516-8c24-d9a483b4b4f0" alt="image" style="max-width: 100%; height: auto;">
        <img src="https://github.com/user-attachments/assets/77ef6def-6012-46a1-ad54-79fc5c6e4560" alt="image" style="max-width: 100%; height: auto;">
    </div>
</details>
<details>


<summary>Assistant: (3164 words)</summary>
<div style="max-height: 200px; overflow-y: auto; padding: 10px; border: 1px solid #e1e4e8; border-radius: 6px;">
    
## Slide 1
Today, we explore the transformative potential of generative AI in K-12 education. This isn't just another tech trend; it's revolutionizing learning and teaching. Collaborating with MIT CSAIL and MIT Media Lab, our esteemed faculty, including Professors Hal Abelson, Cynthia Breazeal, Randall Davis, Prerna Ravi, Safinah Ali, and Kate Moore, are at the forefront of this innovation.

Generative AI offers educators unprecedented tools to personalize and enrich learning experiences. Imagine a world where every student receives tailored support, and learning becomes interactive and adaptive. This isn't just a vision—it's becoming a reality.

As we dive into these advancements, consider the impact on both teaching and learning. How can we harness this technology to bridge educational gaps and foster creativity? This isn't about whether AI should be used; it's about how we can employ it to empower students and teachers.

Join us on this journey as we examine the exciting possibilities that generative AI presents for education. Together, we'll explore how these innovations can reshape our classrooms for the better.
## Slide 2
Welcome back, everyone. Let's dive into today's agenda and dive into the exciting world of generative AI, building on our previous discussion. We'll kick off with community announcements and a sneak peek at our final project, setting the stage for an innovative session. 

We'll then dive into examples of generative AI in education, showcasing its transformative impact on teaching and learning. Our activities are designed for interactive engagement; we have a multi-dimensional post-it clustering activity that encourages collaborative thinking, a jigsaw activity to explore the learning sciences, and a brainstorming session to spark new ideas. 

As we wrap up, we'll reflect on how these tools can enrich our educational practices. Your involvement is crucial—these activities are opportunities to innovate and discover fresh perspectives. Let's seize this moment to explore how technology can elevate our educational mission. 

Stay open-minded, curious, and ready to embrace the possibilities. Let's have an amazing session together!
## Slide 3
Now, let's delve into the captivating realm of generative AI and its pivotal role in education. Generative AI goes beyond mere technology; it's a transformative tool with the power to revolutionize learning and teaching. But why is it so significant? 

Consider how educational challenges like content scarcity, individualized learning, and resource inefficiency pale in comparison to the potential of AI to personalize learning, enhance creativity, and make education more accessible. Generative AI offers dynamic solutions to these issues, crafting an educational landscape that is more engaging and effective for every student.

Imagine a world where AI assists in creating personalized learning paths, generates lesson plans, and even provides real-time feedback on assignments. This isn't just a vision—it's happening now. As we explore further, think about how we can harness this technology to address broader educational challenges. 

This journey into generative AI marks the beginning of an exciting exploration into its capabilities and potential impact on education. Let's approach this with curiosity and critically assess how these tools can enrich our educational practices and empower our students.
## Slide 4
As we delve into generative AI, it's vital to approach this rapidly evolving field with a critical and discerning eye. The landscape of generative AI is in a state of dynamic flux, with new advancements emerging almost daily. This technology holds immense promise but also demands careful scrutiny.

Why is it changing so swiftly? Recent breakthroughs, notably with models like GPT-4, are pushing the boundaries of what AI can achieve. These developments are both exhilarating and daunting, presenting both opportunities and challenges as we navigate this new era.

It's crucial to recognize that generative AI is not uniform. A range of applications, from creative pursuits to practical educational tools, is reshaping how we perceive and utilize these technologies. Yet, this diversity also creates complexity, with claims of capability often outpacing empirical evidence.

As educators and learners, we must ask critical questions: How can we harness the potential of generative AI to enhance our educational experiences? What are the limitations we must acknowledge? And how do we ensure ethical and effective implementation?

By fostering this spirit of inquiry, we lay the groundwork for engaging with generative AI in meaningful ways, preparing us to critically assess and navigate its ever-changing landscape.
## Slide 5
As we explore the realm of generative AI, it's essential to understand its historical context within disruptive technologies. A common thread in educational advancement is the impact of pioneering tools like cameras, personal computers, and iPhones. These innovations initially seemed unconventional but ultimately revolutionized society and education.

Cameras, once a niche invention, are now integral to photography and visual storytelling, fundamentally altering how we capture and share moments. Similarly, personal computers transformed the once foreign concept of computing into a ubiquitous part of our daily lives, democratizing information access and creativity.

The iPhone further exemplifies this trend, merging a phone with a powerful mobile device. It revolutionized how we interact with technology and each other, redefining communication and access to information.

In education, disruptive technologies have had a profound impact. Video players introduced a new way to consume content, calculators made complex calculations accessible, and the internet opened a vast knowledge repository. More recently, platforms like Wikipedia have challenged traditional publishing, making information more democratized.

As we consider generative AI, we see a technology following a similar disruptive path. It presents both challenges and opportunities for education, offering personalized learning experiences and innovative problem-solving approaches.

As we delve into these technologies, let's reflect on how they've reshaped education and ponder how generative AI can further transform our classrooms and learning environments.
## Slide 6
As we delve into the disruptive impact of technology in education, it's essential to consider both the benefits and challenges these innovations bring. Let's start with the positives. Disruptive technologies increase accessibility, providing diverse students with equal learning opportunities and resources. They open doors to personalized learning, catering to individual needs and styles, and introduce a variety of learning modalities that were once unimaginable.

Yet, with these advantages come challenges. The introduction of tools like ChatGPT has sparked discussions about misuse and ambiguity. These tools can be directed towards constructive learning, but they also open doors to cheating and bypassing critical thinking. Furthermore, technologies can create biases; for example, ChatGPT's responses are shaped by the data it was trained on, potentially reflecting biases in that information.

The short-term benefits often benefit only those with means, widening the gap between affluent and under-resourced schools. Additionally, technology is often seen as a universal solution, yet not all problems in education require technological fixes, especially when some challenges can be addressed more effectively through traditional methods.

As educators, it's vital to critically evaluate how technology fits into our educational strategies. We must navigate these opportunities and challenges with a deep understanding of their implications, ensuring technology enhances learning in meaningful ways. Let's consider how to integrate these tools thoughtfully, maximizing their benefits while mitigating potential harms.
## Slide 7
The explosive interest in generative AI for education, captured in this slide, highlights its rapidly growing relevance. From April to July 2023, the number of academic papers on generative AI surged dramatically—from around 1,400 to nearly 12,000. This fivefold increase underscores AI's transformative potential in education, showcasing its ability to tackle complex learning challenges and offer personalized experiences.

Yet, this excitement also brings scrutiny. As more studies emerge, so do questions about how generative AI can be effectively and ethically integrated into educational settings. The diverse opinions from educators, researchers, and policymakers reflect a field still defining its path.

This rapid growth presents a unique opportunity for innovation but also demands careful consideration. How can we harness these technologies to genuinely enhance learning outcomes? What are the ethical considerations in using AI to support students? As we delve deeper into this topic, these questions are essential to ponder, guiding us as we navigate this evolving landscape.

Let's approach these challenges with a critical eye, seeking to understand how generative AI can be a force for good in education. By doing so, we ensure that the technology's potential is fully realized for all students.
## Slide 8
As we delve deeper into the realm of generative AI in education, it is essential to examine the key research and resources that illuminate its potential and challenges. Heyam Abunaseer's comprehensive summary offers a vital exploration of how generative AI is reshaping educational landscapes, highlighting current applications and evaluating their impact. Abunaseer provides a balanced perspective on both the opportunities and obstacles, making her work a crucial resource for understanding the field's direction.

Equally significant is UNESCO's in-depth analysis of generative AI's impact on education. This resource is valued for its up-to-date insights and provides a realistic assessment of the opportunities and hurdles AI presents. UNESCO's work is essential for educators aiming to navigate this evolving terrain, offering a foundation for making informed decisions.

These research pieces serve as beacons for further exploration and inquiry. They encourage us to critically assess how AI can be harnessed to enhance educational equity and excellence. As we engage with these resources, a key question arises: How can we, as educators, creatively integrate generative AI into our practices while maintaining the core values of humanized learning?

By critically engaging with these materials, we can chart a course for effectively and ethically incorporating generative AI into education, ensuring it supports, rather than replaces, the essential principles of learning.
## Slide 9
The excitement surrounding generative AI in education is palpable, yet it's crucial to sift through the hype to identify its genuine educational applications. Various publications highlight AI's transformative potential, yet often oversimplify its complex intricacies. While AI presents innovative tools for personalized learning and creative expression, many reports risk reducing its impact to superficial integrations.

It's essential to critically evaluate these portrayals and seek out insights on both the challenges and benefits of AI implementation. Rohan Mehta's piece in MIT Technology Review cautions that banning AI like ChatGPT may do more harm than good, suggesting a measured approach is necessary. Similarly, The Guardian emphasizes AI's potential to revolutionize education, advocating for its thoughtful integration.

These diverse perspectives encourage us to consider AI as a versatile tool that can enhance educational experiences when used judiciously. As educators, it's vital to delve beyond surface-level applications and explore how AI can address deeper educational challenges, such as accessibility and equitable learning opportunities.

By engaging with these articles and forming our opinions, we can better prepare ourselves to effectively incorporate generative AI into our teaching and learning practices. Let's strive to understand its complexities and ensure that our approach aligns with the core principles of humanized education.
## Slide 10
As we delve into the realm of generative AI tools for education, several companies are at the forefront of innovation, pushing the boundaries of what's possible in teaching and learning. Let's examine some of these leaders and their contributions.

OpenAI, creators of chatGPT and DALL-E, have been instrumental in exploring AI's potential to revolutionize education. Their focus often includes personalized learning experiences and creative applications that cater to diverse learning styles.

Similarly, Meta's LLAMA and Google's Bard are platforms that underscore the importance of accessible and scalable AI tools, enabling educators to customize content and deliver it to students in innovative ways.

Microsoft's Bing and Midjourney are also making waves by offering AI-driven solutions that enhance content creation and personalization, empowering educators to craft engaging and interactive learning materials.

Hugging Face stands out by providing a community-driven platform with diverse models, allowing educators to select the most suitable tools for their specific needs, fostering a collaborative learning environment.

These companies are pivotal in developing AI tools that address varied educational contexts, making learning more personalized and effective. As we explore these advancements, we must consider how they can be integrated thoughtfully into our educational practices. This exploration sets the stage for understanding chatGPT's core characteristics, paving the way for deeper engagement with these transformative technologies.
## Slide 11
Now, let's delve into chatGPT, a pivotal example of generative AI. Its generative capability means it generates new, original content, complementing its pre-training, which involves being trained on vast datasets to grasp language patterns before being fine-tuned for specific tasks. This dual process equips chatGPT to engage in coherent conversations and perform complex language tasks.

The transformer architecture is crucial to chatGPT's functionality, allowing it to excel in language modeling by processing and understanding input to produce accurate, contextually appropriate outputs. This architecture enables chatGPT to tackle a variety of educational roles, from providing personalized feedback to creating dynamic content, making it an invaluable educational asset.

As we explore chatGPT's applications in education, consider how its generative, pre-trained, and transformer capabilities can be harnessed to enhance learning and teaching. This exploration will guide us in integrating these technologies thoughtfully into educational settings, capitalizing on their potential to enrich educational experiences.
## Slide 12
To truly understand chatGPT, we need to explore its core functionality in predicting the next word in a sentence, a task performed by its sophisticated algorithms. When you input a partial phrase, chatGPT analyzes the text and suggests words that best continue the thought, powered by deep learning models like transformers.

These models are trained on extensive datasets, enabling chatGPT to deliver responses that are coherent and contextually suitable. This word prediction algorithm is fundamental to chatGPT's capability to generate human-like text, making it incredibly useful in educational settings. For instance, it can assist students in writing or help teachers craft lesson plans.

The implications for education are profound. ChatGPT's ability to predict words supports language learning by enhancing vocabulary and comprehension skills. In classrooms, it can create dynamic, interactive learning experiences that cater to individual student needs.

As we delve deeper, consider how these algorithms can be leveraged to develop innovative educational tools that leverage chatGPT's capabilities to enrich the learning journey. Emphasizing critical thinking and creativity remains essential, as these tasks still require human oversight and insight.
## Slide 13
Now, let's explore how chatGPT operates, focusing on its training, reward system, and the model's interaction, which is pivotal to its functionality. The journey begins with collecting demonstration data to train a supervised policy, where human evaluators determine the desired outputs. This data, primarily scraped from the internet, forms the basis for the AI's learning.

Following this, a reward model is trained using comparison data, also evaluated by humans who rank responses to guide the AI's learning direction. This iterative process ensures the AI delivers more accurate and useful outputs over time.

In the final step, the model and reward model engage in a reciprocal training process. The reward model refines its capabilities, optimizing policies with the PPO reinforcement learning algorithm. As new prompts emerge, the system generates outputs based on the learned policies. Crucially, the reward output drives policy updates via PPO, ensuring continual enhancement.

While these algorithms generate plausible outputs, it's important to recognize they are not based on absolute truth. This distinction is essential for fostering critical thinking and encouraging students to focus on the message rather than the medium. As we incorporate chatGPT into education, grasping these mechanisms aids in harnessing its full potential while maintaining its role as a tool to complement, not replace, human judgment.
## Slide 14
Let's delve into the impressive capabilities of chatGPT across various exams, providing a glimpse into its potential as a study aid. According to OpenAI, chatGPT has scored in the 90th percentile on the Uniform Bar Exam, though independent research suggests a lower 68th percentile. Similarly, it performed competitively on the LSAT, SAT, and GRE, achieving scores that place it in the top tiers across multiple exams.

These results demonstrate chatGPT's proficiency in handling complex tasks and its potential to support educational endeavors. By generating detailed, relevant responses, it can assist students in crafting essays, understanding intricate concepts, and practicing exam strategies. Its ability to adapt to different question styles makes it a versatile tool for learners seeking additional support.

Yet, it's crucial to approach these capabilities with perspective. While chatGPT shows promise, it doesn't replace the need for critical thinking and critical analysis. Students must use it as a supplement, focusing on comprehension and critical engagement with the material. 

As we explore chatGPT's role in education, understanding its strengths and limitations helps us leverage its potential effectively, ensuring it enhances learning without replacing essential cognitive processes.
## Slide 15
ChatGPT showcases impressive skills in key areas like providing accurate information based on its training datasets, allowing for style customization in writing, and efficiently summarizing or paraphrasing text. These capabilities make it a valuable asset in educational settings, offering students a tool to enhance their writing and comprehension skills. Additionally, ChatGPT can assist by writing code, demonstrating its utility across various technical tasks.

Yet, it's important to recognize the current limitations. ChatGPT cannot deliver up-to-date information or news, as it relies on its initial dataset and doesn't incorporate real-time data. This limitation underscores the necessity for educators to keep the content current and relevant. Moreover, while ChatGPT can generate references, its current capabilities are limited, highlighting the importance of teachers guiding students in proper citation and research methods.

Navigating these strengths and limitations is crucial for educators and students. By understanding ChatGPT's capabilities, they can effectively use it as a tool to enhance learning and teaching, while also emphasizing the importance of critical thinking and independent research skills.
## Slide 16
In education, generative AI tools like chatGPT offer transformative potential for both teachers and students. For teachers, these tools simplify the creation of lesson plans, activities, and assessments, providing valuable feedback that can be tailored to individual student needs. Formative assessments and the ability to adjust assignment difficulty further customize educational experiences, allowing educators to nurture students' skills in generative AI itself, creating a feedback loop of continuous learning and innovation.

For students, chatGPT serves as an extensive, accessible resource that can be used for in-depth research and learning. It offers immediate feedback on assignments, helping students refine their understanding and skills. Additionally, it can simplify or translate complex language, making learning more accessible. It assists in generating seed ideas, debugging code, and providing step-by-step instructions for intricate tasks, empowering students to explore and correct their mistakes.

The educational potential of these tools is immense when used correctly. They can enhance critical thinking and creativity while supporting diverse learning styles. However, it's crucial to integrate these technologies thoughtfully, ensuring they complement traditional teaching methods and uphold academic integrity. By doing so, educators can help students thrive in an ever-evolving digital landscape, equipping them for future challenges.
</div>
</details>
<br/>

*User:
这张图片是一个经典电影中的场景，请分析这个经典场景，写一个此电影的长篇评论，要求1600字。*
<details>
    <summary>Images: (1 page)</summary>
    <div style="max-height: 200px; overflow-y: auto; padding: 10px; border: 1px solid #e1e4e8; border-radius: 6px;">
        <img src="https://github.com/user-attachments/assets/e9ac4f58-732e-494f-bb80-78db8f524e7e" alt="image" style="max-width: 100%;">
    </div>
</details>

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

```
@misc{tu2025longwriterv,
      title={LongWriter-V: Enabling Ultra-Long and High-Fidelity Generation in Vision-Language Models}, 
      author={Shangqing Tu and Yucheng Wang and Daniel Zhang-Li and Yushi Bai and Jifan Yu and Yuhao Wu and Lei Hou and Huiqin Liu and Zhiyuan Liu and Bin Xu and Juanzi Li},
      year={2025},
      eprint={2502.14834},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2502.14834}, 
}
```
