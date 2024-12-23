from lecgen.generator import GPT_Interface
from tqdm import tqdm
import os

def lecgen_caption(imgs, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    prompt_caption = """请尽可能详细地描述当前提供的图片中的内容，包括图片中的文字、图表、公式、图片中的内容等，请以markdown的格式输出，不要输出任何其他内容。"""
    captions = dict()
    for i in tqdm(range(len(imgs))):
        messages = [dict(role="user", content=[dict(type="text", text=prompt_caption)] + [dict(type="image_url", image_url=dict(url=f"data:image/png;base64,{imgs[i]}"))])]
        response = GPT_Interface.call_gpt4o(messages=messages)
        captions[str(i+1)] = response
        print(response)
    
    prompt = """你是一名出色的教师，你的任务是根据当前提供的PPT图片的描述，为第{i}页PPT生成一份讲稿。讲稿需要与PPT内容一致，可以连贯地衔接在上一页PPT的讲稿后面，在同一堂课中由老师在课堂上直接进行授课讲解；最后，讲稿需要有一定的启发性和教育意义，能够激发学生的学习兴趣和思考。请直接输出讲稿内容，不要输出任何其他内容。
    【PPT描述】{caption}
    请注意，你只需要生成第{i}页PPT的讲稿，不要生成其他PPT的讲稿。请直接输出讲稿内容，不要输出任何其他内容。"""
    history = []
    ctx_len = 5
    res = []
    for i in tqdm(range(len(imgs))):
        messages = [dict(role="user", content=prompt.format(i=i+1, caption=captions[str(i+1)]))]
        response = GPT_Interface.call_gpt4o(messages=history + messages)
        with open(f"{output_dir}/{i+1}.txt", "w") as f:
            f.write(response)
        print(response)
        messages.append(dict(role="assistant", content=response))
        history.extend(messages)
        if len(history) > ctx_len*2:
            history = history[-ctx_len*2:]
        res.append(response)
    return res