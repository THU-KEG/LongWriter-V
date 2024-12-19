from inference.api import GPT_Interface
from tqdm import tqdm
import os

def lecgen_iter(imgs, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    prompt = """你是一名出色的教师，你的任务是为当前提供的全部PPT图片中的第一张图片，生成一段讲稿。
    1. 这是一个迭代的过程，在历史消息中可能会有之前几张PPT已经生成的讲稿，所以你需要考虑前后的衔接，使整篇讲稿可以由老师在课上直接流畅地朗读讲授；
    2. 你需要整体地考虑所有PPT内容，包括之前已经生成的若干张PPT，还有本次给出的若干张PPT，全盘地考虑如何写当前这张PPT的讲稿，保证讲授内容突出重点，不要有过多的前后重复，保证不要有前后矛盾的地方。你需要生成的讲稿仅仅是本次提供的PPT图片中的第一张，不要生成前后其他PPT的讲稿。
    3. 你需要保证讲稿的内容与第一张PPT图片内容的一致性，可以流畅、连贯地衔接在上一张PPT的讲稿后面，在同一堂课中由老师在课堂上直接进行授课讲解；最后，讲稿需要有一定的启发性和教育意义，能够激发学生的学习兴趣和思考。请直接输出讲稿内容，不要输出任何其他内容。
    再次提醒，你只需要生成第一张PPT的讲稿，不要生成其他PPT的讲稿；请直接输出讲稿内容，不要输出任何其他内容。"""
    res = []
    for i in tqdm(range(len(imgs))):
        messages = []
        for img, r in zip(imgs[:i], res[:i]):
            messages.extend([
                dict(role="user", content=[dict(type="image_url", image_url=dict(url=f"data:image/png;base64,{img}"))]),
                dict(role="assistant", content=r)
            ])
        messages.append(dict(role="user", content=[dict(type="text", text=prompt)] + [dict(type="image_url", image_url=dict(url=f"data:image/png;base64,{img}")) for img in imgs[i:]]))
        response = GPT_Interface.call_gpt4o(messages=messages)
        with open(f"{output_dir}/{i+1}.txt", "w") as f:
            f.write(response)
        print(response)
        res.append(response)
    return res