from pymongo import MongoClient
from utils import LLM_Interface
import os
from tqdm import tqdm
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List

collection = MongoClient(
    port=27017
).maic["agenda"]

app = FastAPI()

class PolishRequest(BaseModel):
    imgs: List[str] = Field(..., description="List of base64 encoded images")
    scripts: List[str] = Field(..., description="List of corresponding scripts")
    course_name: str = Field(..., min_length=1, description="Name of the course")
    chapter_name: str = Field(..., min_length=1, description="Name of the chapter")

class PolishResponse(BaseModel):
    success: bool
    script: str | None = None
    error: str | None = None

@app.post("/polish", response_model=PolishResponse)
async def polish_endpoint(request: PolishRequest):
    try:
        if len(request.imgs) == 0 or len(request.scripts) == 0:
            raise ValueError("Images and scripts lists cannot be empty")
            
        if len(request.imgs) != len(request.scripts) + 1:
            raise ValueError("Number of images should be equal to number of scripts + 1")

        result = polish(
            imgs=request.imgs,
            scripts=request.scripts,
            course_name=request.course_name,
            chapter_name=request.chapter_name
        )
        return PolishResponse(success=True, script=result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

def get_online_scripts(chapter_id: str, output_dir):
    agendas = list(collection.find({"chapter_id": chapter_id}))
    for ag in agendas:
        if ag["agenda_type"] == "text":
            content = ag["content"]
            import re
            match = re.search(r'P(\d+):', content)
            if match:
                idx = match.group(1)
            else:
                idx = None
            assert idx, content
            
            functions = ag["function"]
            for f in functions:
                if f["call"] == "ReadScript":
                    value = dict(f["value"])
                    with open(f"{output_dir}/{idx}.txt", "w") as f:
                        f.write(value["script"])

def encode_images_to_base64(img_dir):
    import os
    import base64
    
    image_files = os.listdir(img_dir)
    image_files = sorted(image_files, key=lambda x: int(x.split("/")[-1].split('.')[0]))
    image_urls = []
    
    for filename in image_files:
        with open(os.path.join(img_dir, filename), 'rb') as f:
            image_data = f.read()
            image_url = base64.b64encode(image_data).decode('utf-8')
            image_urls.append(image_url)
    
    return image_urls

def plan(img_dir):
    prompt = """你是一名出色的教师。你的任务是根据PPT内容，为每个页面生成一个教学大纲。请以以下格式输出：
    [{
        "index": 1
        "agenda": "{本页的教学大纲}"
    },...]
    """
    image_urls = encode_images_to_base64(img_dir)
    
    response = LLM_Interface.call_gpt4v(
        messages=[
            dict(
                role="user",
                content=[
                    dict(
                        type="text",
                        text=prompt
                    )
                ] + [dict(
                    type="image_url",
                    image_url=dict(url=f"data:image/png;base64,{image_url}")
                ) for image_url in image_urls[20:30]],
            )
        ],
    )
    return response


def polish(imgs, scripts, course_name, chapter_name):
    prompt = """你是一名出色的教师。你需要根据聊天历史中的来自{}课程的{}章节的PPT的图片和讲稿对，来学习如何解读PPT并生成讲稿。然后，请你为当前PPT图片生成一段讲稿。请仔细观察示例中的讲稿是如何描述和解释PPT内容的，尽可能模仿示例的语言风格和解读方式，包括语气、详细程度、专业术语的使用等特点。并且保证讲稿的内容与PPT图片内容的一致性，可以流畅、连贯地衔接在上一张PPT的讲稿后面，在同一堂课中由老师在课堂上直接进行授课讲解；最后，讲稿需要有一定的启发性和教育意义，能够激发学生的学习兴趣和思考。请直接输出讲稿内容，不要输出任何其他内容。
    """.format(course_name, chapter_name)
    print(len(imgs), len(scripts))
    print(scripts)
    messages = []
    
    # Add example image-script pairs to chat history
    for script, img in zip(scripts, imgs[:-1]):
        messages.extend([
            dict(
                role="user",
                content=[
                    dict(type="image_url", image_url=dict(url=f"data:image/png;base64,{img}"))
                ]
            ),
            dict(
                role="assistant",
                content=script
            )
        ])
    
    # Add the target image that needs a new script
    messages.append(
        dict(
            role="user",
            content=[
                dict(type="text", text=prompt),
                dict(type="image_url", image_url=dict(url=f"data:image/png;base64,{imgs[-1]}"))
            ]
        )
    )

    response = LLM_Interface.call_gpt4v(messages=messages)
    return response


def get_old_scripts(txt_path, output_dir):
    import re
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the text file
    with open(txt_path, 'r') as f:
        content = f.read()
    
    # Find all script contents using regex

    pattern = r'script=(.*?)\)'
    matches = re.finditer(pattern, content)
    
    # Write each script to numbered files
    for i, match in enumerate(matches, 1):
        script = match.group(1)
        with open(f"{output_dir}/{i}.txt", "w") as f:
            f.write(script)

def get_scripts(txt_path):
    txt_files = os.listdir(txt_path)
    txt_files = sorted(txt_files, key=lambda x: int(x.split("/")[-1].split('.')[0]))
    scripts = []
    for txt_file in txt_files:
        with open(os.path.join(txt_path, txt_file), 'r') as f:
            scripts.append(f.read())
    return scripts

def polish_k_shot(k, imgs, scripts, course_name, chapter_name, output_dir):
    window_length = 10 
    history = scripts[:k]
    for i in range(k):
        with open(f"{output_dir}/{i+1}.txt", "w") as f:
            f.write(history[i])
    for i in tqdm(range(k, len(imgs))):
        with open(f"{output_dir}/{i+1}.txt", "w") as f:
            hist_imgs = imgs[:i][-min(window_length, len(imgs[:i+1])):]
            res = polish(hist_imgs + [imgs[i]], history, course_name, chapter_name)
            f.write(res)
            print(res, flush=True)
            history.append(res)
            history = history[-min(window_length, len(history)):]
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)