from inference.api import GPT_Interface
from utils import encode_images_to_base64
from PIL import Image
from .eval import eval_metrics
from pymongo import MongoClient
import os
from tqdm import tqdm

collection = MongoClient(
    port=27017
).maic["agenda"]


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


def encode_images_to_pil(img_dir):
    image_files = os.listdir(img_dir)
    image_files = sorted(image_files, key=lambda x: int(x.split("/")[-1].split('.')[0]))
    images = []
    
    for filename in image_files:
        images.append(Image.open(os.path.join(img_dir, filename)).convert("RGB"))
    
    return images

def plan(img_dir):
    prompt = """你是一名出色的教师。你的任务是根据PPT内容，为每个页面生成一个教学大纲。请以以下格式输出：
    [{
        "index": 1
        "agenda": "{本页的教学大纲}"
    },...]
    """
    image_urls = encode_images_to_base64(img_dir)
    
    response = GPT_Interface.call_gpt4o(
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


def polish(imgs, scripts):
    prompt = """You are an excellent teacher. Based on the chat history containing slide images and scripts, learn how to interpret slides and generate teaching scripts. Then, generate a script for the current slide image.

Please carefully observe how the example scripts describe and explain the slide content, and emulate their style of presentation, including tone, level of detail, use of technical terminology, and overall approach. 

Ensure that:
1. The script content accurately matches the slide content
2. It flows smoothly and connects logically with the script from the previous slide
3. It is suitable for direct classroom delivery
4. It has educational value and engages students by:
   - Encouraging critical thinking
   - Sparking interest in the subject
   - Making complex concepts accessible
   - Fostering active participation

Important: Generate the script in the same language as the slide content - use Chinese if the slides are in Chinese, or English if the slides are in English.

Please output only the script content, with no additional text or formatting.
    """
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

    response = GPT_Interface.call_gpt4o(messages=messages, use_cache=False)
    return response[0]


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


def polish_k_shot(k, imgs, scripts, course_name, chapter_name, output_dir, window_length=10):
    os.makedirs(output_dir, exist_ok=True)
    history = scripts[:k]
    results = []
    for i in range(k):
        with open(f"{output_dir}/{i+1}.txt", "w") as f:
            f.write(history[i])
        results.append(history[i])
    for i in tqdm(range(k, len(imgs))):
        hist_imgs = imgs[:i][-min(window_length, len(imgs[:i+1])):]
        res = polish(hist_imgs + [imgs[i]], history)
        with open(f"{output_dir}/{i+1}.txt", "w") as f:
            f.write(res)
        results.append(res)
        history.append(res)
        history = history[-min(window_length, len(history)):]
    return results


def extract_style(imgs, scripts):
    prompt = """你是一名出色的讲稿风格分析专家，请根据老师在课堂上讲课的PPT图片和讲稿，分析老师的讲课风格，包括语气、详细程度、专业术语的使用等特点。你需要尽可能详细地进行分析，并使你的分析结果可以用作生成讲稿的参考风格。请直接输出分析结果，不要输出任何其他内容。"""
    messages = []
    for img, script in zip(imgs, scripts):
        messages.extend([
            dict(role="user", content=[dict(type="image_url", image_url=dict(url=f"data:image/png;base64,{img}"))]),
            dict(role="assistant", content=script)
        ])
    messages.append(dict(role="user", content=prompt))
    response = GPT_Interface.call_gpt4o(messages=messages)
    return response

def classify_page_type(img):
    """Classify PPT page into types (1=title, 2=knowledge, 3=interactive) based on image content only"""
    messages = [
        dict(role="user", content=[
            dict(type="image_url", image_url=dict(url=img)),
            "Based on this PPT slide, classify it into one of three types and respond with ONLY the number:\n"
            "1 - Title/connecting page (large text, section headings, minimal content)\n"
            "2 - Knowledge page (detailed information, diagrams, bullet points)\n" 
            "3 - Interactive page (discussion questions, exercises, practice problems)\n\n"
            "Respond with just the number 1, 2, or 3."
        ])
    ]
    
    response, _, _ = GPT_Interface.call_gpt4o(messages=messages)

    print(response)
    
    # Extract just the number from response
    for char in response:
        if char in ['1','2','3']:
            return int(char)
    
    # Default to knowledge page if classification fails
    return 2

def classify_pages(imgs):
    """Classify all pages in a presentation, returning list of integers 1-3"""
    classifications = []
    for img in tqdm(imgs):
        page_type = classify_page_type(img)
        classifications.append(page_type)
    return classifications

