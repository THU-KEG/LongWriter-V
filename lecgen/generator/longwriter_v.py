from utils import extract_json
# from inference.local.qwen2_vl import get_model as get_qwen2_vl_model
# from inference.local.qwen2_5_vl import get_model as get_qwen2_5_vl_model
from inference.api.gpt import GPT_Interface, VllmServer_Interface

def find_header_slide(imgs):
    model = get_qwen2_5_vl_model('72b')
    prompt = """You are an expert teacher. Please analyze this page of a lecture and determine if it is a header slide. Output 1 if it is a header slide, 0 otherwise. Don't output any other text.
"""
    chunks = []
    chunk = []
    for img in imgs:
        messages = [
            {"role": "user", "content": [{"type": "image", "image": "file://" + img}, {"type": "text", "text": prompt}]}
        ]
        res = model.inference(messages)
        print(res)
    
    return chunks
    
def split_slides(imgs):
    if len(imgs) <= 30:
        return [{"start_slide": 1, "end_slide": len(imgs)}]

    prompt = """You are an expert teacher. Please analyze these lecture slides and split them into logical chapters of less than 30 slides each. Each chapter should cover a coherent topic or theme.

Output your response as JSON in this format:
{
    "chapters": [
        {
            "start_slide": 1,
            "end_slide": 25
        },
        ...
    ]
}

IMPORTANT:
- Each chapter must contain less than 30 slides
- Every slide must be included in exactly one chapter
- Chapter boundaries should align with natural topic transitions
- Output json only, no other text
"""

    messages = [
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": img}} for img in imgs] + [{"type": "text", "text": prompt}]}
    ]

    res = GPT_Interface.call(model='gpt-4o-2024-05-13', messages=messages, use_cache=True, temperature=0.7)

    res_json = extract_json(res)

    return res_json

def longwriter_v(imgs):
    import time
    start_time = time.time()

    # Split images into chunks of max 26 images
    chunks = []
    chunk = []
    for img in imgs:
        chunk.append(img)
        if len(chunk) == 26:
            chunks.append(chunk)
            chunk = []
    if chunk:  # Add remaining images if any
        chunks.append(chunk)
    
    # Generate scripts for each chunk and combine results

    system_msg = {
        "role": "system",
        "content": "You must respond with valid JSON. Each script should be a single line string without line breaks. All JSON properties must be enclosed in double quotes."
    }

    scripts = []
    for chunk in chunks:
        messages = [system_msg] + [
            dict(role="user", content=[
                dict(type="text", text=f"""Convert these lecture slides into a clear, engaging script that follows the slides' language and structure. Your output should be a JSON object where:
- Each key is a slide number (1 to {len(chunk)})
- Each value is that slide's corresponding script
- Scripts should include:
  - Clear explanations of concepts
  - Relevant examples and analogies
  - Natural transitions between topics
  - Engaging delivery style

REQUIREMENTS:
1. Generate EXACTLY {len(chunk)} scripts - one for each slide
2. Number scripts sequentially from 1 to {len(chunk)}
3. The script should be in the same language as the slides
4. Before responding, verify that:
   - You have the correct number of scripts ({len(chunk)})
   - All slide numbers from 1 to {len(chunk)} are present
   - Each script appropriately covers its slide's content
   - The script is in the same language as the slides
""")
            ] + [dict(type="image_url", image_url=dict(url=img)) for img in chunk])
        ]

        max_retry = 3
        
        for retry in range(max_retry):
            try:
                res = VllmServer_Interface.call(model="/home/test/test09/wyuc/model/trained/qwen/qwen2.5_vl-7b/dpo/mixed-1", messages=messages, max_tokens=8192, temperature=0.7, use_cache=False)

                print(res)    

                res_json = extract_json(res)
                
                if len(chunk) != len(res_json):
                    raise Exception(f"Failed to generate {len(chunk)} scripts")
                for i in range(1, len(chunk) + 1):
                    if str(i) not in res_json:
                        raise Exception(f"Missing script for slide {i}")

                break

            except Exception as e:
                print(f"Error: {e}")
                print(f"Retrying... ({retry+1}/{max_retry})")
                if retry == max_retry - 1:
                    raise Exception(f"Failed to generate scripts after {max_retry} retries")

        for key, value in res_json.items():
            scripts.append(value)

    total_time = time.time() - start_time
    print(f"Total time cost: {total_time:.2f} seconds")

    return scripts

if __name__ == "__main__":
    from utils import encode_image_to_base64, pptx_to_pdf, convert_pdf_to_png, encode_images_to_base64

    imgs = encode_images_to_base64("/home/test/test09/wyuc/code/LongWriter-V/buffer/1.1 自主学习原理/images")
    res = longwriter_v(imgs)
    print(res)
    
    