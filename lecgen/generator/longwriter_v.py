from utils import extract_json
from inference.local.qwen2_vl import get_model as get_qwen2_vl_model

def longwriter_v(imgs):
    system_msg = {
        "role": "system",
        "content": "You must respond with valid JSON. Each script should be a single line string without line breaks. All JSON properties must be enclosed in double quotes."
    }

    messages = [system_msg] + [
        dict(role="user", content=[
            dict(type="text", text="""Convert the lecture slides into a structured script, using the same language as the slides. Output a JSON where each key is a slide number and its value is the corresponding lecture script. Include detailed explanations, examples, and analogies to help students understand complex topics.
IMPORTANT: You MUST generate exactly one script for EACH slide image provided. The number of scripts in your output MUST MATCH the number of input slides EXACTLY. Do not skip any slides. Check that every slide has a corresponding script before responding.

For example, if there are 10 slides, your output must have exactly 10 entries with keys "1" through "10". Missing or extra scripts are not acceptable.""")
        ] + [dict(type="image_url", image_url=dict(url=img)) for img in imgs])
    ]

    # Try generating scripts with retries
    model = get_qwen2_vl_model('longwriter-v')

    max_retry = 3
    
    for retry in range(max_retry):
        try:
            res = model.inference_vllm(messages)

            print(res)    

            res_json = extract_json(res)

            break              

        except Exception as e:
            print(f"Error: {e}")
            print(f"Retrying... ({retry+1}/{max_retry})")
            if retry == max_retry - 1:
                raise Exception(f"Failed to generate scripts after {max_retry} retries")

    return res_json