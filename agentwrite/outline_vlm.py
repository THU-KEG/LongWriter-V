import os
import json
from pathlib import Path
from inference.api import GPT_Interface
from utils import extract_json
from tqdm import tqdm
from utils import encode_image_to_base64
import re

def lecgen_outline(imgs, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    number_of_slides = len(imgs)

    input_tokens = 0
    output_tokens = 0

    prompt_outline = f"""You are an excellent teacher. Your task is to generate a lecture outline for all the provided PPT slides. You should consider the entire course content holistically and generate a brief outline for each slide, describing what content should be included in the lecture script, including key points to teach and logical connections between slides. You don't need to elaborate on each point - just briefly describe what content should be included.

Please analyze the language of the slides and respond in the same language as the slides.

For each slide, please also specify a target word count that adds up to (number of slides * 200) = {number_of_slides * 200} words total. For example, if there are 3 slides, allocate the 600 total words across the slides based on content density.

Please output in the following format:
    ```
    {{
        "1": {{
            "outline": "Outline for first slide",
            "target_words": 120
        }},
        "2": {{
            "outline": "Outline for second slide", 
            "target_words": 180
        }},
        ...
    }}
    ```

    """
    attention = f"""Note: You must generate an outline for every single slide, so please don't miss any slides. The length of your output JSON must match the number of slides. Please verify that your output includes an outline for every slide. Output only the JSON result without any other content. The sum of all target_words should equal (number of slides * 200) = {number_of_slides * 200}."""
    max_try = 3
    for i in range(max_try):
        prompt_outline = prompt_outline + attention
        complete = True
        messages = [dict(role="user", content=[dict(type="text", text=prompt_outline)] + [dict(type="image_url", image_url=dict(url=img)) for img in imgs])]
        response, it, ot = GPT_Interface.call_gpt4o(messages=messages)
        input_tokens += it
        output_tokens += ot
        outlines = extract_json(response)
        for j in range(len(imgs)):
            if not str(j+1) in outlines:
                complete = False
                break
        if complete:
            break
    if not complete:
        raise ValueError("Failed to generate complete outlines")

    with open(f"{output_dir}/outline.json", "w") as f:
        json.dump(outlines, f, ensure_ascii=False, indent=4)

    prompt_gen = """You are an excellent teacher. Your task is to generate a lecture script based on the provided outline and PPT slide. The script should:
1. Cover all content points described in the outline
2. Match the content shown in the PPT slide
3. Flow naturally from the previous slide's script as part of the same lecture
4. Be engaging and educational, inspiring students' interest and critical thinking
5. Be limited to {target_words} words maximum

Please analyze the language of the slides and respond in the same language as the slides.

Please output only the lecture script content without any other text. Keep your response within {target_words} words.

Outline for this slide:
{outline}"""
    history = []
    ctx_len = 5

    res = []
    for i in tqdm(range(len(imgs)), desc=f"Generating lecture scripts for {os.path.basename(output_dir)}"):
        prompt_gen_formatted = prompt_gen.format(outline=outlines[str(i+1)]["outline"], target_words=outlines[str(i+1)]["target_words"])
        messages = [dict(role="user", content=[dict(type="text", text=prompt_gen_formatted)] + [dict(type="image_url", image_url=dict(url=img)) for img in imgs])] 
        response, it, ot = GPT_Interface.call_gpt4o(messages=history + messages)
        input_tokens += it
        output_tokens += ot
        messages.append(dict(role="assistant", content=response))
        history.extend(messages)
        res.append(response)
        if len(history) > ctx_len*2:
            history = history[-ctx_len*2:]
    
    # Replace the single rewrite section with iterative rewriting
    prompt_rewrite = """You are an excellent editor. Your task is to rewrite this lecture script section to:
1. Flow naturally with the previous and next sections
2. Stay within the {target_words} word limit while preserving key content
3. Maintain an engaging and educational tone
4. Keep the same language as the original

Previous section (for context):
{prev_script}

Current script to rewrite:
{current_script}

Next section (for context):
{next_script}

Please output only the rewritten script without any other text or formatting."""

    rewritten_res = []
    for i in tqdm(range(len(res)), desc=f"Rewriting lecture scripts for {os.path.basename(output_dir)}"):
        prev_script = res[i-1] if i > 0 else "<<START OF LECTURE>>"
        current_script = res[i]
        next_script = res[i+1] if i < len(res)-1 else "<<END OF LECTURE>>"
        
        messages = [dict(role="user", content=prompt_rewrite.format(
            target_words=outlines[str(i+1)]['target_words'],
            prev_script=prev_script,
            current_script=current_script,
            next_script=next_script
        ))]
        
        response, it, ot = GPT_Interface.call_gpt4o(messages=messages)
        input_tokens += it
        output_tokens += ot
        rewritten_res.append(response)

    # Save rewritten scripts
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, script in enumerate(rewritten_res, 1):
        output_file = output_dir / f"{i}.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(script)
    
    return rewritten_res, input_tokens, output_tokens


def instruction_outline(instruction, img_paths):

    prompt_plan = """You are an expert planner. Your task is to break down a writing task into clear subtasks based on the provided images and writing instruction.

Please analyze the images and writing instruction carefully, then create a detailed outline in this format:

Section 1 - Main Point: [Key points to cover based on images and instruction] - Word Count: [200-1000 words]
Section 2 - Main Point: [Key points to cover based on images and instruction] - Word Count: [200-1000 words]
...

Make each section focused and specific while ensuring the full outline:
1. Covers all key content from both images and writing instruction
2. Flows logically from section to section
3. Has reasonable word count targets (200-1000 words per section)
4. Forms a cohesive whole that fulfills the writing instruction

Writing instruction:
$INST$

Output only the outline with no other text."""

    prompt_write = """You are an expert writer. Your task is to write the next section of a longer piece based on:

1. The provided images and writing instruction
2. The outline plan
3. Previously written sections

Writing instruction:
$INST$

Outline plan:
$PLAN$

Previous sections:
$TEXT$

Please write section $STEP$ following these guidelines:
1. Focus on the main points specified in the outline
2. Stay within the target word count
3. Flow naturally from previous sections
4. Integrate relevant details from the images
5. Maintain a consistent tone and style
6. Write only this section, not a full conclusion

Output only the new section with no other text."""

    imgs = [encode_image_to_base64(img) for img in img_paths]

    # First generate the outline plan
    messages = [dict(
        role="user", 
        content=[
            dict(type="text", text=prompt_plan.replace("$INST$", instruction)),
            *[dict(type="image_url", image_url=dict(url=img)) for img in imgs]
        ]
    )]

    plan, input_tokens, output_tokens = GPT_Interface.call_gpt4o(messages=messages)

    # Updated section extraction using regex to keep "Section X" prefix
    section_pattern = r'(Section \d+[^:]*:.*?)(?=Section \d+|$)'
    sections = re.findall(section_pattern, plan, re.DOTALL)
    sections = [section.strip() for section in sections]
    
    if not sections:
        # Fallback pattern if sections aren't numbered
        sections = [s.strip() for s in plan.split('\n') if s.strip()]

    results = {
        'metadata': {
            'instruction': instruction,
            'image_paths': img_paths if img_paths else []
        },
        'plan': plan,
        'sections': [],
        'total_word_count': 0  # Add total word count tracking
    }

            
    # Generate each section iteratively
    output_text = []
    for i, section in enumerate(sections):
        messages = [dict(
            role="user",
            content=[
                {"type": "text", "text": prompt_write.replace(
                    "$INST$", instruction
                ).replace(
                    "$PLAN$", plan
                ).replace(
                    "$TEXT$", "\n\n".join(output_text)
                ).replace(
                    "$STEP$", section
                )},
                *[{"type": "image_url", "image_url": {"url": img}} for img in imgs]
            ]
        )]
        
        response, it, ot = GPT_Interface.call_gpt4o(messages=messages)
        input_tokens += it
        output_tokens += ot
        
        word_count = len(response.split())  # Count words in response
        
        # Record section, response, and word count
        results['sections'].append({
            'section': section,
            'response': response,
            'word_count': word_count  # Add word count for section
        })
        
        results['total_word_count'] += word_count  # Update total word count

    return results
