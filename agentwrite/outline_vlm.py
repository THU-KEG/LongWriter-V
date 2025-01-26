from inference.api import GPT_Interface
from utils import encode_image_to_base64
import re


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
        
        results['sections'].append({
            'section': section,
            'response': response,
            'word_count': word_count  # Add word count for section
        })
        
        results['total_word_count'] += word_count  # Update total word count

    return results
