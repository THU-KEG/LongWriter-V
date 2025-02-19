from inference.api.gpt import GPT_Interface
from utils import encode_image_to_base64
import re


def instruction_outline(instruction, img_paths):

    prompt_plan = open('prompts/plan.txt', 'r').read()

    prompt_write = open('prompts/write.txt', 'r').read()

    imgs = [encode_image_to_base64(img) for img in img_paths]

    # First generate the outline plan
    messages = [dict(
        role="user", 
        content=[
            dict(type="text", text=prompt_plan.replace("$INST$", instruction)),
            *[dict(type="image_url", image_url=dict(url=img)) for img in imgs]
        ]
    )]

    plan = GPT_Interface.call(model="gpt-4o", messages=messages)

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
        
        response = GPT_Interface.call(model="gpt-4o", messages=messages)
        
        word_count = len(response.split())  # Count words in response
        
        results['sections'].append({
            'section': section,
            'response': response,
            'word_count': word_count  # Add word count for section
        })
        
        results['total_word_count'] += word_count  # Update total word count

    return results
