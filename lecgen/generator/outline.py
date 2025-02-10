import os
import json
from pathlib import Path
from inference.api.gpt import GPT_Interface, DeepSeek_Interface
from utils import extract_json
from tqdm import tqdm
from lecgen.generator.type import classify_page_type

def lecgen_outline(imgs, output_dir, progress_manager):

    os.makedirs(output_dir, exist_ok=True)

    number_of_slides = len(imgs)
    # Outline Generation Stage
    progress_manager.update_progress("Outline Generation", 0, "Generating outlines...")
    prompt_outline = f"""You are an excellent teacher. Your task is to analyze the PPT slides and generate a comprehensive lecture outline.

When analyzing each slide, consider its characteristics and purpose:
- Is it a title or transition slide with prominent headings and minimal content?
- Is it a knowledge-focused slide with detailed information, diagrams, or explanations?
- Is it an interactive slide with questions, exercises, or discussion prompts?

Based on the slide's nature, generate an appropriate outline that:
- Matches the slide's teaching purpose (introducing topics, explaining concepts, or engaging students)
- Includes key points to teach
- Establishes logical connections with other slides
- Suggests appropriate teaching approach (brief introduction, detailed explanation, or interactive discussion)
- Allocates suitable word count based on content density

Please analyze the language of the slides and respond in the same language as the slides.
The total word count should add up to (number of slides * 200) = {number_of_slides * 200} words.

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

Note: You must generate an outline for every single slide. The length of your output JSON must match the number of slides. The sum of all target_words should equal {number_of_slides * 200}. Output only the JSON result."""

    max_try = 3
    use_cache = True
    for i in range(max_try):
        complete = True
        messages = [dict(role="user", content=[dict(type="text", text=prompt_outline)] + [dict(type="image_url", image_url=dict(url=img)) for img in imgs])]
        response = GPT_Interface.call(model="gpt-4o-2024-05-13", messages=messages, use_cache=use_cache)
        use_cache = False
        outlines = extract_json(response)
        for j in range(len(imgs)):
            if not str(j+1) in outlines:
                complete = False
                break
        if complete:
            break
    if not complete:
        raise ValueError("Failed to generate complete outlines")

    progress_manager.update_progress("Outline Generation", 100, "Outlines generated!")
    
    with open(f"{output_dir}/outline.json", "w") as f:
        json.dump(outlines, f, ensure_ascii=False, indent=4)

    # Initial Script Generation Stage
    system_prompt = """You are an excellent teacher creating lecture scripts for online teaching. Follow these core principles:

1. Information Fragmentation (碎片化原则):
   - Break down information into digestible chunks
   - Maintain appropriate information density
   - Keep learner's attention and reduce cognitive load
   - Avoid overwhelming students with too much information at once

2. PPT-Script Correspondence (对应原则):
   - Strictly align script content with PPT content
   - Use the same examples as shown in slides
   - Follow the same information organization order
   - Ensure perfect synchronization between visual and audio

3. Cognitive Load Optimization (降低认知成本原则):
   - Use language that's easy to understand
   - Follow students' cognitive patterns
   - Use precise vocabulary and simple sentence structures
   - Make complex concepts accessible and clear

Your goal is to create natural, engaging scripts that make learning efficient and enjoyable."""

    prompt_gen = """Generate a lecture script for this PPT slide that follows the provided outline.

Key Requirements:
1. Follow Outline Strictly:
   - Cover all points mentioned in the outline
   - Follow the same logical structure as the outline
   - Maintain the same emphasis and priorities
   - Keep the same teaching sequence

2. Match PPT Content:
   - Cover exactly what's shown on the slide
   - Use the same examples and cases
   - Follow the same content order
   - Stay within {target_words} words

3. Language and Style:
   - Use simple, clear language
   - Break down complex concepts
   - Make content easily digestible
   - MOST IMPORTANT: Use the same language as the slide (Chinese for Chinese slides, English for English slides)

4. Teaching Approach:
   - Explain concepts step by step
   - Connect with previous knowledge
   - Use a natural teaching rhythm
   - Make abstract concepts concrete

Please output only the script content without any other text.

Outline for this slide:
{outline}"""

    history = []
    ctx_len = 3

    res = []
    for i in tqdm(range(len(imgs)), desc="Generating initial scripts"):
        progress = int((i + 1) / len(imgs) * 100)
        progress_manager.update_progress("Initial Script Generation", progress, f"Generating initial script for slide {i+1}/{len(imgs)}...")
        
        slide_num = str(i + 1)
        
        prompt_gen_formatted = prompt_gen.format(
            outline=outlines[slide_num]["outline"],
            target_words=outlines[slide_num]["target_words"]
        )
        messages = [
            dict(role="system", content=system_prompt),
            dict(role="user", content=[dict(type="text", text=prompt_gen_formatted)] + [dict(type="image_url", image_url=dict(url=imgs[i]))])
        ] 
        response = GPT_Interface.call(model="gpt-4o-2024-05-13", messages=history + messages)
        
        messages.append(dict(role="assistant", content=response))
        history.extend(messages)
        res.append(response)
        if len(history) > ctx_len*2:
            history = history[-ctx_len*2:]
    
    progress_manager.update_progress("Initial Script Generation", 100, "Initial scripts generated!")
    
    # Continuity Fix Phase
    system_prompt_continuity = """You are a professional lecture script editor. Your task is to improve the flow and continuity of an entire lecture's scripts.

Two critical requirements:
1. Language Matching: ALWAYS maintain the exact same language as the original scripts (Chinese for Chinese, English for English)
2. Content Boundaries: Each slide's content must stay within its own boundaries - do not copy or move content between slides"""

    prompt_fix_continuity = """Here are all the lecture scripts. Please improve their flow and continuity while maintaining each script's core content.

Original Scripts:
{scripts_json}

Requirements:
1. Keep the exact same language as the original scripts
2. Keep each slide's core content unchanged
3. Improve transitions and flow between slides
4. Keep each script within its word limit

Output the improved scripts in valid JSON format with slide numbers as keys and improved scripts as values. For example:
{{"1": "improved script 1", "2": "improved script 2", ...}}

Output only the JSON."""

    # Single GPT-4 call for all scripts
    progress_manager.update_progress("Continuity Fix", 50, "Improving lecture flow...")
    
    # Prepare scripts in JSON format
    scripts_json = {str(i+1): script for i, script in enumerate(res)}
    
    messages = [
        dict(role="system", content=system_prompt_continuity),
        dict(role="user", content=prompt_fix_continuity.format(
            scripts_json=json.dumps(scripts_json, ensure_ascii=False, indent=2)
        ))
    ]
    
    # Retry mechanism
    max_try = 3
    use_cache = True
    fixed_scripts = res  # Default to original scripts
    
    for attempt in range(max_try):
        try:
            response = GPT_Interface.call(model="gpt-4o-2024-05-13", messages=messages, max_tokens=8192, use_cache=use_cache)
            print(response)
            use_cache = False
            improved_scripts = extract_json(response)
            
            # Validate all slides are present
            complete = True
            temp_scripts = []
            for i in range(len(res)):
                slide_num = str(i + 1)
                if slide_num not in improved_scripts:
                    print(f"Missing script for slide {slide_num} in attempt {attempt + 1}")
                    complete = False
                    break
                temp_scripts.append(improved_scripts[slide_num])
            
            if complete:
                fixed_scripts = temp_scripts
                print(f"Successfully improved scripts on attempt {attempt + 1}")
                break
                
        except Exception as e:
            print(f"Error in attempt {attempt + 1}: {e}")
            if attempt == max_try - 1:
                print("All attempts failed, falling back to original scripts")
    
    progress_manager.update_progress("Continuity Fix", 100, "Continuity fixes complete!")
    
    # Save final scripts
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, script in enumerate(fixed_scripts, 1):
        output_file = output_dir / f"{i}.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(script)
    
    progress_manager.cleanup()
    return fixed_scripts