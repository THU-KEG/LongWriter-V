import os
import json
from pathlib import Path
from inference.api.gpt import GPT_Interface
from utils import extract_json
from tqdm import tqdm

def lecgen_outline(imgs, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    number_of_slides = len(imgs)

    input_tokens = 0
    output_tokens = 0


    # Initialize progress tracking
    try:
        import streamlit as st
        with st.sidebar:
            st.markdown("### Generation Progress")
            progress_container = st.empty()
            status_text = st.empty()
        has_streamlit = True
    except:
        has_streamlit = False
    
    # Update progress function
    def update_progress(stage, progress, status):
        if has_streamlit:
            status_text.text(status)
            progress_value = progress / 100.0  # Convert percentage to 0-1 range
            with progress_container:
                st.progress(progress_value, f"{stage}")
    
    try:
        # Outline Generation Stage
        update_progress("Outline Generation", 0, "Generating outlines...")
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

Note: You must generate an outline for every single slide, so please don't miss any slides. The length of your output JSON must match the number of slides. Please verify that your output includes an outline for every slide. Output only the JSON result without any other content. The sum of all target_words should equal (number of slides * 200) = {number_of_slides * 200}.
"""
        max_try = 3
        for i in range(max_try):
            complete = True
            messages = [dict(role="user", content=[dict(type="text", text=prompt_outline)] + [dict(type="image_url", image_url=dict(url=img)) for img in imgs])]
            response = GPT_Interface.call(model="gpt-4o", messages=messages, use_cache=False)
            outlines = extract_json(response)
            for j in range(len(imgs)):
                if not str(j+1) in outlines:
                    complete = False
                    break
            if complete:
                break
        if not complete:
            raise ValueError("Failed to generate complete outlines")

        update_progress("Outline Generation", 100, "Outlines generated!")
        
        with open(f"{output_dir}/outline.json", "w") as f:
            json.dump(outlines, f, ensure_ascii=False, indent=4)

        # Initial Script Generation Stage
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
        for i in tqdm(range(len(imgs)), desc="Generating initial scripts"):
            progress = int((i + 1) / len(imgs) * 100)
            update_progress("Initial Script Generation", progress, f"Generating initial script for slide {i+1}/{len(imgs)}...")
            
            prompt_gen_formatted = prompt_gen.format(outline=outlines[str(i+1)]["outline"],
                                                     target_words=outlines[str(i+1)]["target_words"])
            messages = [dict(role="user", content=[dict(type="text", text=prompt_gen_formatted)] + [dict(type="image_url", image_url=dict(url=imgs[i]))])] 
            response = GPT_Interface.call(model="gpt-4o", messages=history + messages, use_cache=False)
            
            messages.append(dict(role="assistant", content=response))
            history.extend(messages)
            res.append(response)
            if len(history) > ctx_len*2:
                history = history[-ctx_len*2:]
        
        update_progress("Initial Script Generation", 100, "Initial scripts generated!")
        
        # Rewrite Stage
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
        for i in tqdm(range(len(res)), desc="Rewriting scripts"):
            progress = int((i + 1) / len(res) * 100)
            update_progress("Script Rewriting", progress, f"Rewriting script for slide {i+1}/{len(res)}...")
            
            prev_script = res[i-1] if i > 0 else "<<START OF LECTURE>>"
            current_script = res[i]
            next_script = res[i+1] if i < len(res)-1 else "<<END OF LECTURE>>"
            
            messages = [dict(role="user", content=prompt_rewrite.format(
                target_words=outlines[str(i+1)]['target_words'],
                prev_script=prev_script,
                current_script=current_script,
                next_script=next_script
            ))]
            
            response = GPT_Interface.call(model="gpt-4o", messages=messages)
            rewritten_res.append(response)

        update_progress("Script Rewriting", 100, "Script rewriting complete!")
        
        # Save rewritten scripts
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, script in enumerate(rewritten_res, 1):
            output_file = output_dir / f"{i}.txt"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(script)
        
        return rewritten_res
    finally:
        # Clean up progress indicators
        if has_streamlit:
            try:
                progress_container.empty()
                status_text.empty()
            except:
                pass