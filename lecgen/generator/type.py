from inference.api.gpt import GPT_Interface
from tqdm import tqdm

def classify_page_type(img):
    """Classify PPT page into types (1=title, 2=knowledge, 3=interactive) based on image content only"""
    messages = [
        dict(role="user", content=[
            dict(type="image_url", image_url=dict(url=img)),
            """Analyze this PPT slide and classify it into one of three types. Respond with ONLY the number (1, 2, or 3).

Type 1 - Title/Connecting Page:
- Large, prominent text (typically >36pt font)
- Section headings or chapter titles
- Minimal content (usually <30 words)
- Often includes course name, chapter numbers
- May have decorative elements or simple graphics
- No detailed bullet points or diagrams
- Examples: "Chapter 1: Introduction", "Course Overview", "Section Break"

Type 2 - Knowledge/Content Page:
- Contains detailed information
- Multiple bullet points or paragraphs
- Diagrams, charts, or illustrations
- Technical content or explanations
- Definitions or concept descriptions
- Examples, formulas, or code snippets
- Examples: lecture content, concept explanations, process diagrams

Type 3 - Interactive/Exercise Page:
- Contains explicit questions or problems
- Has practice exercises or tasks
- Discussion prompts or debate topics
- Student activity instructions
- Quiz questions or review problems
- Space for student input or responses
- Examples: "Discussion Question:", "Practice Problem:", "Group Activity:"

First, identify the key visual characteristics of the slide.
Then, match these characteristics against the criteria above.
Finally, output ONLY the single digit (1, 2, or 3) that best matches.

Output format: Just the number (1, 2, or 3) with no other text."""
        ])
    ]
    
    response = GPT_Interface.call(model="gpt-4o-2024-05-13", messages=messages)

    # Extract just the number from response
    for char in response:
        if char in ['1','2','3']:
            print(f"classify_page_type: {char}")
            return int(char)
    
    # Default to knowledge page if classification fails
    return 2

def generate_script_by_type(img, page_type, prev_script=""):
    """Generate appropriate script based on page type classification"""
    
    # Different prompts optimized for each page type
    prompts = {
        1: """This appears to be a title or transition slide. Generate a brief, engaging introduction or transition that:
        - Previews the upcoming content or summarizes previous content
        - Uses a conversational, welcoming tone
        - Keeps the explanation concise (2-3 sentences)
        Respond with just the script text.""",
        
        2: """This is a knowledge-focused slide. Generate a detailed explanation that:
        - Thoroughly explains all key concepts shown
        - Uses clear examples and analogies where helpful
        - Maintains an educational but engaging tone
        - Connects ideas to previous content where relevant
        Respond with just the script text.""",
        
        3: """This is an interactive slide. Generate a script that:
        - Poses thought-provoking questions to the audience
        - Encourages participation and discussion
        - Provides space for student responses
        - Guides students through exercises or problems
        Respond with just the script text."""
    }

    system_prompt = """You are a professional lecturer generating scripts for PowerPoint slides. Your most critical requirement is to match the language of the slides:

1. Language Matching (HIGHEST PRIORITY):
   - YOU MUST GENERATE THE SCRIPT IN THE SAME LANGUAGE AS THE SLIDE CONTENT
   - If the slide is in Chinese, generate Chinese script
   - If the slide is in English, generate English script
   - If the slide contains mixed languages, follow the dominant language
   - NEVER translate the content to a different language

2. Language Consistency:
   - Keep all technical terms, proper nouns, and key phrases exactly as shown in the slides
   - Use the same writing style and tone as the slide content
   - Maintain consistent terminology throughout the lecture
   - Match the language level and formality of the slide content

3. Style Requirements:
   - Write in a clear, conversational teaching style
   - Use natural transitions between concepts
   - Keep the tone engaging but professional
   - Adapt formality to match the slide's style

4. Content Guidelines:
   - Focus on explaining what's actually shown in the slide
   - Don't introduce major concepts that aren't present in the slide
   - Maintain logical flow with previous content
   - Be concise but thorough in explanations

5. Format Requirements:
   - Generate pure script text without any markdown or formatting
   - Don't include speaker notes or meta-instructions
   - Don't use bullet points or numbering in the script

IMPORTANT: The language of your response MUST MATCH the language used in the slide. This is the highest priority requirement."""

    messages = [
        dict(role="system", content=system_prompt),
        dict(role="user", content=[
            dict(type="text", text=f"Previous script context:\n{prev_script}\n\n{prompts[page_type]}"),
            dict(type="image_url", image_url=dict(url=img))
        ])
    ]
    
    script = GPT_Interface.call(model="gpt-4o", messages=messages)
    return script

def type_based_generation(images, progress_manager=None):
    """Main function to handle type-based script generation workflow.
    
    Args:
        images: List of base64 encoded images
        progress_manager: Optional ProgressManager instance for progress tracking
        
    Returns:
        List of generated scripts
    """
    scripts = []
    prev_script = ""
    total_images = len(images)
    
    for idx, img_base64 in enumerate(images):
        if progress_manager:
            progress = ((idx + 1) / total_images) * 100
            progress_manager.update_progress(
                "Type-based Generation",
                progress,
                f"Generating script for slide {idx + 1}/{total_images}"
            )
        
        # First classify the page type
        page_type = classify_page_type(img_base64)
        
        # Then generate script based on type and previous context
        script = generate_script_by_type(img_base64, page_type, prev_script)
        scripts.append(script)
        prev_script = script
    
    if progress_manager:
        progress_manager.update_progress(
            "Type-based Generation",
            100,
            "Script generation complete!"
        )
    
    return scripts

def classify_pages(imgs):
    """Classify all pages in a presentation, returning list of integers 1-3"""
    classifications = []
    for img in tqdm(imgs):
        page_type = classify_page_type(img)
        classifications.append(page_type)
    return classifications
