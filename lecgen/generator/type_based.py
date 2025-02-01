from inference.api.gpt import GPT_Interface
from tqdm import tqdm

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
    
    response = GPT_Interface.call(model="gpt-4o", messages=messages)

    print(response)
    
    # Extract just the number from response
    for char in response:
        if char in ['1','2','3']:
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

    messages = [
        dict(role="user", content=[
            dict(type="text", text=f"Previous script context:\n{prev_script}\n\n{prompts[page_type]}"),
            dict(type="image_url", image_url=dict(url=img))
        ])
    ]
    
    script = GPT_Interface.call(model="gpt-4o", messages=messages)
    return script


def classify_pages(imgs):
    """Classify all pages in a presentation, returning list of integers 1-3"""
    classifications = []
    for img in tqdm(imgs):
        page_type = classify_page_type(img)
        classifications.append(page_type)
    return classifications

