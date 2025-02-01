from inference.api.gpt import GPT_Interface

def polish(imgs, scripts):
    # Initialize progress tracking
    try:
        import streamlit as st
        progress_bar = st.progress(0)
        status_text = st.empty()
        has_streamlit = True
    except:
        has_streamlit = False
    
    try:
        if has_streamlit:
            status_text.text("Processing slide...")
            progress_bar.progress(33)
        
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
                        dict(type="image_url", image_url=dict(url=img))
                    ]
                ),
                dict(
                    role="assistant",
                    content=script
                )
            ])
        
        if has_streamlit:
            status_text.text("Generating polished script...")
            progress_bar.progress(66)
        
        # Add the target image that needs a new script
        messages.append(
            dict(
                role="user",
                content=[
                    dict(type="text", text=prompt),
                    dict(type="image_url", image_url=dict(url=imgs[-1]))
                ]
            )
        )

        response = GPT_Interface.call_gpt4o(messages=messages, use_cache=False)
        
        if has_streamlit:
            progress_bar.progress(100)
            status_text.text("Script polishing complete!")
        
        return response[0]
        
    finally:
        # Clean up progress indicators
        if has_streamlit:
            try:
                progress_bar.empty()
                status_text.empty()
            except:
                pass

