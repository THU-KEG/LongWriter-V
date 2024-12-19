import streamlit as st
import os
from pathlib import Path
from PIL import Image
import requests
import base64
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
import httpx

@dataclass
class ScriptContent:
    """Data class to hold script content and metadata"""
    content: str
    file_name: Optional[str] = None
    is_empty: bool = False
    last_modified: Optional[float] = None

class ScriptViewer:
    def __init__(self):
        self.title = "PPT Slides and Scripts Viewer"
        self.init_session_state()
        
    def init_session_state(self):
        """Initialize session state variables"""
        if 'scripts' not in st.session_state:
            st.session_state.scripts = {}
        if 'current_images' not in st.session_state:
            st.session_state.current_images = []
        if 'modified_scripts' not in st.session_state:
            st.session_state.modified_scripts = set()

    def load_images(self, image_directory: Path) -> List[Path]:
        """Load and sort image files from directory"""
        try:
            image_files = sorted(
                [f for f in image_directory.glob("*.png")],
                key=lambda x: int(x.stem)
            )
            return image_files
        except Exception as e:
            st.error(f"Error loading images: {str(e)}")
            return []

    def load_scripts(self, script_directory: Path) -> List[ScriptContent]:
        """Load and sort script files from directory"""
        if not script_directory.is_dir():
            return []
        
        scripts = []
        try:
            script_files = sorted(
                [f for f in script_directory.glob("*.txt")],
                key=lambda x: int(x.stem)
            )
            
            for script_path in script_files:
                try:
                    content = script_path.read_text(encoding='utf-8').strip()
                    scripts.append(ScriptContent(
                        content=content,
                        file_name=script_path.name,
                        is_empty=(not content),
                        last_modified=script_path.stat().st_mtime
                    ))
                except Exception as e:
                    st.error(f"Error loading {script_path.name}: {str(e)}")
                    
            return scripts
        except Exception as e:
            st.error(f"Error processing script directory: {str(e)}")
            return []

    async def polish_script(self, images: List[Path], previous_scripts: List[str]) -> Optional[str]:
        """Call API to polish script with improved error handling"""
        try:
            encoded_imgs = [self._encode_image(img) for img in images]
            
            async with httpx.AsyncClient() as client:
                try:
                    response = await client.post(
                        "http://0.0.0.0:8000/polish",
                        json={
                            "imgs": encoded_imgs,
                            "scripts": previous_scripts
                        },
                        timeout=120.0
                    )
                    
                    if response.status_code == 422:
                        error_detail = response.json().get('detail', 'Unknown validation error')
                        st.error(f"API Validation Error: {error_detail}")
                        return None
                    
                    if response.status_code == 200:
                        result = response.json()
                        if result.get("success"):
                            return result["script"]
                        st.error(f"API returned error: {result.get('error', 'Unknown error')}")
                    else:
                        st.error(f"API call failed with status code {response.status_code}: {response.text}")
                except httpx.TimeoutException:
                    st.error("API request timed out. Please try again.")
                except httpx.RequestError as e:
                    st.error(f"API request failed: {str(e)}")
        except Exception as e:
            st.error(f"Error during polish: {str(e)}")
        return None

    @staticmethod
    def _encode_image(image_path: Path) -> str:
        """Encode image to base64"""
        return base64.b64encode(image_path.read_bytes()).decode('utf-8')

    def save_scripts(self, script_directory: Path):
        """Save modified scripts back to files"""
        if not st.session_state.modified_scripts:
            return
            
        for script_key in st.session_state.modified_scripts:
            content = st.session_state.scripts[script_key]
            file_path = script_directory / f"{script_key}.txt"
            try:
                file_path.write_text(content, encoding='utf-8')
                st.success(f"Saved {file_path.name}")
            except Exception as e:
                st.error(f"Error saving {file_path.name}: {str(e)}")
        
        st.session_state.modified_scripts.clear()

    def render_script_editor(self, script: ScriptContent, key: str,
                           can_polish: bool = True) -> None:
        """Render an individual script editor with polish button"""
        # if script.file_name:
        #     st.markdown(f"**{script.file_name}**")
        
        # Use session state for script content
        if key not in st.session_state.scripts:
            st.session_state.scripts[key] = script.content
        
        edited_content = st.text_area(
            label=f"Script Content {key}",
            value=st.session_state.scripts[key],
            key=f"textarea_{key}",
            height=400,
            label_visibility="collapsed",
            on_change=self.handle_script_change,
            args=(key,)
        )
        
        if can_polish:
            if st.button("Polish", key=f"polish_{key}"):
                self.handle_polish_request(key)

    def handle_script_change(self, key: str):
        """Handle script content changes"""
        new_content = st.session_state[f"textarea_{key}"]
        if new_content != st.session_state.scripts[key]:
            st.session_state.scripts[key] = new_content
            st.session_state.modified_scripts.add(key)

    def handle_polish_request(self, key: str):
        """Handle polish button click for a specific script"""
        import asyncio
        
        # Get current image and script indices from the key
        i, j = map(int, key.split('_')[1:])
        
        # Get relevant images (current and previous)
        image_path = Path(st.session_state.image_dir)
        images = self.load_images(image_path)
        context_images = images[max(0, i-10):i+1]  # Current and up to 10 previous images
        
        # Get previous scripts
        previous_scripts = []
        for idx in range(max(0, i-10), i):
            script_key = f"script_{idx}_{j}"
            if script_key in st.session_state.scripts:
                previous_scripts.append(st.session_state.scripts[script_key])
        
        # Call polish API
        with st.spinner("Polishing script..."):
            result = asyncio.run(self.polish_script(
                images=context_images,
                previous_scripts=previous_scripts
            ))
            
            if result:
                st.session_state.scripts[key] = result
                st.session_state.modified_scripts.add(key)
                st.rerun()

    def render(self):
        """Main render method with improved layout and functionality"""
        st.title(self.title)
        
        with st.sidebar:
            self.render_sidebar_controls()
        
        if not self.validate_inputs():
            return
        
        self.render_content()
        
        # Add save button at the bottom
        if st.session_state.modified_scripts:
            if st.button("Save All Changes"):
                self.save_scripts(Path(st.session_state.script_dirs[0]))

    def render_sidebar_controls(self):
        """Render sidebar controls"""
        st.sidebar.header("Configuration")
        
        image_dir = st.sidebar.text_input(
            "Image Directory:",
            help="Directory containing numbered PNG files"
        )
        
        num_script_dirs = st.sidebar.number_input(
            "Number of Script Columns:",
            min_value=1,
            max_value=5,
            value=1
        )
        
        # Check if script directories have changed
        old_script_dirs = getattr(st.session_state, 'script_dirs', [])
        script_dirs = [
            st.sidebar.text_input(f"Script Directory {i+1}:")
            for i in range(num_script_dirs)
        ]
        
        # If script directories changed, clear the scripts state
        if old_script_dirs != script_dirs:
            st.session_state.scripts = {}
            st.session_state.modified_scripts = set()
        
        # Store in session state
        st.session_state.image_dir = image_dir
        st.session_state.script_dirs = script_dirs

    def validate_inputs(self) -> bool:
        """Validate input directories"""
        if not st.session_state.image_dir:
            st.warning("Please enter an image directory path.")
            return False
            
        image_path = Path(st.session_state.image_dir)
        if not image_path.is_dir():
            st.error("Invalid image directory path.")
            return False
            
        return True

    def render_content(self):
        """Render main content area"""
        image_path = Path(st.session_state.image_dir)
        script_paths = [Path(d) for d in st.session_state.script_dirs if d]
        
        images = self.load_images(image_path)
        script_collections = [
            self.load_scripts(path) for path in script_paths
        ]
        
        if not images:
            st.warning("No PNG files found in the specified directory.")
            return
            
        for i, image_file in enumerate(images):
            # Add index display (i + 1 for 1-based indexing)
            st.markdown(f"**#{i + 1}**")
            
            cols = st.columns([1] + [2] * len(script_paths))
            
            with cols[0]:
                st.image(str(image_file), use_container_width=True)
            
            for j, scripts in enumerate(script_collections):
                with cols[j + 1]:
                    script = (scripts[i] if i < len(scripts)
                            else ScriptContent("", is_empty=True))
                    self.render_script_editor(script, f"script_{i}_{j}")

def main():
    viewer = ScriptViewer()
    viewer.render()

if __name__ == "__main__":
    main()
