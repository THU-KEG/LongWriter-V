import streamlit as st

# Set page config to wide mode
st.set_page_config(
    page_title="讲稿生成",
    layout="wide"
)

import sys
from pathlib import Path

# Add the project root to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from lecgen.optimizer.polish import polish
from PIL import Image
import base64
from dataclasses import dataclass
from typing import List, Optional
from io import BytesIO
import tempfile
from lecgen.generator.outline import lecgen_outline 
from utils import convert_pdf_to_png, pptx_to_pdf, encode_image_to_base64

@dataclass
class ScriptContent:
    """Data class to hold script content and metadata"""
    content: str
    file_name: Optional[str] = None
    is_empty: bool = False
    last_modified: Optional[float] = None

class ScriptViewer:
    def __init__(self):
        self.title = "Lecture Script Generator"
        self.init_session_state()
        
    def init_session_state(self):
        """Initialize session state variables"""
        if 'scripts' not in st.session_state:
            st.session_state.scripts = {}
        if 'current_images' not in st.session_state:
            st.session_state.current_images = []
        if 'modified_scripts' not in st.session_state:
            st.session_state.modified_scripts = set()
        if 'temp_dir' not in st.session_state:
            st.session_state.temp_dir = None
        if 'generation_method' not in st.session_state:
            st.session_state.generation_method = 'outline'
        if 'image_dir' not in st.session_state:
            st.session_state.image_dir = None
        if 'temp_base_dir' not in st.session_state:
            st.session_state.temp_base_dir = Path(tempfile.mkdtemp())
        if 'processed_files' not in st.session_state:
            st.session_state.processed_files = set()
        if 'should_generate' not in st.session_state:
            st.session_state.should_generate = False
        # New state variables for polish preview
        if 'polish_preview' not in st.session_state:
            st.session_state.polish_preview = {}  # key -> polished content
        if 'pending_polish' not in st.session_state:
            st.session_state.pending_polish = set()  # keys with pending polish previews

    def process_uploaded_file(self, uploaded_file) -> List[str]:
        """Process uploaded PPTX/PDF and return list of base64 encoded images"""
        try:
            # Create unique temp directory for this file
            basename = Path(uploaded_file.name).stem
            temp_dir = st.session_state.temp_base_dir / basename
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            # Save uploaded file
            input_path = temp_dir / uploaded_file.name
            with open(input_path, 'wb') as f:
                f.write(uploaded_file.getvalue())

            base64_images = []
            output_dir = temp_dir / 'images'
            output_dir.mkdir(parents=True, exist_ok=True)  # Ensure output directory exists
            
            if uploaded_file.name.endswith('.pdf'):
                # Use pdf2png utility function
                convert_pdf_to_png(str(input_path), str(output_dir))
            
            elif uploaded_file.name.endswith('.pptx'):
                # First convert PPTX to PDF
                pdf_success = pptx_to_pdf(str(input_path), str(temp_dir))
                if not pdf_success:
                    raise Exception("Failed to convert PPTX to PDF")
                
                # Then convert PDF to PNG
                pdf_path = temp_dir / f"{basename}.pdf"
                convert_pdf_to_png(str(pdf_path), str(output_dir))
            
            # Store temp_dir in session state for later use
            st.session_state.temp_dir = str(output_dir)
            
            # Scale and save images
            image_files = self.load_images(output_dir)
            base64_images = []
            for img_path in image_files:
                # Open and scale image
                with Image.open(img_path) as img:
                    # Scale down if width is greater than 1024 pixels
                    if img.width > 1024:
                        ratio = 1024 / img.width
                        new_size = (1024, int(img.height * ratio))
                        img = img.resize(new_size, Image.Resampling.LANCZOS)
                        # Save scaled image back to the same path
                        img.save(img_path, format="PNG")
                
                # Encode the saved image
                base64_images.append(encode_image_to_base64(str(img_path)))
            
            return base64_images
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            return []

    def generate_scripts(self, images: List[str], method: str) -> List[str]:
        try:
            if not images:
                raise ValueError("No images provided")
            
            total_images = len(images)
            
            if method == 'outline':
                scripts = lecgen_outline(images, st.session_state.temp_dir)
               
                # Update session state with generated scripts
                for i, script in enumerate(scripts):
                    script_key = f"script_{i}_0"
                    st.session_state.scripts[script_key] = script
                    st.session_state.modified_scripts.add(script_key)
                
                return scripts
                
            elif method == 'type_based':
                scripts = []
                prev_script = ""
                
                for idx, img_base64 in enumerate(images):
                    status_text.text(f"Generating script for slide {idx + 1}/{total_images}")
                    progress = (idx + 1) / total_images  # Convert to 0-1 range
                    progress_bar.progress(progress)
                    
                    page_type = 2  # Default to knowledge-focused
                    script = generate_script_by_type(img_base64, page_type, prev_script)
                    scripts.append(script)
                    prev_script = script
                    
                    # Update session state as each script is generated
                    script_key = f"script_{idx}_0"
                    st.session_state.scripts[script_key] = script
                    st.session_state.modified_scripts.add(script_key)
                
                status_text.text("Script generation complete!")
                progress_bar.empty()
                status_text.empty()
                return scripts
            
            return []  # Return empty list for unknown methods
            
        except Exception as e:
            st.error(f"Error generating scripts: {str(e)}")
            return []
        finally:
            # Clean up progress indicators if they still exist
            try:
                progress_bar.empty()
                status_text.empty()
            except:
                pass

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

    def polish_script(self, images: List[Path], previous_scripts: List[str]) -> Optional[str]:
        """Polish script using local polish function"""
        try:
            return polish(images, previous_scripts)
        except Exception as e:
            st.error(f"Error during polish: {str(e)}")
            return None


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
        # Use session state for script content
        if key not in st.session_state.scripts:
            st.session_state.scripts[key] = script.content
        
        # If there's a pending polish preview for this key, show it with accept/reject buttons
        if key in st.session_state.pending_polish:
            cols = st.columns([1, 1])
            with cols[0]:
                st.markdown("**Current Version:**")
                st.text_area(
                    label=f"Current Script {key}",
                    value=st.session_state.scripts[key],
                    key=f"current_{key}",
                    height=200,
                    disabled=True,
                    label_visibility="collapsed"
                )
            with cols[1]:
                st.markdown("**Polished Version (Preview):**")
                st.text_area(
                    label=f"Polished Script {key}",
                    value=st.session_state.polish_preview[key],
                    key=f"preview_{key}",
                    height=200,
                    disabled=True,
                    label_visibility="collapsed"
                )
            
            # Add accept/reject buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Accept Polish", key=f"accept_{key}"):
                    st.session_state.scripts[key] = st.session_state.polish_preview[key]
                    st.session_state.modified_scripts.add(key)
                    # Clean up preview state
                    del st.session_state.polish_preview[key]
                    st.session_state.pending_polish.remove(key)
                    st.rerun()
            with col2:
                if st.button("Reject Polish", key=f"reject_{key}"):
                    # Clean up preview state
                    del st.session_state.polish_preview[key]
                    st.session_state.pending_polish.remove(key)
                    st.rerun()
        else:
            # Show normal editor when no preview is pending
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
        # Get current image and script indices from the key
        i, j = map(int, key.split('_')[1:])
        
        # Get relevant images (current and up to 10 previous images)
        image_path = Path(st.session_state.temp_dir)
        images = self.load_images(image_path)
        context_images = images[max(0, i-10):i+1]  # Current and up to 10 previous images
        
        # Get previous scripts
        previous_scripts = []
        for idx in range(max(0, i-10), i):
            script_key = f"script_{idx}_{j}"
            if script_key in st.session_state.scripts:
                previous_scripts.append(st.session_state.scripts[script_key])
        
        # Call polish function directly
        with st.spinner("Polishing script..."):
            result = self.polish_script(
                images=[encode_image_to_base64(str(f)) for f in context_images],
                previous_scripts=previous_scripts
            )
            
            if result:
                # Store the polished result in preview state
                st.session_state.polish_preview[key] = result
                st.session_state.pending_polish.add(key)
                st.rerun()

    def render(self):
        """Main render method with improved layout and functionality"""
        st.title(self.title)
        
        with st.sidebar:
            self.render_sidebar_controls()
        
        if not self.validate_inputs():
            return
        
        # Handle script generation if requested
        if st.session_state.should_generate:
            if st.session_state.current_images:
                self.generate_scripts(
                    st.session_state.current_images,
                    st.session_state.generation_method
                )
            st.session_state.should_generate = False
        
        self.render_content()
        
        # Add save button at the bottom
        if st.session_state.modified_scripts:
            if st.button("Save All Changes"):
                self.save_scripts(Path(st.session_state.script_dirs[0]))

    def render_sidebar_controls(self):
        """Render sidebar controls"""
        st.sidebar.header("Configuration")
        
        uploaded_file = st.sidebar.file_uploader(
            "Upload PPTX/PDF file",
            type=['pptx', 'pdf']
        )
        
        if uploaded_file:
            # Only process the file if it hasn't been processed before
            file_identifier = f"{uploaded_file.name}_{uploaded_file.size}"
            if file_identifier not in st.session_state.processed_files:
                st.session_state.current_images = self.process_uploaded_file(uploaded_file)
                st.session_state.processed_files.add(file_identifier)
            
            st.sidebar.selectbox(
                "Script Generation Method",
                options=['outline', 'type_based'],
                key='generation_method'
            )
            
            if st.sidebar.button("Generate Scripts", key="generate_button"):
                st.session_state.should_generate = True
                st.rerun()

    def validate_inputs(self) -> bool:
        """Validate input directories"""
        return True

    def render_content(self):
        """Render main content area"""
        if not st.session_state.current_images:
            st.info("Please upload a PPTX or PDF file to begin.")
            return
            
        for i, img_base64 in enumerate(st.session_state.current_images):
            st.markdown(f"**Slide {i + 1}**")
            
            cols = st.columns([1, 2])
            
            with cols[0]:
                try:
                    # Extract the actual base64 data from the data URL
                    if ';base64,' in img_base64:
                        img_base64 = img_base64.split(';base64,')[1]
                    
                    image_bytes = base64.b64decode(img_base64)
                    image = Image.open(BytesIO(image_bytes))
                    st.image(image, use_container_width=True)
                except Exception as e:
                    st.error(f"Error displaying image {i+1}: {str(e)}")
                    continue  # Skip to next image if there's an error
            
            with cols[1]:
                script_key = f"script_{i}_0"
                script_content = st.session_state.scripts.get(script_key, "")
                script = ScriptContent(
                    content=script_content,
                    is_empty=(not script_content)
                )
                self.render_script_editor(script, script_key)

def main():
    viewer = ScriptViewer()
    viewer.render()

if __name__ == "__main__":
    main()
