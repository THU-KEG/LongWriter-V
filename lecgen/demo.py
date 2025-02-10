import streamlit as st
import pandas as pd
import io
import base64
from pathlib import Path
from lecgen.optimizer.polish import polish
from PIL import Image
from dataclasses import dataclass
from typing import List, Optional
from io import BytesIO
from utils import convert_pdf_to_png, pptx_to_pdf, encode_image_to_base64
from config import BASE_DIR
import openpyxl
import time
import uuid

# Set page config to wide mode
st.set_page_config(
    page_title="Lecture Script Generator",
    layout="wide"
)

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
            st.session_state.temp_base_dir = BASE_DIR / 'buffer'
            st.session_state.temp_base_dir.mkdir(parents=True, exist_ok=True)
        if 'processed_files' not in st.session_state:
            st.session_state.processed_files = set()
        if 'should_generate' not in st.session_state:
            st.session_state.should_generate = False
        # New state variables for polish preview
        if 'polish_preview' not in st.session_state:
            st.session_state.polish_preview = {}  # key -> polished content
        if 'pending_polish' not in st.session_state:
            st.session_state.pending_polish = set()  # keys with pending polish previews
        # New state variable for user edited scripts with original content
        if 'user_edited_scripts' not in st.session_state:
            st.session_state.user_edited_scripts = {}  # key -> {content: str, timestamp: float, original_content: str}
        # New state variable to store original content
        if 'original_scripts' not in st.session_state:
            st.session_state.original_scripts = {}  # key -> original content

    def process_uploaded_file(self, uploaded_file) -> List[str]:
        """Process uploaded PPTX and return list of base64 encoded images"""
        try:
            # Create unique temp directory for this file
            basename = Path(uploaded_file.name).stem
            temp_dir = st.session_state.temp_base_dir / basename
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate a unique PPT ID
            if 'ppt_id' not in st.session_state:
                st.session_state.ppt_id = str(uuid.uuid4())
            
            # Save uploaded file
            input_path = temp_dir / uploaded_file.name
            with open(input_path, 'wb') as f:
                f.write(uploaded_file.getvalue())

            base64_images = []
            output_dir = temp_dir / 'images'
            output_dir.mkdir(parents=True, exist_ok=True)  # Ensure output directory exists
           
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
                from lecgen.generator.outline import lecgen_outline
                progress_manager = ProgressManager()
                scripts = lecgen_outline(images, st.session_state.temp_dir, progress_manager)
                return scripts
            elif method == 'text':
                from lecgen.generator.text import gen_script
                progress_manager = ProgressManager()

                ppt_name = Path(st.session_state.temp_dir).parent.name
                ppt_file_name = ppt_name + '.pptx'
                ppt_path = Path(st.session_state.temp_dir).parent / ppt_file_name
                
                scripts = gen_script(str(ppt_path), images, progress_manager)
                return scripts
            elif method == 'longwriter-v':
                from lecgen.generator.longwriter_v import longwriter_v
                progress_manager = ProgressManager()
                scripts = longwriter_v(images)
                return scripts
            elif method == 'type':
                from lecgen.generator.type import type_based_generation
                progress_manager = ProgressManager()
                scripts = type_based_generation(images, progress_manager)
                
                return scripts
            
            return []  # Return empty list for unknown methods
            
        except Exception as e:
            st.error(f"Error generating scripts: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
        finally:
            progress_manager.cleanup()

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

    def polish_script(self, key: str) -> None:
        """Polish script using agentic workflow"""
        try:
            from lecgen.optimizer.agent import ScriptPolishAgent
            
            # Get current image and script indices from the key
            i, j = map(int, key.split('_')[1:])
            
            # Get all images
            image_path = Path(st.session_state.temp_dir)
            all_images = self.load_images(image_path)
            
            # Collect images for edited scripts and current script
            edited = []
            # First collect edited scripts and their corresponding images
            for edit_key, edit_info in st.session_state.user_edited_scripts.items():
                edit_i, edit_j = map(int, edit_key.split('_')[1:])
                edited.append({
                    "original_content": edit_info['original_content'],
                    "edited_content": edit_info['content'],
                    "image": encode_image_to_base64(str(all_images[edit_i]))
                })
            
            # Get previous and next scripts for context
            prev_key = f"script_{i-1}_0" if i > 0 else None
            next_key = f"script_{i+1}_0" if i < len(all_images) - 1 else None
            
            context = {
                "prev_script": st.session_state.scripts.get(prev_key, ""),
                "next_script": st.session_state.scripts.get(next_key, "")
            }
            
            to_polish = {
                "original_content": st.session_state.scripts[key],
                "image": encode_image_to_base64(str(all_images[i])),
                "context": context
            }

            agent = ScriptPolishAgent()
            
            # Run agentic polish workflow with images and PPT ID
            with st.spinner("Polishing script..."):
                result = agent.agentic_polish(
                    edited=edited,
                    to_polish=to_polish
                )
                
                if result:
                    # Store the polished result in preview state
                    st.session_state.polish_preview[key] = result.polished_script
                    st.session_state.pending_polish.add(key)
                    st.rerun()
            
        except Exception as e:
            st.error(f"Error during polish: {str(e)}")
            import traceback
            st.error(traceback.format_exc())

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
            # Store original content when first loading the script
            if key not in st.session_state.original_scripts:
                st.session_state.original_scripts[key] = script.content
        
        # Show edit status and original content option if this script has been edited by user
        if key in st.session_state.user_edited_scripts:
            # Create a container for the edit status and view original button
            status_container = st.container()
            with status_container:
                col1, col2 = st.columns([6, 1])
                with col1:
                    st.markdown("✏️ *Edited by user*", help="Click 'View Original' to see the original content")
                with col2:
                    view_original = st.button("View Original", key=f"view_original_{key}", type="secondary")
            
            # Show original content in a side-by-side view when button is clicked
            if view_original:
                st.markdown("### Content Comparison")
                cols = st.columns(2)
                with cols[0]:
                    st.markdown("**Original Version:**")
                    st.text_area(
                        "Original Script",
                        value=st.session_state.original_scripts[key],
                        height=200,
                        disabled=True,
                        key=f"original_{key}",
                        label_visibility="collapsed"
                    )
                with cols[1]:
                    st.markdown("**Current Version:**")
                    st.text_area(
                        "Current Script",
                        value=st.session_state.scripts[key],
                        height=200,
                        disabled=True,
                        key=f"current_view_{key}",
                        label_visibility="collapsed"
                    )
                st.markdown("---")  # Add a separator
        
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
                    self.polish_script(key)

    def handle_script_change(self, key: str):
        """Handle script content changes"""
        new_content = st.session_state[f"textarea_{key}"]
        if new_content != st.session_state.scripts[key]:
            # Store original content if this is the first edit
            if key not in st.session_state.original_scripts:
                st.session_state.original_scripts[key] = st.session_state.scripts[key]
            
            # Update current content
            st.session_state.scripts[key] = new_content
            st.session_state.modified_scripts.add(key)
            
            # Store user edit with timestamp and original content
            st.session_state.user_edited_scripts[key] = {
                'content': new_content,
                'timestamp': time.time(),
                'original_content': st.session_state.original_scripts[key]
            }

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
                scripts = self.generate_scripts(
                    st.session_state.current_images,
                    st.session_state.generation_method
                )
                # Update session state with generated scripts
                for i, script in enumerate(scripts):
                    script_key = f"script_{i}_0"
                    st.session_state.scripts[script_key] = script
                    st.session_state.modified_scripts.add(script_key)
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
            "Upload PPTX file",
            help="Please upload a file with less than 50 slides for better performance",
            type=['pptx']
        )
        
        if uploaded_file:
            # Only process the file if it hasn't been processed before
            file_identifier = f"{uploaded_file.name}_{uploaded_file.size}"
            if file_identifier not in st.session_state.processed_files:
                st.session_state.current_images = self.process_uploaded_file(uploaded_file)
                st.session_state.processed_files.add(file_identifier)
            
            st.sidebar.selectbox(
                "Script Generation Method",
                options=['outline', 'type', 'text', 'longwriter-v'],
                key='generation_method'
            )
            
            if st.sidebar.button("Generate Scripts", key="generate_button"):
                st.session_state.should_generate = True
                st.rerun()
            
            output = io.BytesIO()
            try:
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    # Create DataFrame for Excel
                    data = []
                    for i, img_base64 in enumerate(st.session_state.current_images):
                        script_key = f"script_{i}_0"
                        script_content = st.session_state.scripts.get(script_key, "")
                        data.append({
                            "Script": script_content
                        })
                    
                    df = pd.DataFrame(data)
                    df.to_excel(writer, index=False, sheet_name='Lecture Script')
                    worksheet = writer.sheets['Lecture Script']
                    
                    # Insert a column for images
                    worksheet.insert_cols(0)
                    worksheet.column_dimensions['A'].width = 80  # Image column
                    worksheet.column_dimensions['B'].width = 100  # Script column
                    
                    # Add column headers
                    worksheet['A1'] = 'Slide'
                    
                    # Add images to the first column
                    for i, img_base64 in enumerate(st.session_state.current_images, start=2):
                        try:
                            # Extract the actual base64 data from the data URL
                            if ';base64,' in img_base64:
                                img_base64 = img_base64.split(';base64,')[1]
                            
                            # Decode base64 image
                            image_bytes = base64.b64decode(img_base64)
                            image = Image.open(BytesIO(image_bytes))
                            
                            # Resize image to fit better in Excel cell
                            target_height = 300  # pixels
                            aspect_ratio = image.width / image.height
                            target_width = int(target_height * aspect_ratio)
                            image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
                            
                            # Save resized image to bytes
                            img_byte_arr = io.BytesIO()
                            image.save(img_byte_arr, format='PNG')
                            img_byte_arr.seek(0)
                            
                            # Add image to worksheet
                            img = openpyxl.drawing.image.Image(img_byte_arr)
                            img.anchor = f'A{i}'  # Position image in the first column
                            worksheet.add_image(img)
                            
                            # Adjust row height to fit image
                            worksheet.row_dimensions[i].height = 250  # Adjust as needed
                            
                        except Exception as e:
                            worksheet[f'A{i}'] = f"Error adding image {i-1}: {str(e)}"
                
                # Prepare download button
                output.seek(0)
                st.sidebar.download_button(
                    label="Export to Excel",
                    data=output,
                    file_name="lecture_script.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            except Exception as e:
                st.sidebar.error(f"Error creating Excel file: {str(e)}")

    def validate_inputs(self) -> bool:
        """Validate input directories"""
        return True

    def render_content(self):
        """Render main content area"""
        if not st.session_state.current_images:
            st.info("Please upload a PPTX file to begin.")
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

class ProgressManager:
    def __init__(self):
        try:
            import streamlit as st
            with st.sidebar:
                st.markdown(f"### Generation Progress ({len(st.session_state.current_images)} slides)")
                self.progress_container = st.empty()
                self.status_text = st.empty()
                self.time_text = st.empty()
            self.has_streamlit = True
            self.start_time = None
            self.last_progress = 0
        except:
            self.has_streamlit = False
    
    def update_progress(self, stage, progress, status):
        if self.has_streamlit:
            import time
            
            # Initialize start time if not set
            if self.start_time is None:
                self.start_time = time.time()
            
            current_time = time.time()
            elapsed_time = current_time - self.start_time
            
            # Only calculate remaining time if we have made some progress
            if progress > self.last_progress and progress < 100:
                progress_made = progress - self.last_progress
                time_per_percent = elapsed_time / progress if progress > 0 else 0
                remaining_percent = 100 - progress
                eta = time_per_percent * remaining_percent
                
                # Format times
                elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
                eta_str = time.strftime("%H:%M:%S", time.gmtime(eta))
                
                self.time_text.text(f"⏱️ Elapsed: {elapsed_str} | ETA: {eta_str}")
            elif progress >= 100:
                # Show total time at completion
                total_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
                self.time_text.text(f"✅ Total time: {total_time}")
            
            self.status_text.text(status)
            progress_value = progress / 100.0
            with self.progress_container:
                st.progress(progress_value, f"{stage}")
            
            self.last_progress = progress
    
    def cleanup(self):
        if self.has_streamlit:
            try:
                self.progress_container.empty()
                self.status_text.empty()
                self.time_text.empty()
            except:
                pass

if __name__ == "__main__":
    main()
