import os
from pathlib import Path
import json
from typing import Dict, List, Optional, Tuple
from PIL import Image
import base64
from io import BytesIO
from .utils import logger

class DataManager:
    def __init__(self, data_dir: str = "."):
        """初始化数据管理器
        Args:
            data_dir: 数据目录的路径，默认为项目的data/dpo目录
        """
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            logger.warning("Data directory not found: %s", data_dir)
            logger.info("Creating data directory")
            self.data_dir.mkdir(parents=True, exist_ok=True)
        logger.info("DataManager initialized with data directory: %s", self.data_dir)
    
    def save_script(self, major: str, course_name: str, slide_id: int, content: str) -> None:
        """保存标注脚本到文件系统"""
        try:
            # 确保目录存在
            base_path = self.data_dir / major / course_name / "annotated_scripts"
            base_path.mkdir(parents=True, exist_ok=True)
            
            # 保存文件
            file_path = base_path / f"{slide_id}.txt"
            file_path.write_text(content, encoding='utf-8')
            
            logger.info(
                "Script saved successfully - Major: %s, Course: %s, Slide: %d",
                major, course_name, slide_id
            )
            
        except Exception as e:
            error_msg = f"Failed to save script - Major: {major}, Course: {course_name}, Slide: {slide_id}"
            logger.error("%s - Error: %s", error_msg, str(e))
            raise Exception(error_msg) from e
    
    def load_slides_for_major(self, major: str) -> List[Dict]:
        """加载专业的所有幻灯片"""
        try:
            slides = []
            major_dir = self.data_dir / major
            
            if not major_dir.exists():
                logger.warning("Major directory not found: %s", major_dir)
                return []
            
            # 首先获取所有课程并排序
            courses = sorted([d for d in major_dir.iterdir() if d.is_dir()], 
                           key=lambda x: x.name)
            
            for course_dir in courses:
                if not course_dir.is_dir():
                    continue
                    
                course_name = course_dir.name
                png_dir = course_dir / "pngs"
                script_dir = course_dir / "scripts"
                
                if not png_dir.exists():
                    logger.warning("PNG directory not found for course: %s", course_name)
                    continue
                
                logger.info("Loading slides from course: %s", course_name)
                
                # 获取所有PNG文件并按数字顺序排序
                png_files = sorted(
                    [f for f in png_dir.glob("*.png")],
                    key=lambda x: int(x.stem)
                )
                
                # 加载所有PNG文件
                for png_file in png_files:
                    slide_id = int(png_file.stem)
                    script_file = script_dir / f"{slide_id}.txt"
                    
                    with open(png_file, "rb") as f:
                        image_data = base64.b64encode(f.read()).decode()
                    
                    slide_info = {
                        "slide_id": slide_id,
                        "course_name": course_name,
                        "image_base64": f"data:image/png;base64,{image_data}",
                        "has_original_script": script_file.exists(),
                        "script": script_file.read_text(encoding='utf-8').strip() if script_file.exists() else ""
                    }
                    slides.append(slide_info)
                    logger.debug("Loaded slide %d from course %s", slide_id, course_name)
            
            logger.info(
                "Loaded %d slides for major: %s",
                len(slides), major
            )
            return slides
            
        except Exception as e:
            logger.error("Failed to load slides for major: %s - Error: %s", major, str(e))
            return []
    
    def get_course_info(self, major: str, session, user_id: int) -> List[Tuple[str, int, List[bool]]]:
        """获取课程信息和标注状态"""
        try:
            from .models import Annotation
            
            course_info = []
            major_dir = self.data_dir / major
            
            if not major_dir.exists():
                logger.warning("Major directory not found: %s", major_dir)
                return []
            
            # 获取用户的所有标注
            annotations = {
                (a.course_name, a.slide_id): a.is_completed
                for a in session.query(Annotation).filter_by(
                    annotator_id=user_id,
                    major=major
                ).all()
            }
            
            logger.info("Found %d existing annotations for user %d", len(annotations), user_id)
            
            for course_dir in major_dir.iterdir():
                if not course_dir.is_dir():
                    continue
                    
                course_name = course_dir.name
                png_dir = course_dir / "pngs"
                
                if not png_dir.exists():
                    logger.warning("PNG directory not found for course: %s", course_name)
                    continue
                
                # 获取所有幻灯片
                slide_files = sorted(png_dir.glob("*.png"))
                total_slides = len(slide_files)
                
                # 检查每张幻灯片的标注状态
                annotation_status = []
                for png_file in slide_files:
                    slide_id = int(png_file.stem)
                    is_annotated = annotations.get((course_name, slide_id), False)
                    annotation_status.append(is_annotated)
                
                course_info.append((course_name, total_slides, annotation_status))
                logger.info(
                    "Course %s: %d slides, %d annotated",
                    course_name, total_slides, sum(annotation_status)
                )
            
            logger.info(
                "Retrieved course info for major: %s - Found %d courses",
                major, len(course_info)
            )
            return course_info
            
        except Exception as e:
            logger.error(
                "Failed to get course info - Major: %s, User: %d - Error: %s",
                major, user_id, str(e)
            )
            return [] 