import streamlit as st
from datetime import datetime
from src.utils import logger

# Set page config to wide mode and hide sidebar
st.set_page_config(
    page_title="讲稿标注平台",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Hide sidebar completely
st.markdown("""
<style>
    [data-testid="collapsedControl"] {display: none;}
    section[data-testid="stSidebar"] {display: none;}
</style>
""", unsafe_allow_html=True)

import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from PIL import Image
import base64
from io import BytesIO
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.models import init_db, User, Annotation
from src.auth import (
    init_auth, login_user, logout_user, 
    is_logged_in, get_current_user, create_user
)
from src.data import DataManager
from typing import List, Dict

# Constants
MAJORS = [
    "历史", "食品科学与工程", "机械工程", "园林", "土木工程", 
    "金融", "电气工程", "计算机", "物理", "数学"
]

class AnnotationPlatform:
    def __init__(self, title="讲稿标注平台"):
        self.title = title
        
        # Initialize database
        db_path = "annotation.db"
        self.engine = init_db(f"sqlite:///{db_path}")
        self.Session = sessionmaker(bind=self.engine)
        
        # Initialize authentication
        init_auth()
        
        # Initialize data manager with correct path
        self.data_manager = DataManager(data_dir=".")
        
        # Initialize session state
        if 'page' not in st.session_state:
            st.session_state.page = 'login'
        if 'selected_course' not in st.session_state:
            st.session_state.selected_course = None
        if 'current_slide_index' not in st.session_state:
            st.session_state.current_slide_index = 0
            
        logger.info("AnnotationPlatform initialized")
        
    def render_login(self):
        st.header("登录")
        
        username = st.text_input("用户名")
        password = st.text_input("密码", type="password")
        
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("登录", key="login_button", use_container_width=True):
                with self.Session() as session:
                    if login_user(session, username, password):
                        st.success("登录成功！")
                        st.rerun()
                    else:
                        st.error("用户名或密码错误")
        
        st.markdown("---")
        st.markdown("还没有账号？")
        if st.button("注册新用户", key="to_register"):
            st.session_state.page = 'register'
            st.rerun()
    
    def render_register(self):
        st.header("新用户注册")
        
        new_username = st.text_input("用户名")
        new_password = st.text_input("密码", type="password")
        major = st.selectbox("选择专业", MAJORS)
        
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("注册", key="register_button", use_container_width=True):
                with self.Session() as session:
                    if create_user(session, new_username, new_password, major):
                        st.success("注册成功！请返回登录。")
                        st.session_state.page = 'login'
                        st.rerun()
                    else:
                        st.error("注册失败。用户名可能已被使用。")
        
        st.markdown("---")
        st.markdown("已有账号？")
        if st.button("返回登录", key="to_login"):
            st.session_state.page = 'login'
            st.rerun()
    
    def save_annotation(self, slide_id, course_name, original_script, modified_script):
        """Save the modified script to both database and file system"""
        try:
            user = get_current_user()
            if not user:
                error_msg = "No user found in session"
                logger.error(error_msg)
                st.error("会话已过期，请重新登录")
                return False
            
            logger.info(
                "Attempting to save annotation - User: %s, Major: %s, Course: %s, Slide: %d",
                user['username'], user['major'], course_name, slide_id
            )
            
            # Save to database
            with self.Session() as session:
                try:
                    annotation = session.query(Annotation).filter_by(
                        slide_id=slide_id,
                        annotator_id=user['id'],
                        course_name=course_name
                    ).first()
                    
                    if annotation:
                        annotation.modified_script = modified_script
                        annotation.is_completed = True
                        annotation.updated_at = datetime.utcnow()
                        logger.info("Updating existing annotation record")
                    else:
                        annotation = Annotation(
                            slide_id=slide_id,
                            major=user['major'],
                            course_name=course_name,
                            original_script=original_script,
                            modified_script=modified_script,
                            annotator_id=user['id'],
                            is_completed=True
                        )
                        session.add(annotation)
                        logger.info("Creating new annotation record")
                    
                    session.commit()
                    session.refresh(annotation)
                    logger.info("Database save successful")
                    
                except Exception as e:
                    session.rollback()
                    error_msg = f"数据库保存失败: {str(e)}"
                    logger.error(error_msg)
                    st.error(error_msg)
                    return False
            
            # Save to file system
            try:
                self.data_manager.save_script(
                    user['major'],
                    course_name,
                    slide_id,
                    modified_script
                )
                logger.info("File system save successful")
                
            except Exception as e:
                error_msg = f"文件系统保存失败: {str(e)}"
                logger.error(error_msg)
                st.error(error_msg)
                return False
            
            logger.info("Annotation saved successfully")
            return True
            
        except Exception as e:
            error_msg = f"保存过程发生错误: {str(e)}"
            logger.error(error_msg)
            st.error(error_msg)
            return False
    
    def render_course_selection(self, major: str):
        """Render course selection interface"""
        st.subheader("课程列表")
        
        with self.Session() as session:
            user = get_current_user()
            course_info = self.data_manager.get_course_info(major, session, user['id'])
            
            if not course_info:
                st.warning(f"未找到{major}的课程")
                return False
            
            # Create course selection cards
            for course_name, total_slides, annotation_status in course_info:
                with st.expander(f"📚 {course_name}", expanded=True):
                    # Show progress overview
                    completed_count = sum(annotation_status)
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        # Show progress bar
                        progress = completed_count / total_slides if total_slides > 0 else 0
                        st.progress(progress)
                        st.write(f"已标注 {completed_count}/{total_slides} 页")
                    
                    with col2:
                        if st.button("开始标注", key=f"select_{course_name}"):
                            st.session_state.selected_course = course_name
                            st.session_state.current_slide_index = 0
                            st.rerun()
        
        return True
    
    def render_slide_annotation(self, course_name: str, slides: List[Dict]):
        """Render the slide annotation interface"""
        # Get current user's annotations
        user = get_current_user()
        with self.Session() as session:
            annotations = {
                (a.course_name, a.slide_id): a
                for a in session.query(Annotation).filter_by(
                    annotator_id=user['id'],
                    course_name=course_name
                ).all()
            }
        
        # Add a back button and show course name
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader(f"课程：{course_name}")
        with col2:
            if st.button("返回课程列表", key="back_to_courses"):
                st.session_state.selected_course = None
                st.rerun()
        
        # Show overall progress
        completed_count = sum(1 for a in annotations.values() if a.is_completed)
        total_slides = len(slides)
        st.progress(completed_count / total_slides)
        st.write(f"已完成标注: {completed_count}/{total_slides}")
        
        # 分页显示
        ITEMS_PER_PAGE = 3  # 每页显示3个幻灯片
        total_pages = (len(slides) + ITEMS_PER_PAGE - 1) // ITEMS_PER_PAGE
        
        # 页面选择
        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            current_page = st.number_input(
                "当前页面",
                min_value=1,
                max_value=total_pages,
                value=min(total_pages, max(1, st.session_state.current_slide_index // ITEMS_PER_PAGE + 1)),
                key="page_number"
            )
        
        # 计算当前页的幻灯片范围
        start_idx = (current_page - 1) * ITEMS_PER_PAGE
        end_idx = min(start_idx + ITEMS_PER_PAGE, len(slides))
        current_slides = slides[start_idx:end_idx]
        
        # 更新当前幻灯片索引
        st.session_state.current_slide_index = start_idx
        
        # Display current page slides
        for slide in current_slides:
            slide_id = slide['slide_id']
            annotation = annotations.get((course_name, slide_id))
            
            # Create three columns for the layout
            cols = st.columns([2, 1.5, 1.5])
            
            # First column - Image
            with cols[0]:
                st.markdown(f"**第 {slide_id} 页**")
                st.image(
                    slide['image_base64'],
                    use_container_width=True
                )
            
            # Second column - Original script (read-only)
            with cols[1]:
                with st.container():
                    if slide['has_original_script']:
                        st.markdown("**原始讲稿**")
                        st.markdown(
                            f"""<div style="
                                background-color: #f0f2f6;
                                border-radius: 4px;
                                padding: 1rem;
                                height: 200px;
                                overflow-y: auto;
                                white-space: pre-wrap;
                                font-family: monospace;
                                line-height: 1.5;
                                margin-top: 0.5rem;
                            ">{slide['script']}</div>""",
                            unsafe_allow_html=True
                        )
                    else:
                        st.warning("暂无原始讲稿")
            
            # Third column - Annotation area
            with cols[2]:
                with st.container():
                    # Show status icon
                    if annotation and annotation.is_completed:
                        status = "✅ 已标注"
                    else:
                        status = "📝 待标注"
                    st.markdown(f"**标注状态：{status}**")
                    
                    # Initialize with original script if no annotation exists
                    current_script = annotation.modified_script if annotation else slide['script']
                    modified_script = st.text_area(
                        "标注内容",
                        value=current_script,
                        height=200,  # Reduced height
                        key=f"annotation_{slide_id}",
                        label_visibility="collapsed"
                    )
                    
                    if st.button("保存标注", key=f"save_{slide_id}"):
                        if self.save_annotation(
                            slide_id,
                            course_name,
                            slide['script'],
                            modified_script
                        ):
                            st.success("保存成功！")
                            # Force a page refresh to update the status
                            st.rerun()
            
            # Add some vertical spacing between slides
            st.markdown("---")
        
        # 分页导航
        col1, col2, col3 = st.columns(3)
        with col1:
            if current_page > 1:
                if st.button("← 上一页"):
                    st.session_state.current_slide_index = max(0, start_idx - ITEMS_PER_PAGE)
                    st.rerun()
        with col3:
            if current_page < total_pages:
                if st.button("下一页 →"):
                    st.session_state.current_slide_index = min(len(slides) - 1, end_idx)
                    st.rerun()
    
    def render_annotation_interface(self):
        """Render the main annotation interface"""
        try:
            user = get_current_user()
            if not user:
                logger.warning("No user found in session")
                st.error("会话已过期，请重新登录")
                logout_user()
                st.rerun()
                return
            
            logger.info("Rendering annotation interface for user: %s", user['username'])
            
            st.header(f"欢迎 {user['username']} ({user['major']})")
            
            if st.button("登出", key="logout"):
                logger.info("User logged out: %s", user['username'])
                logout_user()
                st.rerun()
            
            # Show either course selection or slide annotation
            if not st.session_state.selected_course:
                self.render_course_selection(user['major'])
            else:
                # Load slides for the selected course
                slides = [
                    slide for slide in self.data_manager.load_slides_for_major(user['major'])
                    if slide['course_name'] == st.session_state.selected_course
                ]
                
                if slides:
                    self.render_slide_annotation(st.session_state.selected_course, slides)
                else:
                    error_msg = f"无法加载课程 {st.session_state.selected_course} 的幻灯片"
                    logger.error(error_msg)
                    st.error(error_msg)
                    st.session_state.selected_course = None
                    
        except Exception as e:
            error_msg = f"渲染界面错误: {str(e)}"
            logger.error(error_msg)
            st.error(error_msg)
            logout_user()
            st.rerun()
    
    def render(self):
        st.title(self.title)
        
        if not is_logged_in():
            if st.session_state.page == 'login':
                self.render_login()
            else:
                self.render_register()
        else:
            self.render_annotation_interface()

# Initialize and run the annotation platform
platform = AnnotationPlatform()
platform.render() 