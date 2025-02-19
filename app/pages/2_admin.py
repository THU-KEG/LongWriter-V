import streamlit as st
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.models import User, Annotation
from src.auth import hash_password, verify_password
import os
from pathlib import Path
from collections import defaultdict

# Set page config to wide mode
st.set_page_config(
    page_title="标注管理平台",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def init_session_state():
    if 'admin_logged_in' not in st.session_state:
        st.session_state.admin_logged_in = False
    if 'selected_course' not in st.session_state:
        st.session_state.selected_course = None

def get_course_slides(major: str, course_name: str) -> list:
    """获取某个课程的所有幻灯片"""
    png_dir = Path(".") / major / course_name / "pngs"
    if not png_dir.exists():
        return []
    
    slides = []
    for png_file in sorted(png_dir.glob("*.png"), key=lambda x: int(x.stem)):
        slides.append({
            'slide_id': int(png_file.stem),
            'path': str(png_file)
        })
    return slides

def get_courses_for_major(major: str) -> list:
    """获取某个专业下的所有课程"""
    base_path = Path(".") / major
    if not base_path.exists():
        return []
    
    courses = []
    for course_dir in base_path.iterdir():
        if course_dir.is_dir() and (course_dir / "pngs").exists():
            courses.append(course_dir.name)
    return sorted(courses)

def admin_login():
    st.title("管理员登录")
    username = st.text_input("用户名")
    password = st.text_input("密码", type="password")
    
    if st.button("登录"):
        engine = create_engine('sqlite:///annotation.db')
        Session = sessionmaker(bind=engine)
        session = Session()
        
        user = session.query(User).filter_by(username=username, major="admin").first()
        if user and verify_password(password, user.password_hash):
            st.session_state.admin_logged_in = True
            st.success("登录成功！")
            st.rerun()
        else:
            st.error("用户名或密码错误")

class AdminPlatform:
    def __init__(self):
        self.engine = create_engine('sqlite:///annotation.db')
        self.Session = sessionmaker(bind=self.engine)
        init_session_state()

    def show_course_progress(self, annotator, course_name):
        """显示某个课程的详细进度"""
        st.markdown(f"### {course_name} 标注进度")
        
        session = self.Session()
        
        # 获取该课程的所有幻灯片
        slides = get_course_slides(annotator.major, course_name)
        
        # 获取已标注的幻灯片
        annotations = {
            a.slide_id: a for a in session.query(Annotation).filter_by(
                annotator_id=annotator.id,
                course_name=course_name
            ).all()
        }
        
        # 创建列标题
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("### 幻灯片")
        with col2:
            st.markdown("### 原始文本")
        with col3:
            st.markdown("### 修改后文本")
        
        # 分页显示
        ITEMS_PER_PAGE = 5
        total_pages = (len(slides) + ITEMS_PER_PAGE - 1) // ITEMS_PER_PAGE
        
        # 页面选择
        page = st.selectbox(
            "选择页面",
            range(1, total_pages + 1),
            format_func=lambda x: f"第 {x} 页 (幻灯片 {(x-1)*ITEMS_PER_PAGE + 1} - {min(x*ITEMS_PER_PAGE, len(slides))})",
            key=f"page_select_{course_name}"
        )
        
        # 计算当前页的幻灯片范围
        start_idx = (page - 1) * ITEMS_PER_PAGE
        end_idx = min(start_idx + ITEMS_PER_PAGE, len(slides))
        current_slides = slides[start_idx:end_idx]
        
        # 显示当前页的幻灯片
        for slide in current_slides:
            cols = st.columns(3)
            
            # 显示幻灯片
            with cols[0]:
                st.markdown(f"**幻灯片 {slide['slide_id']}**")
                st.image(slide['path'])
            
            annotation = annotations.get(slide['slide_id'])
            
            # 显示原始文本
            with cols[1]:
                if annotation:
                    st.text_area(
                        "",
                        value=annotation.original_script,
                        height=300,
                        disabled=True,
                        label_visibility="collapsed",
                        key=f"original_{course_name}_{slide['slide_id']}"
                    )
                else:
                    st.info("未标注")
            
            # 显示修改后的文本
            with cols[2]:
                if annotation:
                    st.text_area(
                        "",
                        value=annotation.modified_script,
                        height=300,
                        disabled=True,
                        label_visibility="collapsed",
                        key=f"modified_{course_name}_{slide['slide_id']}"
                    )
                else:
                    st.info("未标注")
            
            st.markdown("---")
        
        # 显示分页导航
        col1, col2, col3 = st.columns(3)
        with col1:
            if page > 1:
                if st.button("← 上一页", key=f"prev_{course_name}"):
                    st.query_params["page"] = str(page - 1)
                    st.rerun()
        with col3:
            if page < total_pages:
                if st.button("下一页 →", key=f"next_{course_name}"):
                    st.query_params["page"] = str(page + 1)
                    st.rerun()

    def show_annotator_progress(self):
        st.title("标注进度")
        
        session = self.Session()
        
        # 获取所有标注者信息
        annotators = session.query(User).filter(User.major != "admin").all()
        
        # 获取当前的查询参数
        view_annotator = st.query_params.get("annotator", None)
        view_course = st.query_params.get("course", None)
        
        # 如果有查询参数，显示课程详情
        if view_annotator and view_course:
            annotator = session.query(User).filter_by(id=int(view_annotator)).first()
            if annotator:
                col1, col2 = st.columns([1, 11])
                with col1:
                    if st.button("← 返回"):
                        st.query_params.clear()
                        st.rerun()
                with col2:
                    st.title(f"{annotator.username} - {view_course}")
                self.show_course_progress(annotator, view_course)
            return
        
        for annotator in annotators:
            with st.expander(f"标注者: {annotator.username} ({annotator.major})"):
                # 获取该专业下的所有课程
                courses = get_courses_for_major(annotator.major)
                
                # 获取该标注者的所有标注
                annotations = session.query(Annotation).filter_by(annotator_id=annotator.id).all()
                annotation_dict = defaultdict(list)
                for a in annotations:
                    annotation_dict[a.course_name].append(a.slide_id)
                
                # 显示每个课程的进度
                for course in courses:
                    slides = get_course_slides(annotator.major, course)
                    total_slides = len(slides)
                    completed_slides = len(annotation_dict[course])
                    
                    st.markdown(f"#### {course}")
                    
                    col1, col2, col3 = st.columns([2, 2, 1])
                    
                    with col1:
                        st.metric("总页数", total_slides)
                    with col2:
                        st.metric("已完成页数", completed_slides)
                    with col3:
                        if st.button("查看详情", key=f"view_{annotator.id}_{course}"):
                            st.query_params["annotator"] = str(annotator.id)
                            st.query_params["course"] = course
                            st.rerun()
                    
                    # 显示进度条
                    if total_slides > 0:
                        progress = completed_slides / total_slides
                        st.progress(progress)
                        st.write(f"完成进度: {progress:.1%}")
                    
                    st.markdown("---")
                
                if annotations:
                    st.write("最近更新时间:", annotations[-1].updated_at)

    def run(self):
        if not st.session_state.admin_logged_in:
            admin_login()
        else:
            self.show_annotator_progress()

# Initialize and run the admin platform
platform = AdminPlatform()
platform.run() 