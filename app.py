
import streamlit as st
import sqlite3
import datetime
import base64
import io
import pandas as pd
from PIL import Image, ImageEnhance
import dashscope
from dashscope import MultiModalConversation

# ------------------- 页面配置 -------------------
st.set_page_config(
    page_title="无人机巡检智能分析系统",
    page_icon="🛩️",
    layout="wide"
)

# ------------------- 初始化数据库（存储专家反馈） -------------------
def init_db():
    conn = sqlite3.connect('feedback.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_name TEXT,
            query TEXT,
            response TEXT,
            rating INTEGER,
            comment TEXT,
            timestamp TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()
# ------------------- 工具函数 -------------------
def image_to_base64(image, max_size=(1024, 1024)):
    image.thumbnail(max_size, Image.Resampling.LANCZOS)
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=85)
    img_base64 = base64.b64encode(buffered.getvalue()).decode()
    return img_base64

def enhance_image(image):
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.2)
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(1.5)
    return image

# ------------------- 调用通义千问VL API -------------------
def call_qwen_vl(image, prompt, api_key):
    if api_key:
        dashscope.api_key = api_key
    else:
        st.error("请提供有效的API密钥")
        return None

    img_base64 = image_to_base64(image)
    messages = [
        {
            "role": "user",
            "content": [
                {"image": f"data:image/jpeg;base64,{img_base64}"},
                {"text": prompt}
            ]
        }
    ]
    try:
        response = MultiModalConversation.call(
            model="qwen-vl-plus",
            messages=messages,
            max_tokens=800
        )
        if response.status_code == 200:
            output = response.output.choices[0].message.content[0]["text"]
            return output
        else:
            st.error(f"API调用失败: {response.message}")
            return None
    except Exception as e:
        st.error(f"异常: {e}")
        return None
    # ------------------- 提示词模板 -------------------
PROMPT_TEMPLATES = {
    "简单描述": "请用中文简要描述这张无人机巡检图片中的主要设备和场景。",
    "缺陷分析": "你是一位电力巡检专家。请分析这张图片，列出所有可见的缺陷，包括位置和严重程度（高/中/低）。",
    "报告生成": """请根据这张无人机巡检图像生成一份专业报告，包含以下部分：
1. 设备概况：主要设备类型和状态
2. 隐患清单：逐项列出缺陷及位置
3. 处理建议：针对每个隐患给出维护建议
使用行业规范术语。""",
    "批量报告": "请根据以上多张图片的综合信息，生成一份整体巡检报告，包括设备统计、主要隐患和优先级排序。"
}

# ------------------- 保存反馈 -------------------
def save_feedback(image_name, query, response, rating, comment=""):
    conn = sqlite3.connect('feedback.db')
    c = conn.cursor()
    c.execute('''
        INSERT INTO feedback (image_name, query, response, rating, comment, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (image_name, query, response, rating, comment, datetime.datetime.now().isoformat()))
    conn.commit()
    conn.close()

# ------------------- 侧边栏配置 -------------------
with st.sidebar:
    st.header("⚙️ 配置")
    api_key = st.text_input("通义千问API密钥", type="password", 
                             help="从DashScope获取，也可在.streamlit/secrets.toml中设置")
    if st.secrets and "DASHSCOPE_API_KEY" in st.secrets:
        api_key = st.secrets["DASHSCOPE_API_KEY"]
        st.success("已从secrets加载API密钥")
    
    st.divider()
    st.header("📋 功能说明")
    st.markdown("""
    - **问答模式**：上传单张图片，输入问题，获得专业回答
    - **报告模式**：上传单张或多张图片，生成标准化巡检报告
    - **专家反馈**：对结果点赞/点踩，帮助系统优化
    """)
    st.divider()
    if st.button("🗑️ 清空所有反馈"):
        conn = sqlite3.connect('feedback.db')
        c = conn.cursor()
        c.execute("DELETE FROM feedback")
        conn.commit()
        conn.close()
        st.success("反馈已清空")

# ------------------- 主界面 -------------------
st.title("🛩️ 无人机巡检图像智能问答与报告生成")
st.markdown("基于大语言模型的跨模态智能分析，助力巡检效率提升")

mode = st.radio("选择模式", ["问答模式", "报告模式"], horizontal=True)

if mode == "问答模式":
    uploaded_files = st.file_uploader("上传一张巡检图片", type=["jpg", "jpeg", "png"], accept_multiple_files=False)
else:
    uploaded_files = st.file_uploader("上传一张或多张巡检图片", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    files = uploaded_files if isinstance(uploaded_files, list) else [uploaded_files]
    
    cols = st.columns(min(len(files), 4))
    images = []
    for i, file in enumerate(files):
        img = Image.open(file)
        images.append(img)
        with cols[i % 4]:
            st.image(img, caption=file.name, use_column_width=True)
    
    enhance = st.checkbox("启用图像增强（对比度/锐度）", value=True)
    if enhance:
        images = [enhance_image(img) for img in images]
    
    if mode == "问答模式":
        st.subheader("💬 智能问答")
        question = st.text_input("请输入您的问题", placeholder="例如：这张图片中有哪些缺陷？")
        if st.button("提交问题", type="primary") and question:
            if not api_key:
                st.error("请在侧边栏输入API密钥")
            else:
                with st.spinner("模型正在思考..."):
                    prompt = f"基于巡检图像，请回答：{question}"
                    result = call_qwen_vl(images[0], prompt, api_key)
                if result:
                    st.markdown("#### 回答")
                    st.write(result)
                    col1, col2, col3 = st.columns([1,1,5])
                    with col1:
                        if st.button("👍 有帮助"):
                            save_feedback(files[0].name, question, result, 1)
                            st.success("感谢反馈！")
                    with col2:
                        if st.button("👎 无帮助"):
                            comment = st.text_input("请输入改进建议（可选）", key="comment")
                            if st.button("提交反馈"):
                                save_feedback(files[0].name, question, result, -1, comment)
                                st.success("感谢反馈！")
    else:
        st.subheader("📄 报告生成")
        report_type = st.selectbox("报告类型", ["单图报告", "批量综合报告"])
        
        if st.button("生成报告", type="primary"):
            if not api_key:
                st.error("请在侧边栏输入API密钥")
            else:
                if report_type == "单图报告" and len(images) == 1:
                    with st.spinner("正在生成报告..."):
                        result = call_qwen_vl(images[0], PROMPT_TEMPLATES["报告生成"], api_key)
                    if result:
                        st.markdown("#### 巡检报告")
                        st.markdown(result)
                        if st.button("👍 报告有帮助"):
                            save_feedback(files[0].name, "单图报告", result, 1)
                            st.success("感谢反馈！")
                
                elif report_type == "批量综合报告" and len(images) > 1:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    per_image_reports = []
                    
                    for i, img in enumerate(images):
                        status_text.text(f"正在分析第 {i+1}/{len(images)} 张图片...")
                        brief = call_qwen_vl(img, "请用一句话描述这张图片的主要缺陷或设备状态。", api_key)
                        if brief:
                            per_image_reports.append(f"图片{i+1}({files[i].name}): {brief}")
                        progress_bar.progress((i+1)/len(images))
                        status_text.text("正在生成综合报告...")
                    combined_prompt = f"以下是多张无人机巡检图片的摘要信息：\n" + "\n".join(per_image_reports) + "\n\n请根据以上信息生成一份综合巡检报告，包括设备统计、主要隐患清单及处理建议。"
                    final_report = call_qwen_vl(images[0], combined_prompt, api_key)
                    
                    if final_report:
                        st.markdown("#### 综合巡检报告")
                        st.markdown(final_report)
                        if st.button("👍 报告有帮助"):
                            save_feedback("批量", "批量报告", final_report, 1)
                            st.success("感谢反馈！")
                else:
                    st.warning("请确保图片数量与报告类型匹配（单图报告需一张，批量报告需多张）")
else:
    st.info("请上传图片开始使用")

# ------------------- 查看历史反馈（管理员） -------------------
with st.expander("📊 专家反馈记录（仅管理员查看）"):
    conn = sqlite3.connect('feedback.db')
    df = pd.read_sql_query("SELECT * FROM feedback ORDER BY timestamp DESC", conn)
    conn.close()
    if not df.empty:
        st.dataframe(df)
    else:
        st.write("暂无反馈记录。")
        progress_bar = st.progress(0)
status_text = st.empty()
for i, img in enumerate(images):
    status_text.text(f"正在分析第 {i+1}/{len(images)} 张...")


import streamlit as st
from PIL import Image
import datetime

# 初始化 session_state
if 'history' not in st.session_state:
    st.session_state['history'] = []

# 文件上传
uploaded_files = st.file_uploader(
    "上传巡检图片", 
    type=["jpg", "jpeg", "png"], 
    accept_multiple_files=True
)

images = []
if uploaded_files:
    for file in uploaded_files:
        img = Image.open(file)
        images.append(img)
    
    # 只有当 images 不为空时才执行循环
    if images:
        # 初始化进度条和状态文本
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # 遍历图片处理
        for i, img in enumerate(images):
            # 你的处理逻辑（比如调用模型分析）
            # ----------------------
            # 示例：模拟分析过程
            import time
            time.sleep(0.5)
            # ----------------------
            
            # 更新进度
            progress_bar.progress((i+1)/len(images))
            status_text.text(f"正在分析第 {i+1}/{len(images)} 张图片...")
        
        # 所有图片分析完成
        status_text.text("分析完成！")
        
        # 假设得到分析结果（示例）
        question = "图片分析结果"
        result = "图片中未发现异常"
        
        # 保存历史记录
        st.session_state['history'].append({
            "question": question,
            "answer": result,
            "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
else:
    st.info("请上传图片后再继续")

# 在 session_state 中存储历史
if f'history_{username}' not in st.session_state:
    st.session_state[f'history_{username}'] = []

# 添加问答后
st.session_state[f'history_{username}'].append({
    "question": question,
    "answer": result,
    "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
})

# 在侧边栏显示历史
with st.sidebar.expander("📜 历史记录"):
    for item in st.session_state[f'history_{username}'][-10:]:  # 最近10条
        st.markdown(f"**Q:** {item['question'][:30]}...")
        st.markdown(f"**A:** {item['answer'][:50]}...")
        st.caption(item['time'])
        
