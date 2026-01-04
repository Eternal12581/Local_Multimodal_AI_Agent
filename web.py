"""Web界面"""
from flask import Flask, request, render_template_string, Response
from src.image_processor import ImageProcessor
from src.utils import logger
import os
import glob
from datetime import datetime

def import_document_processor():
    try:
        from src.document_processor import DocumentProcessor
        return DocumentProcessor
    except Exception as e:
        logger.error(f"导入论文处理模块失败: {str(e)}")
        return None

# 初始化Flask应用和处理器
app = Flask(__name__)
image_processor = ImageProcessor()
document_processor_cls = import_document_processor()

# 全局HTML模板
FULL_FUNCTION_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>本地 AI 智能文献与图像管理助手</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: Arial, sans-serif; max-width: 1600px; margin: 0 auto; padding: 20px; background-color: #f5f5f5; }
        .container { background-color: #fff; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); padding: 30px; }
        h1 { text-align: center; color: #333; margin-bottom: 40px; }
        .tab-container { margin-bottom: 30px; }
        .tab-buttons { display: flex; border-bottom: 1px solid #ccc; margin-bottom: 20px; }
        .tab-btn { padding: 12px 24px; border: none; background: none; cursor: pointer; font-size: 16px; color: #666; }
        .tab-btn.active { color: #007bff; border-bottom: 2px solid #007bff; font-weight: bold; }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
        .function-card { 
            display: flex; 
            justify-content: space-between; 
            background-color: #f9f9f9; 
            border-radius: 8px; 
            padding: 25px; 
            margin-bottom: 30px; 
            gap: 20px;
        }
        .form-area { width: 55%; }
        .result-area { 
            width: 43%; 
            padding: 15px; 
            background-color: #f8f9fa; 
            border-radius: 6px; 
            border: 1px solid #eee; 
            align-self: flex-start;
            min-height: 200px; 
            max-height: 800px; 
            overflow-y: auto; 
        }
        h2 { color: #444; margin-bottom: 30px; padding-bottom: 15px; border-bottom: 1px solid #eee; }
        h3 { color: #555; margin-bottom: 20px; font-size: 18px; }
        .form-group { margin-bottom: 18px; }
        label { display: inline-block; width: 180px; color: #666; font-size: 14px; }
        input[type="text"], input[type="number"] { 
            width: 400px; 
            padding: 8px 12px; 
            border: 1px solid #ccc; 
            border-radius: 4px; 
            font-size: 14px; 
        }
        input[type="checkbox"] { margin-left: 180px; margin-top: 10px; }
        .checkbox-label { width: auto; margin-left: 5px; }
        button { 
            padding: 10px 24px; 
            background-color: #007bff; 
            color: #fff; 
            border: none; 
            border-radius: 4px; 
            cursor: pointer; 
            font-size: 14px; 
            margin-top: 10px;
        }
        button:hover { background-color: #0056b3; }
        pre { font-size: 12px; line-height: 1.6; color: #333; overflow-x: auto; white-space: pre-wrap; }
        .success { color: #28a745; }
        .error { color: #dc3545; }
        .info { color: #17a2b8; }
        .empty-result { color: #999; font-style: italic; font-size: 14px; }
        .loading { color: #ffc107; font-style: italic; }
    </style>
    <script>
        // 标签页切换功能
        function switchTab(tabName) {
            const tabContents = document.getElementsByClassName('tab-content');
            const tabBtns = document.getElementsByClassName('tab-btn');
            for (let i = 0; i < tabContents.length; i++) {
                tabContents[i].classList.remove('active');
                tabBtns[i].classList.remove('active');
            }
            document.getElementById(tabName).classList.add('active');
            event.target.classList.add('active');
        }

        // 核心：AJAX异步提交通用函数
        function ajaxSubmit(formId, resultId, apiUrl) {
            // 1. 强校验：确保表单和结果区域元素存在（避免报错导致阻止默认提交失效）
            const form = document.getElementById(formId);
            const resultArea = document.getElementById(resultId);
            if (!form || !resultArea) {
                console.error("表单或结果区域元素不存在！", formId, resultId);
                return false; // 即使元素不存在，也返回false阻止提交
            }

            const formData = new FormData(form);

            // 2. 展示加载状态
            resultArea.innerHTML = `<h3>操作结果</h3><pre><span class="loading">正在处理，请稍候...</span></pre>`;

            // 3. 创建AJAX对象
            const xhr = new XMLHttpRequest();
            xhr.open('POST', apiUrl, true); // true = 异步请求

            // 4. 接收响应
            xhr.onload = function() {
                if (xhr.status >= 200 && xhr.status < 300) {
                    // 成功：更新结果区域内容，留在原位置
                    resultArea.innerHTML = `<h3>操作结果</h3><pre>${xhr.responseText}</pre>`;
                } else {
                    // 失败：展示错误信息，不跳转
                    resultArea.innerHTML = `<h3>操作结果</h3><pre><span class="error">❌ 请求失败：${xhr.status} - ${xhr.statusText}</span></pre>`;
                }
            };

            // 5. 网络错误处理
            xhr.onerror = function() {
                resultArea.innerHTML = `<h3>操作结果</h3><pre><span class="error">❌ 网络错误：无法连接到服务器</span></pre>`;
            };

            // 6. 发送请求
            xhr.send(formData);

            // 7. 强制返回false：阻止表单默认提交行为
            console.log("阻止表单默认提交，仅局部更新结果");
            return false;
        }

        // ---------------------- 论文模块AJAX提交函数 ----------------------
        // 1. 添加单篇论文
        function submitAddSinglePaper() {
            // 明确返回AJAX函数结果，确保阻止默认提交
            return ajaxSubmit('addSinglePaperForm', 'addSinglePaperResult', '/add_single_paper');
        }

        // 2. 批量添加论文
        function submitBatchAddPapers() {
            return ajaxSubmit('batchAddPapersForm', 'batchAddPapersResult', '/batch_add_papers');
        }

        // 3. 一键整理论文
        function submitOrganizePapers() {
            return ajaxSubmit('organizePapersForm', 'organizePapersResult', '/organize_papers');
        }

        // 4. 语义搜索论文
        function submitSearchPapers() {
            return ajaxSubmit('searchPapersForm', 'searchPapersResult', '/search_papers');
        }

        // ---------------------- 图像模块AJAX提交函数 ----------------------
        // 1. 添加单张图像
        function submitAddSingleImage() {
            return ajaxSubmit('addSingleImageForm', 'addSingleImageResult', '/add_single_image');
        }

        // 2. 批量添加图像
        function submitBatchAddImages() {
            return ajaxSubmit('batchAddImagesForm', 'batchAddImagesResult', '/batch_add_images');
        }

        // 3. 文本搜索图像
        function submitSearchImages() {
            return ajaxSubmit('searchImagesForm', 'searchImagesResult', '/search_images');
        }

        window.onload = function() {
            // 获取所有功能表单，添加onsubmit事件，强制返回false
            const allForms = [
                'addSinglePaperForm', 'batchAddPapersForm', 'organizePapersForm', 'searchPapersForm',
                'addSingleImageForm', 'batchAddImagesForm', 'searchImagesForm'
            ];
            allForms.forEach(formId => {
                const form = document.getElementById(formId);
                if (form) {
                    form.onsubmit = function() {
                        console.log("表单onsubmit：阻止默认提交");
                        return false;
                    };
                }
            });
        };
    </script>
</head>
<body>
    <div class="container">
        <h1>本地 AI 智能文献与图像管理助手</h1>
        
        <div class="tab-container">
            <div class="tab-buttons">
                <button class="tab-btn active" onclick="switchTab('paperTab')">论文管理</button>
                <button class="tab-btn" onclick="switchTab('imageTab')">图像管理</button>
            </div>

            <!-- 论文管理标签页 -->
            <div id="paperTab" class="tab-content active">
                <h2>论文管理模块</h2>

                <!-- 功能1：添加单篇论文 -->
                <div class="function-card">
                    <div class="form-area">
                        <h3>1. 添加单篇论文并分类</h3>
                        <form id="addSinglePaperForm" onsubmit="return false;">
                            <div class="form-group">
                                <label for="paper_path">论文本地路径（必填）：</label>
                                <input type="text" id="paper_path" name="paper_path" placeholder="/data/papers/xxx.pdf" required>
                            </div>
                            <div class="form-group">
                                <label for="paper_topics">分类主题（可选）：</label>
                                <input type="text" id="paper_topics" name="paper_topics" placeholder="CV,NLP,RL（逗号分隔）">
                            </div>
                            <button type="button" onclick="return submitAddSinglePaper();">添加并分类</button>
                        </form>
                    </div>
                    <!-- 结果区域唯一ID，用于局部更新 -->
                    <div id="addSinglePaperResult" class="result-area">
                        <h3>操作结果</h3>
                        <pre><span class="empty-result">未执行操作，结果将显示在这里...</span></pre>
                    </div>
                </div>

                <!-- 功能2：批量添加论文 -->
                <div class="function-card">
                    <div class="form-area">
                        <h3>2. 批量添加目录中的论文</h3>
                        <form id="batchAddPapersForm" onsubmit="return false;">
                            <div class="form-group">
                                <label for="paper_dir">论文目录路径（必填）：</label>
                                <input type="text" id="paper_dir" name="paper_dir" placeholder="/data/papers" required>
                            </div>
                            <div class="form-group">
                                <label for="batch_paper_topics">分类主题（可选）：</label>
                                <input type="text" id="batch_paper_topics" name="batch_paper_topics" placeholder="CV,NLP,RL（逗号分隔）">
                            </div>
                            <button type="button" onclick="return submitBatchAddPapers();">批量添加并分类</button>
                        </form>
                    </div>
                    <div id="batchAddPapersResult" class="result-area">
                        <h3>操作结果</h3>
                        <pre><span class="empty-result">未执行操作，中间过程和结果将显示在这里...</span></pre>
                    </div>
                </div>

                <!-- 功能3：一键整理论文 -->
                <div class="function-card">
                    <div class="form-area">
                        <h3>3. 一键整理论文（重新分类+清理重复）</h3>
                        <form id="organizePapersForm" onsubmit="return false;">
                            <div class="form-group">
                                <label for="organize_paper_dir">论文根目录（可选）：</label>
                                <input type="text" id="organize_paper_dir" name="organize_paper_dir" placeholder="/data/papers（默认data/papers）">
                            </div>
                            <div class="form-group">
                                <label for="organize_paper_topics">分类主题（可选）：</label>
                                <input type="text" id="organize_paper_topics" name="organize_paper_topics" placeholder="CV,NLP,RL（逗号分隔）">
                            </div>
                            <button type="button" onclick="return submitOrganizePapers();">一键整理</button>
                        </form>
                    </div>
                    <div id="organizePapersResult" class="result-area">
                        <h3>操作结果</h3>
                        <pre><span class="empty-result">未执行操作，中间过程和结果将显示在这里...</span></pre>
                    </div>
                </div>

                <!-- 功能4：语义搜索论文 -->
                <div class="function-card">
                    <div class="form-area">
                        <h3>4. 语义搜索论文（支持精细化检索）</h3>
                        <form id="searchPapersForm" onsubmit="return false;">
                            <div class="form-group">
                                <label for="paper_query">搜索查询词（必填）：</label>
                                <input type="text" id="paper_query" name="paper_query" placeholder="深度学习 图像分类" required>
                            </div>
                            <div class="form-group">
                                <label for="paper_limit">返回结果数量（默认3）：</label>
                                <input type="number" id="paper_limit" name="paper_limit" value="3" min="1" max="50">
                            </div>
                            <div class="form-group">
                                <input type="checkbox" id="paper_index" name="paper_index">
                                <label for="paper_index" class="checkbox-label">文件索引模式（仅返回文件名列表）</label>
                            </div>
                            <div class="form-group">
                                <input type="checkbox" id="paper_no_expand" name="paper_no_expand">
                                <label for="paper_no_expand" class="checkbox-label">直接使用原始查询</label>
                            </div>
                            <div class="form-group">
                                <input type="checkbox" id="paper_snippets" name="paper_snippets">
                                <label for="paper_snippets" class="checkbox-label">返回匹配片段和页码（精细化检索）</label>
                            </div>
                            <button type="button" onclick="return submitSearchPapers();">开始搜索</button>
                        </form>
                    </div>
                    <div id="searchPapersResult" class="result-area">
                        <h3>操作结果</h3>
                        <pre><span class="empty-result">未执行操作，结果将显示在这里...</span></pre>
                    </div>
                </div>
            </div>

            <!-- 图像管理标签页 -->
            <div id="imageTab" class="tab-content">
                <h2>图像管理模块</h2>

                <!-- 功能1：添加单张图像 -->
                <div class="function-card">
                    <div class="form-area">
                        <h3>1. 添加单张图像到数据库</h3>
                        <form id="addSingleImageForm" onsubmit="return false;">
                            <div class="form-group">
                                <label for="image_path">图像本地路径（必填）：</label>
                                <input type="text" id="image_path" name="image_path" placeholder="/data/images/xxx.jpg" required>
                            </div>
                            <button type="button" onclick="return submitAddSingleImage();">添加到数据库</button>
                        </form>
                    </div>
                    <div id="addSingleImageResult" class="result-area">
                        <h3>操作结果</h3>
                        <pre><span class="empty-result">未执行操作，结果将显示在这里...</span></pre>
                    </div>
                </div>

                <!-- 功能2：批量添加图像 -->
                <div class="function-card">
                    <div class="form-area">
                        <h3>2. 批量添加目录中的图像</h3>
                        <form id="batchAddImagesForm" onsubmit="return false;">
                            <div class="form-group">
                                <label for="image_dir">图像目录路径（必填）：</label>
                                <input type="text" id="image_dir" name="image_dir" placeholder="/data/images" required>
                            </div>
                            <button type="button" onclick="return submitBatchAddImages();">批量添加到数据库</button>
                        </form>
                    </div>
                    <div id="batchAddImagesResult" class="result-area">
                        <h3>操作结果</h3>
                        <pre><span class="empty-result">未执行操作，结果将显示在这里...</span></pre>
                    </div>
                </div>

                <!-- 功能3：文本搜索图像 -->
                <div class="function-card">
                    <div class="form-area">
                        <h3>3. 文本描述搜索图像</h3>
                        <form id="searchImagesForm" onsubmit="return false;">
                            <div class="form-group">
                                <label for="image_query">搜索描述词（必填）：</label>
                                <input type="text" id="image_query" name="image_query" placeholder="海边的日落、高山流水" required>
                            </div>
                            <div class="form-group">
                                <label for="image_limit">返回结果数量（默认3）：</label>
                                <input type="number" id="image_limit" name="image_limit" value="3" min="1" max="50">
                            </div>
                            <button type="button" onclick="return submitSearchImages();">开始搜索</button>
                        </form>
                    </div>
                    <div id="searchImagesResult" class="result-area">
                        <h3>操作结果</h3>
                        <pre><span class="empty-result">未执行操作，结果将显示在这里...</span></pre>
                    </div>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
"""

# ---------------------- Flask后端 ----------------------
@app.route('/')
def index():
    """首页：返回完整HTML模板（仅初始加载一次）"""
    return render_template_string(FULL_FUNCTION_TEMPLATE)

@app.route('/add_single_paper', methods=['POST'])
def add_single_paper():
    """添加单篇论文"""
    add_paper_result = ""
    if not document_processor_cls:
        add_paper_result = "❌ 论文模块导入失败：未找到DocumentProcessor或依赖缺失"
        return Response(add_paper_result, mimetype='text/plain')
    
    try:
        paper_path = request.form.get('paper_path', '').strip()
        paper_topics = request.form.get('paper_topics', '').strip()
        
        if not paper_path or not os.path.exists(paper_path):
            add_paper_result = "❌ 错误：论文路径不存在或无效"
            return Response(add_paper_result, mimetype='text/plain')
        
        topics = None
        if paper_topics:
            topics = [t.strip() for t in paper_topics.split(',') if t.strip()]
        
        processor = document_processor_cls()
        result = processor.process_single_paper(paper_path, topics)
        
        if result:
            add_paper_result = f"✅ 论文已成功处理并分类到 {result['topic']} 类别\n  文件路径: {result['path']}"
        else:
            add_paper_result = "❌ 论文处理失败，请检查日志获取详细信息"
    
    except Exception as e:
        logger.error(f"添加单篇论文失败: {str(e)}")
        add_paper_result = f"❌ 错误: 添加论文失败 - {str(e)}"
    
    return Response(add_paper_result, mimetype='text/plain')

@app.route('/batch_add_papers', methods=['POST'])
def batch_add_papers():
    """批量添加论文"""
    batch_paper_result = []
    if not document_processor_cls:
        batch_paper_result = ["❌ 论文模块导入失败：未找到DocumentProcessor或依赖缺失"]
        return Response("\n".join(batch_paper_result), mimetype='text/plain')
    
    try:
        paper_dir = request.form.get('paper_dir', '').strip()
        batch_paper_topics = request.form.get('batch_paper_topics', '').strip()
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        batch_paper_result.extend([
            f"📋 批量添加论文开始（{current_time}）",
            f"🔍 目标目录：{paper_dir}",
            "----------------------------------------",
            ""
        ])
        
        if not paper_dir or not os.path.exists(paper_dir):
            batch_paper_result.append("❌ 错误：论文目录不存在或无效")
            return Response("\n".join(batch_paper_result), mimetype='text/plain')
        if not os.path.isdir(paper_dir):
            batch_paper_result.append("❌ 错误：输入路径不是有效目录")
            return Response("\n".join(batch_paper_result), mimetype='text/plain')
        
        pdf_files = glob.glob(os.path.join(paper_dir, "**/*.pdf"), recursive=True)
        batch_paper_result.extend([
            f"ℹ️  扫描完成：共找到 {len(pdf_files)} 个PDF文件",
            "ℹ️  开始逐个处理论文（分类→入库）...",
            "----------------------------------------",
            ""
        ])
        
        if len(pdf_files) == 0:
            batch_paper_result.append("ℹ️  目录中未找到任何PDF文件，无需处理")
            return Response("\n".join(batch_paper_result), mimetype='text/plain')
        
        topics = None
        if batch_paper_topics:
            topics = [t.strip() for t in batch_paper_topics.split(',') if t.strip()]
            batch_paper_result.append(f"ℹ️  分类主题：{','.join(topics) if topics else '默认自动分类'}")
            batch_paper_result.append("")
        
        processor = document_processor_cls()
        success_count = 0
        fail_count = 0
        fail_records = []
        
        for idx, pdf_file in enumerate(pdf_files, 1):
            pdf_filename = os.path.basename(pdf_file)
            batch_paper_result.append(f"[{idx}/{len(pdf_files)}] 正在处理：{pdf_filename}")
            
            try:
                result = processor.process_single_paper(pdf_file, topics)
                if result:
                    batch_paper_result.append(f"   ✅ 处理成功：归属「{result['topic']}」类别")
                    success_count += 1
                else:
                    batch_paper_result.append(f"   ❌ 处理失败：未返回有效分类结果")
                    fail_count += 1
                    fail_records.append(pdf_filename)
            except Exception as e:
                error_msg = str(e)[:100]
                batch_paper_result.append(f"   ❌ 处理异常：{error_msg}...")
                fail_count += 1
                fail_records.append(pdf_filename)
            
            batch_paper_result.append("")
        
        batch_paper_result.extend([
            "----------------------------------------",
            "📊 批量添加论文处理完成",
            f"✅ 成功处理：{success_count} 篇",
            f"❌ 失败处理：{fail_count} 篇",
        ])
        
        if fail_records:
            batch_paper_result.append(f"📝 失败文件列表：{','.join(fail_records[:10])}{'...' if len(fail_records) > 10 else ''}")
        batch_paper_result.append(f"⏰ 处理结束时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    except Exception as e:
        logger.error(f"批量添加论文失败: {str(e)}")
        batch_paper_result.extend([
            f"❌ 批量处理异常：{str(e)}",
            f"⏰ 异常发生时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        ])
    
    return Response("\n".join(batch_paper_result), mimetype='text/plain')

@app.route('/organize_papers', methods=['POST'])
def organize_papers():
    """一键整理论文"""
    organize_paper_result = []
    if not document_processor_cls:
        organize_paper_result = ["❌ 论文模块导入失败：未找到DocumentProcessor或依赖缺失"]
        return Response("\n".join(organize_paper_result), mimetype='text/plain')
    
    try:
        organize_paper_dir = request.form.get('organize_paper_dir', '').strip() or None
        organize_paper_topics = request.form.get('organize_paper_topics', '').strip()
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        target_dir = organize_paper_dir or "data/papers（默认目录）"
        organize_paper_result.extend([
            f"📋 一键整理论文开始（{current_time}）",
            f"🔍 目标根目录：{target_dir}",
            "----------------------------------------",
            ""
        ])
        
        topics = None
        if organize_paper_topics:
            topics = [t.strip() for t in organize_paper_topics.split(',') if t.strip()]
        organize_paper_result.extend([
            f"ℹ️  分类主题：{','.join(topics) if topics else '默认自动分类'}",
            f"ℹ️  处理模式：重新分类 + 重复文件清理",
            "----------------------------------------",
            "ℹ️  开始扫描所有PDF文件（含子目录）...",
            ""
        ])
        
        processor = document_processor_cls()
        organize_paper_result.append("ℹ️  初始化整理引擎，验证文件有效性...")
        organize_paper_result.append("")
        
        result = processor.organize_papers(organize_paper_dir, topics)
        
        if result:
            total = result['total']
            success = result['success']
            failed = result['failed']
            cleaned = result.get('cleaned', 0)
            
            organize_paper_result.extend([
                f"ℹ️  扫描完成：共发现 {total} 篇论文",
                "----------------------------------------",
                "ℹ️  论文分类与移动过程：",
                ""
            ])
            
            organize_paper_result.extend([
                f"   1. 已验证 {total} 篇PDF文件的完整性",
                f"   2. 已重新分类 {success} 篇论文，匹配到对应类别目录",
                f"   3. 已将 {success} 篇论文移动到正确的类别文件夹",
                f"   4. 共 {failed} 篇论文因分类失败/文件损坏未完成移动",
            ])
            
            if cleaned > 0:
                organize_paper_result.extend([
                    "",
                    "ℹ️  重复文件清理过程：",
                    f"   已保留原始文件，删除重复副本/缓存文件",
                ])
            
            organize_paper_result.extend([
                "",
                "----------------------------------------",
                "📊 一键整理论文处理完成",
                f"✅ 成功整理：{success}/{total} 篇",
                f"❌ 整理失败：{failed} 篇",
                f"⏰ 处理结束时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            ])
            
            if failed > 0:
                organize_paper_result.append(f"📝 提示：失败论文可查看日志获取详细原因")
        else:
            organize_paper_result.append("❌ 一键整理失败：未返回有效处理结果，请检查日志")
    
    except Exception as e:
        logger.error(f"一键整理论文失败: {str(e)}")
        organize_paper_result.extend([
            f"❌ 一键整理异常：{str(e)}",
            f"⏰ 异常发生时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        ])
    
    return Response("\n".join(organize_paper_result), mimetype='text/plain')

@app.route('/search_papers', methods=['POST'])
def search_papers():
    """语义搜索论文"""
    search_paper_result = ""
    if not document_processor_cls:
        search_paper_result = "❌ 论文模块导入失败：未找到DocumentProcessor或依赖缺失"
        return Response(search_paper_result, mimetype='text/plain')
    
    try:
        paper_query = request.form.get('paper_query', '').strip()
        paper_limit = int(request.form.get('paper_limit', 10))
        paper_index = 'paper_index' in request.form
        paper_no_expand = 'paper_no_expand' in request.form
        paper_snippets = 'paper_snippets' in request.form
        
        if not paper_query:
            search_paper_result = "❌ 错误：请输入有效的搜索查询词"
            return Response(search_paper_result, mimetype='text/plain')
        if paper_limit < 1 or paper_limit > 50:
            search_paper_result = "❌ 错误：返回数量请限制在1-50之间"
            return Response(search_paper_result, mimetype='text/plain')
        
        processor = document_processor_cls()
        results = processor.search_papers(
            query=paper_query,
            limit=paper_limit,
            use_query_expansion=not paper_no_expand,
            snippets=paper_snippets
        )
        
        if not results:
            search_paper_result = "🔍 未找到相关论文"
        else:
            if paper_index:
                result_lines = [f"🔍 找到 {len(results)} 篇相关论文:"]
                for i, result in enumerate(results, 1):
                    result_lines.append(f"{i}. {result['filename']}")
                search_paper_result = "\n".join(result_lines)
            else:
                result_lines = [f"🔍 找到 {len(results)} 篇相关论文:", "-" * 80]
                for i, result in enumerate(results, 1):
                    result_lines.extend([
                        f"{i}. 文件名：{result['filename']}",
                        f"   路径：{result['path']}",
                        f"   类别：{result['topic']}",
                        f"   相似度：{result['similarity']:.4f}（{result['llm_score']:.1f}分）"
                    ])
                    if paper_snippets and result.get('snippets') and len(result['snippets']) > 0:
                        result_lines.append(f"   匹配片段（共{len(result['snippets'])}条）：")
                        for snippet_idx, snippet in enumerate(result['snippets'], 1):
                            result_lines.extend([
                                f"     [{snippet_idx}] 页码：{snippet['page']} | 片段相似度：{snippet['similarity']:.4f}",
                                f"        内容：{snippet['text'][:500]}{'...' if len(snippet['text']) > 500 else ''}"
                            ])
                    elif paper_snippets:
                        result_lines.append(f"   匹配片段：无有效匹配片段")
                    result_lines.append("-" * 80)
                search_paper_result = "\n".join(result_lines)
    
    except Exception as e:
        logger.error(f"搜索论文失败: {str(e)}")
        search_paper_result = f"❌ 错误: 搜索论文失败 - {str(e)}"
    
    return Response(search_paper_result, mimetype='text/plain')

# ---------------------- 图像模块 ----------------------
@app.route('/add_single_image', methods=['POST'])
def add_single_image():
    add_image_result = ""
    try:
        image_path = request.form.get('image_path', '').strip()
        if not image_path or not os.path.exists(image_path):
            add_image_result = "❌ 错误：图像路径不存在或无效"
            return Response(add_image_result, mimetype='text/plain')
        
        result = image_processor.add_image(image_path)
        if result:
            add_image_result = f"✅ 图像已成功添加到数据库\n 文件路径: {result['path']}"
        else:
            add_image_result = "❌ 错误: 图像添加失败"
    
    except Exception as e:
        logger.error(f"添加单张图像失败: {str(e)}")
        add_image_result = f"❌ 错误: 添加图像失败 - {str(e)}"
    
    return Response(add_image_result, mimetype='text/plain')

@app.route('/batch_add_images', methods=['POST'])
def batch_add_images():
    batch_image_result = ""
    try:
        image_dir = request.form.get('image_dir', '').strip()
        if not image_dir or not os.path.exists(image_dir) or not os.path.isdir(image_dir):
            batch_image_result = "❌ 错误：图像目录不存在或不是有效目录"
            return Response(batch_image_result, mimetype='text/plain')
        
        results = image_processor.batch_add_images(image_dir)
        batch_image_result = f"✅ 批量处理完成，共成功添加 {len(results)} 张图像"
    
    except Exception as e:
        logger.error(f"批量添加图像失败: {str(e)}")
        batch_image_result = f"❌ 错误: 批量添加图像失败 - {str(e)}"
    
    return Response(batch_image_result, mimetype='text/plain')

@app.route('/search_images', methods=['POST'])
def search_images():
    search_image_result = ""
    try:
        image_query = request.form.get('image_query', '').strip()
        image_limit = int(request.form.get('image_limit', 10))
        
        if not image_query:
            search_image_result = "❌ 错误：请输入有效的搜索描述词"
            return Response(search_image_result, mimetype='text/plain')
        if image_limit < 1 or image_limit > 50:
            search_image_result = "❌ 错误：返回数量请限制在1-50之间"
            return Response(search_image_result, mimetype='text/plain')
        
        search_results = image_processor.search_images(image_query, image_limit)
        if not search_results:
            search_image_result = "🔍 未找到相关图像"
        else:
            result_lines = [
                f"✅ 找到 {len(search_results)} 张相关图像：",
                "-" * 60
            ]
            for idx, result in enumerate(search_results, start=1):
                filename = result.get('filename', '未知文件名')
                path = result.get('path', '未知路径')
                similarity = result.get('similarity', 0.0)
                result_lines.extend([
                    f"{idx}. {filename}",
                    f"   路径: {path}",
                    f"   相似度: {similarity:.4f}",
                    ""
                ])
            result_lines.append("-" * 60)
            search_image_result = "\n".join(result_lines)
    
    except Exception as e:
        logger.error(f"搜索图像失败: {str(e)}")
        search_image_result = f"❌ 错误: 搜索图像失败 - {str(e)}"
    
    return Response(search_image_result, mimetype='text/plain')

# ---------------------- 程序入口：启动Web服务 ----------------------
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=False)