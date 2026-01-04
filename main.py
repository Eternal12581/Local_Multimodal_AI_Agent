"""主程序入口，处理命令行参数并调用相应功能"""
import click
from src.image_processor import ImageProcessor
from src.utils import logger
from datetime import datetime

@click.group()
def cli():
    """本地多模态AI文献与图像管理助手"""
    pass

@cli.command(name="add_paper") 
@click.argument('path', type=click.Path(exists=True))
@click.option('--topics', default=None, help='分类主题，用逗号分隔，例如 "CV,NLP,RL"')
def add_paper(path, topics):
    """添加并分类单篇论文"""
    try:
        from src.document_processor import DocumentProcessor
        processor = DocumentProcessor()
       
        if topics:
            topics = [t.strip() for t in topics.split(',') if t.strip()]
        
        result = processor.process_single_paper(path, topics)
        if result:
            click.echo(f"✓ 论文已成功处理并分类到 {result['topic']} 类别")
            click.echo(f"  文件路径: {result['path']}")
        else:
            click.echo("✗ 论文处理失败，请检查日志获取详细信息", err=True)
    except Exception as e:
        logger.error(f"添加论文失败: {str(e)}")
        click.echo(f"✗ 错误: 添加论文失败 - {str(e)}", err=True)

@cli.command(name="batch_add_papers")
@click.argument('directory', type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option('--topics', default=None, help='分类主题，用逗号分隔，例如 "CV,NLP,RL"')
def batch_add_papers(directory, topics):
    """批量添加并分类目录中的所有论文"""
    try:
        from src.document_processor import DocumentProcessor
        processor = DocumentProcessor()
        if topics:
            topics = [t.strip() for t in topics.split(',') if t.strip()]
        
        results = processor.batch_process_papers(directory, topics)
        if results:
            click.echo(f"✓ 批量处理完成，共成功处理 {len(results)} 篇论文")
        else:
            click.echo("✗ 批量处理完成，但没有成功处理任何论文", err=True)
    except Exception as e:
        logger.error(f"批量添加论文失败: {str(e)}")
        click.echo(f"✗ 错误: 批量添加论文失败 - {str(e)}", err=True)

@cli.command(name="organize_papers")
@click.option('--papers-dir', default=None, type=click.Path(exists=True, file_okay=False, dir_okay=True), help='论文根目录，默认为 data/papers')
@click.option('--topics', default=None, help='分类主题，用逗号分隔，例如 "CV,NLP,RL"')
def organize_papers(papers_dir, topics):
    """一键整理论文：扫描所有PDF，重新分类并移动到正确文件夹"""
    try:
        from src.document_processor import DocumentProcessor
        processor = DocumentProcessor()
        if topics:
            topics = [t.strip() for t in topics.split(',') if t.strip()]
        
        result = processor.organize_papers(papers_dir, topics)
        if result:
            click.echo(f"一键整理完成，共成功整理 {result['success']}/{result['total']} 篇论文")
            if result['failed'] > 0:
                click.echo(f"失败: {result['failed']} 篇论文")
            if result.get('cleaned', 0) > 0:
                click.echo(f"已清理 {result['cleaned']} 条重复记录")
    except Exception as e:
        logger.error(f"一键整理失败: {str(e)}")
        click.echo(f"✗ 错误: 一键整理失败 - {str(e)}", err=True)

@cli.command(name="search_paper")
@click.argument('query')
@click.option('--limit', default=3, type=int, help='返回结果数量，默认3')
@click.option('--index', is_flag=True, help='文件索引模式：仅返回文件列表')
@click.option('--no-expand', is_flag=True, help='不使用查询扩写，直接使用原始查询')
@click.option('--snippets', is_flag=True, help='返回匹配的段落和页码（精细化检索）')
def search_paper(query, limit, index, no_expand, snippets):  
    """通过语义搜索论文"""
    try:
        from src.document_processor import DocumentProcessor
        processor = DocumentProcessor()
        
        results = processor.search_papers(
            query=query,
            limit=limit,
            use_query_expansion=not no_expand,
            snippets=snippets
        )
        
        if not results:
            click.echo("未找到相关论文")
            return
        
        # 文件索引模式：仅返回文件列表，不显示片段
        if index:
            click.echo(f"找到 {len(results)} 篇相关论文:")
            for i, result in enumerate(results, 1):
                click.echo(f"{i}. {result['filename']}")
            return
        
        # 默认模式：详细输出
        click.echo(f"找到 {len(results)} 篇相关论文:")
        click.echo("-" * 80)
        for i, result in enumerate(results, 1):
            # 基础信息输出
            click.echo(f"{i}. 文件名：{result['filename']}")
            click.echo(f"   路径：{result['path']}")
            click.echo(f"   类别：{result['topic']}")
            click.echo(f"   相似度：{result['similarity']:.4f}（{result['llm_score']:.1f}分）")
            if result.get('llm_reason'):
                click.echo(f"   备注：{result['llm_reason'][:100]}...")
            
            # 格式化输出匹配片段
            if snippets and result.get('snippets') and len(result['snippets']) > 0:
                click.echo(f"   匹配片段（共{len(result['snippets'])}条）：")
                for snippet_idx, snippet in enumerate(result['snippets'], 1):
                    click.echo(f"     [{snippet_idx}] 页码：{snippet['page']} | 片段相似度：{snippet['similarity']:.4f}")
                    snippet_text = snippet['text'][:500] if len(snippet['text']) > 500 else snippet['text']
                    click.echo(f"        内容：{snippet_text}{'...' if len(snippet['text']) > 500 else ''}")
            elif snippets:
                click.echo(f"   匹配片段：无有效匹配片段")
            
            click.echo("-" * 80)
    except Exception as e:
        logger.error(f"搜索论文失败: {str(e)}")
        click.echo(f"✗ 错误: 搜索论文失败 - {str(e)}", err=True)

@cli.command(name="add_image")
@click.argument('path', type=click.Path(exists=True))
def add_image(path):
    """添加单张图像到数据库"""
    try:
        processor = ImageProcessor()
        result = processor.add_image(path)
        if result:
            click.echo(f"图像已成功添加到数据库")
            click.echo(f"文件路径: {result['path']}")
    except Exception as e:
        logger.error(f"添加图像失败: {str(e)}")
        click.echo(f"错误: 添加图像失败 - {str(e)}", err=True)

@cli.command(name="batch_add_images")
@click.argument('directory', type=click.Path(exists=True, file_okay=False, dir_okay=True))
def batch_add_images(directory):
    """批量添加目录中的所有图像"""
    try:
        processor = ImageProcessor()
        results = processor.batch_add_images(directory)
        click.echo(f"批量处理完成，共成功添加 {len(results)} 张图像")
    except Exception as e:
        logger.error(f"批量添加图像失败: {str(e)}")
        click.echo(f"错误: 批量添加图像失败 - {str(e)}", err=True)

@cli.command(name="search_image")
@click.argument('query')
@click.option('--limit', default=3, type=int, help='返回结果数量，默认3')
def search_image(query, limit):
    """通过文本描述搜索图像（格式化输出序号、路径、相似度）"""
    try:
        processor = ImageProcessor()
        # 1. 获取搜索结果
        search_results = processor.search_images(query, limit)
        
        if not search_results:
            click.echo("未找到相关图像（相似度低于阈值）")
            return
        
        # 2. 格式化输出
        click.echo(f"找到 {len(search_results)} 张相关图像：")
        click.echo("-" * 60)
        for idx, result in enumerate(search_results, start=1):
            filename = result.get('filename', '未知文件名')
            path = result.get('path', '未知路径')
            similarity = result.get('similarity', 0.0)
            
            click.echo(f"{idx}. {filename}")
            click.echo(f"   路径: {path}")
            click.echo(f"   相似度: {similarity:.4f}")
            click.echo()  # 空行分隔，提升可读性
        click.echo("-" * 60)
    
    except Exception as e:
        logger.error(f"搜索图像失败: {str(e)}")
        click.echo(f"错误: 搜索图像失败 - {str(e)}", err=True)

if __name__ == '__main__':
    cli()