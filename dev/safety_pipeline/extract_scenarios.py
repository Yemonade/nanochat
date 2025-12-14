"""
从 more-harmful_dataset 中使用大模型抽取和分析敏感场景

使用 qwen3-8b 对数据集中的有害内容进行智能分析和分类，
提取关键的安全场景用于后续的 safety 数据生成。
"""

import os
import sys
import json
import requests
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from nanochat.common import get_base_dir


# API 配置
api_key = open("openroutertoken.txt", "r", encoding="utf-8").read().strip()
url = "https://openrouter.ai/api/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}


# 大模型分析提示词
ANALYSIS_PROMPT = """你是一个安全内容分析专家。请从以下有害请求中提取敏感词。

请求内容：
{content}

请以 JSON 格式返回敏感词列表，格式如下：
{{
  "keywords": ["敏感词1", "敏感词2", "敏感词3", ...]
}}

要求：
- 提取 3-10 个最关键的敏感词
- 敏感词应该是具体的、有代表性的词语
- 包括动词、名词、关键短语
- 如果内容不适合用于训练，返回空列表

只返回 JSON，不要其他文字。"""


def load_harmful_dataset():
    """加载 more-harmful_dataset"""
    print("正在加载 more-harmful_dataset...")
    try:
        ds = load_dataset("harryn0502/more-harmful_dataset", split="train")
        print(f"✓ 成功加载数据集，共 {len(ds)} 条记录")
        return ds
    except Exception as e:
        print(f"✗ 加载失败: {e}")
        return None


def analyze_with_llm(content, idx):
    """
    使用大模型提取敏感词
    
    参数:
        content: 内容文本
        idx: 索引（用于错误追踪）
    
    返回:
        result: {"original_content": content, "keywords": [...]}
    """
    try:
        # 构建请求
        prompt = ANALYSIS_PROMPT.format(content=content)
        payload = {
            "model": "qwen/qwen-2-7b-instruct",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,  # 较低温度以获得更一致的输出
            "response_format": {"type": "json_object"}  # 请求 JSON 格式输出
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        result = response.json()
        
        # 解析响应
        content_str = result['choices'][0]['message']['content']
        analysis = json.loads(content_str)
        
        # 提取关键词列表
        keywords = analysis.get('keywords', [])
        
        # 如果关键词为空，跳过这条
        if not keywords:
            return None
        
        return {
            'original_content': content,
            'keywords': keywords
        }
        
    except Exception as e:
        print(f"  ✗ 分析失败 (索引 {idx}): {e}")
        return None


def extract_and_analyze_scenarios(dataset, max_samples=None, num_workers=4):
    """
    使用大模型批量分析数据集
    
    参数:
        dataset: 数据集
        max_samples: 最多分析多少条（None = 全部）
        num_workers: 并行线程数
    
    返回:
        analyzed_scenarios: 分析结果列表
    """
    print(f"\n正在使用大模型分析数据集 (并行线程: {num_workers})...")
    
    # 准备待分析的内容
    contents = []
    for idx, item in enumerate(dataset):
        if max_samples and idx >= max_samples:
            break
        
        # 提取内容
        content = None
        for field in ['prompt', 'instruction', 'question', 'text', 'input']:
            if field in item and item[field]:
                content = item[field].strip()
                break
        
        if content:
            contents.append((idx, content))
    
    print(f"  准备分析 {len(contents)} 条内容")
    
    # 并行分析
    analyzed_scenarios = []
    completed = 0
    failed = 0
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # 提交所有任务
        futures = {
            executor.submit(analyze_with_llm, content, idx): (idx, content)
            for idx, content in contents
        }
        
        # 处理结果
        for future in as_completed(futures):
            try:
                analysis = future.result()
                
                if analysis:
                    analyzed_scenarios.append(analysis)
                    completed += 1
                else:
                    failed += 1
                
                # 显示进度
                total_processed = completed + failed
                if total_processed % 10 == 0:
                    print(f"  进度: {total_processed}/{len(contents)} "
                          f"(有效: {completed}, 无效: {failed})")
                    
            except Exception as e:
                failed += 1
                print(f"  ✗ 处理失败: {e}")
    
    print(f"\n✓ 分析完成: {completed} 条有效, {failed} 条无效/失败")
    return analyzed_scenarios


def extract_all_keywords(analyzed_scenarios):
    """提取所有唯一的敏感词"""
    all_keywords = set()
    keyword_to_contents = defaultdict(list)
    
    for scenario in analyzed_scenarios:
        content = scenario['original_content']
        keywords = scenario.get('keywords', [])
        
        for keyword in keywords:
            all_keywords.add(keyword)
            keyword_to_contents[keyword].append(content)
    
    return all_keywords, keyword_to_contents


def analyze_results(analyzed_scenarios, all_keywords, keyword_to_contents):
    """分析和展示结果统计"""
    print("\n" + "=" * 80)
    print("敏感词提取结果统计")
    print("=" * 80)
    
    print(f"分析的场景数: {len(analyzed_scenarios)}")
    print(f"提取的唯一敏感词数: {len(all_keywords)}")
    
    # 统计关键词频率
    from collections import Counter
    keyword_freq = Counter()
    for scenario in analyzed_scenarios:
        for keyword in scenario.get('keywords', []):
            keyword_freq[keyword] += 1
    
    # 显示最常见的敏感词
    print("\n最常见的敏感词 (Top 20):")
    for keyword, count in keyword_freq.most_common(20):
        print(f"  {keyword}: {count} 次")
    
    # 显示一些示例
    print("\n" + "=" * 80)
    print("场景和敏感词示例 (显示前 5 条)")
    print("=" * 80)
    
    for i, scenario in enumerate(analyzed_scenarios[:5]):
        content = scenario['original_content']
        preview = content[:100] + "..." if len(content) > 100 else content
        keywords = ", ".join(scenario.get('keywords', []))
        
        print(f"\n{i+1}. {preview}")
        print(f"   敏感词: {keywords}")


def save_results(analyzed_scenarios, all_keywords, keyword_to_contents, output_dir):
    """保存分析结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 保存敏感词列表
    keywords_output = os.path.join(output_dir, "sensitive_keywords.txt")
    print(f"\n正在保存敏感词列表到 {keywords_output}...")
    
    with open(keywords_output, 'w', encoding='utf-8') as f:
        for keyword in sorted(all_keywords):
            f.write(keyword + '\n')
    
    print(f"✓ 已保存 {len(all_keywords)} 个唯一敏感词")
    
    # 2. 保存详细的敏感词到内容映射
    detailed_output = os.path.join(output_dir, "keywords_to_contents.json")
    print(f"\n正在保存敏感词到内容映射到 {detailed_output}...")
    
    # 统计每个关键词的频率
    keyword_stats = {}
    for keyword, contents in keyword_to_contents.items():
        keyword_stats[keyword] = {
            "count": len(contents),
            "examples": contents[:3]  # 只保存前3个示例
        }
    
    output_data = {
        "metadata": {
            "source": "harryn0502/more-harmful_dataset",
            "analyzer": "qwen-2-7b-instruct",
            "total_scenarios": len(analyzed_scenarios),
            "unique_keywords": len(all_keywords)
        },
        "keyword_stats": keyword_stats
    }
    
    with open(detailed_output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"✓ 已保存")
    
    # 3. 保存完整的场景和关键词
    scenarios_output = os.path.join(output_dir, "scenarios_with_keywords.json")
    print(f"\n正在保存完整场景数据到 {scenarios_output}...")
    
    with open(scenarios_output, 'w', encoding='utf-8') as f:
        json.dump(analyzed_scenarios, f, ensure_ascii=False, indent=2)
    
    print(f"✓ 已保存 {len(analyzed_scenarios)} 条场景")
    
    # 4. 生成提示列表（用于后续生成）
    prompts_output = os.path.join(output_dir, "harmful_prompts.txt")
    print(f"\n正在生成提示列表到 {prompts_output}...")
    
    with open(prompts_output, 'w', encoding='utf-8') as f:
        for scenario in analyzed_scenarios:
            content = scenario['original_content']
            # 清理换行符
            clean_content = content.replace('\n', ' ').replace('\r', ' ').strip()
            f.write(clean_content + '\n')
    
    print(f"✓ 已保存 {len(analyzed_scenarios)} 条提示")
    
    return keywords_output, detailed_output, scenarios_output, prompts_output


def main():
    """主函数"""
    print("\n")
    print("=" * 80)
    print("使用大模型从 more-harmful_dataset 提取和分析敏感场景")
    print("=" * 80)
    print()
    
    # 配置
    max_samples = 200  # 限制分析数量（可调整，None = 全部）
    num_workers = 4    # 并行线程数
    
    # 1. 加载数据集
    dataset = load_harmful_dataset()
    if dataset is None:
        print("\n✗ 无法加载数据集，退出")
        return
    
    # 2. 使用大模型分析和提取场景
    analyzed_scenarios = extract_and_analyze_scenarios(
        dataset, 
        max_samples=max_samples,
        num_workers=num_workers
    )
    
    if not analyzed_scenarios:
        print("\n✗ 没有提取到有效场景，退出")
        return
    
    # 3. 提取所有敏感词
    all_keywords, keyword_to_contents = extract_all_keywords(analyzed_scenarios)
    
    # 4. 分析结果
    analyze_results(analyzed_scenarios, all_keywords, keyword_to_contents)
    
    # 5. 保存结果
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    keywords_file, detailed_file, scenarios_file, prompts_file = save_results(
        analyzed_scenarios, all_keywords, keyword_to_contents, output_dir
    )
    
    # 总结
    print("\n" + "=" * 80)
    print("完成!")
    print("=" * 80)
    print(f"✓ 敏感词列表: {keywords_file}")
    print(f"✓ 敏感词详细映射: {detailed_file}")
    print(f"✓ 完整场景数据: {scenarios_file}")
    print(f"✓ 提示列表: {prompts_file}")
    print()
    print("下一步:")
    print(f"  1. 查看敏感词列表: head -30 {keywords_file}")
    print(f"  2. 查看敏感词统计: cat {detailed_file} | python -m json.tool | head -100")
    print(f"  3. 查看提示列表: head -20 {prompts_file}")
    print()


if __name__ == "__main__":
    main()

