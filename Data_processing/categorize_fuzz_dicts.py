#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CVDBFuzz字典分类转换器
将FuzzLists中的字典文件按攻击类型分类并转换为JSONL格式
"""

import json
import os
from pathlib import Path

class FuzzCategorizer:
    def __init__(self, fuzz_dir="./../Data/payload/FuzzLists", output_dir="test"):
        self.fuzz_dir = Path(fuzz_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # 文件分类映射（老王的精准分类）
        self.categories = {
            # SQL注入攻击
            "SQLi": {
                "files": [
                    "sqli-error-based.txt",
                    "sqli-time-based.txt",
                    "sqli-union-select.txt",
                    "sqli_escape_chars.txt",
                    "auth_bypass.txt"  # 认证绕过也经常用于SQL注入
                ],
                "description": "SQL注入攻击载荷"
            },

            # XSS攻击
            "XSS": {
                "files": [
                    "xss_payloads_quick.txt",
                    "xss-payload-list.txt",
                    "xss_escape_chars.txt",
                    "xss_grep.txt",
                    "xss_find_inject.txt",
                    "xss_funny_stored.txt",
                    "xss_remote_payloads-http.txt",
                    "xss_remote_payloads-https.txt",
                    "xss_swf_fuzz.txt"
                ],
                "description": "跨站脚本攻击载荷"
            },

            # 命令注入
            "CMDi": {
                "files": [
                    "command_exec.txt"
                ],
                "description": "命令注入攻击载荷"
            },

            # 路径遍历
            "PathTraversal": {
                "files": [
                    "traversal.txt",
                    "traversal-short.txt",
                    "lfi.txt"  # 本地文件包含
                ],
                "description": "路径遍历和文件包含攻击"
            },

            # 目录扫描
            "Directory": {
                "files": [
                    "dirbuster-dirs.txt",
                    "dirbuster-quick.txt",
                    "dirbuster-top1000.txt",
                    "dirbuster-cgi.txt"
                ],
                "description": "目录和文件枚举"
            },

            # 认证爆破
            "Auth": {
                "files": [
                    "passwords_long.txt",
                    "passwords_medium.txt",
                    "passwords_quick.txt",
                    "usernames.txt"
                ],
                "description": "认证爆破字典"
            },

            # 缓冲区溢出
            "Overflow": {
                "files": [
                    "overflow.txt",
                    "overflow-dos.txt"
                ],
                "description": "缓冲区溢出攻击"
            },

            # XML攻击
            "XML": {
                "files": [
                    "xml-attacks.txt",
                    "XXE-payload.txt"
                ],
                "description": "XML外部实体注入攻击"
            },

            # SSI注入
            "SSI": {
                "files": [
                    "ssi_quick.txt"
                ],
                "description": "服务器端包含注入"
            },

            # 综合Fuzz
            "Fuzz": {
                "files": [
                    "basic_fuzz.txt",
                    "quick_fuzz.txt",
                    "full_fuzz.txt",
                    "vulnerability_discovery.txt",
                    "toplist-sorted.txt",
                    "url_payloads.txt",
                    "bad_chars.txt",
                    "payload_injectx.txt",
                    "grep_injectx.txt"
                ],
                "description": "综合模糊测试载荷"
            }
        }

    def read_file_lines(self, file_path):
        """读取文件内容，过滤空行和注释"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = []
                for line in f:
                    line = line.strip()
                    # 跳过空行和注释行
                    if line and not line.startswith('#') and not line.startswith('//'):
                        lines.append(line)
                return lines
        except Exception as e:
            print(f"[ERROR] 读取文件失败 {file_path}: {e}")
            return []

    def categorize_payloads(self):
        """按分类处理所有字典文件"""
        results = {}

        for category, config in self.categories.items():
            print(f"\n[PROCESSING] 正在处理分类: {category} - {config['description']}")
            payloads = []

            for filename in config["files"]:
                file_path = self.fuzz_dir / filename
                if file_path.exists():
                    print(f"  [READING] 读取文件: {filename}")
                    lines = self.read_file_lines(file_path)
                    payloads.extend(lines)
                    print(f"    [SUCCESS] 获取了 {len(lines)} 个载荷")
                else:
                    print(f"  [WARNING] 文件不存在: {filename}")

            # 去重并过滤
            unique_payloads = list(set(payloads))
            unique_payloads = [p for p in unique_payloads if p and len(p.strip()) > 0]

            results[category] = {
                "payloads": unique_payloads,
                "count": len(unique_payloads),
                "description": config["description"]
            }

            print(f"  [STATS] {category} 分类总计: {len(unique_payloads)} 个唯一载荷")

        return results

    def save_jsonl_files(self, results):
        """保存为JSONL格式文件"""
        print(f"\n[SAVING] 开始保存JSONL文件到: {self.output_dir}")

        for category, data in results.items():
            output_file = self.output_dir / f"{category.lower()}.jsonl"

            with open(output_file, 'w', encoding='utf-8') as f:
                for payload in data["payloads"]:
                    json_line = {
                        "payload": payload,
                        "type": category
                    }
                    f.write(json.dumps(json_line, ensure_ascii=False) + '\n')

            print(f"  [SUCCESS] 保存完成: {output_file} ({data['count']} 行)")

    def generate_summary_report(self, results):
        """生成汇总报告"""
        total_payloads = sum(data["count"] for data in results.values())

        report = f"""
# CVDBFuzz字典分类汇总报告
# 自动生成 - {total_payloads} 个总载荷

## 分类统计
"""

        for category, data in results.items():
            report += f"- **{category}**: {data['count']} 个载荷 - {data['description']}\n"

        report += f"""
## 文件清单
"""

        for category, data in results.items():
            report += f"- `{category.lower()}.jsonl`: {data['count']} 行\n"

        # 保存报告
        report_file = self.output_dir / "categorization_report.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"\n[REPORT] 汇总报告已保存: {report_file}")
        print(report)

    def run(self):
        """执行分类转换"""
        print("[CVDBFuzz] 字典分类器启动！")
        print(f"[INFO] 输入目录: {self.fuzz_dir}")
        print(f"[INFO] 输出目录: {self.output_dir}")

        if not self.fuzz_dir.exists():
            print(f"[ERROR] 输入目录不存在: {self.fuzz_dir}")
            return

        # 执行分类
        results = self.categorize_payloads()

        # 保存文件
        self.save_jsonl_files(results)

        # 生成报告
        self.generate_summary_report(results)

        print(f"[SUCCESS] 分类完成！共生成 {len(results)} 个分类文件")

def main():
    """主函数"""
    categorizer = FuzzCategorizer()
    categorizer.run()

if __name__ == "__main__":
    main()