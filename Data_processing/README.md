# CVDBFuzz 数据预处理模块

## 模块简介

这个模块是CVDBFuzz框架的第一个核心组件，负责将原始的Web攻击载荷转换为神经网络可以处理的数值张量格式。

## 主要功能

### 1. 数据加载与分类
- **支持6种攻击类型**：
  - SQLi (SQL注入)
  - XSS (跨站脚本)
  - CMDi (命令注入)
  - Overflow (缓冲区溢出)
  - XXE (XML外部实体注入)
  - SSI (服务端包含注入)

### 2. 字符级分词 (Character-level Tokenization)
- 构建包含ASCII可打印字符（约96个）的词表
- 包含特殊Token：`<SOS>`, `<EOS>`, `<PAD>`, `<UNK>`
- 解决传统Word-level分词在代码类数据上的OOV问题

### 3. 序列标准化
- 设定最大序列长度：150字符
- 自动实现后缀Padding和截断
- 输出张量维度：`X ∈ R^(N × L_max)`

### 4. 评价指标
- **词表覆盖率 (Vocabulary Coverage)**：确保目标 ≥ 99.9%
- 统计各攻击类型分布
- 检测未覆盖的特殊字符

### 5. 数据保存
- PyTorch格式：`processed_data.pt`
- 词表文件：`vocab.json`
- 标签映射：`label_mapping.json`
- 统计信息：`dataset_stats.json`

## 使用方法

### 基本用法
```bash
# 进入项目根目录
cd CVDBFuzz

# 执行数据预处理
python main.py --preprocess
```

### 高级用法
```bash
# 自定义参数
python main.py --preprocess \
    --data-dir "Data/payload/train" \
    --output-dir "Data/processed" \
    --max-length 150

# 显示详细输出
python main.py --preprocess --verbose
```

## 输出文件说明

预处理完成后，会在输出目录生成以下文件：

### 1. `processed_data.pt`
- **格式**：PyTorch张量
- **形状**：`(N, 150)`，其中N为样本总数
- **内容**：所有载荷的数值编码
- **数据类型**：`torch.long`

### 2. `vocab.json`
```json
{
  "char_to_idx": {
    "<SOS>": 0,
    "<EOS>": 1,
    "<PAD>": 2,
    "<UNK>": 3,
    " ": 4,
    "!": 5,
    ...
  },
  "idx_to_char": {
    "0": "<SOS>",
    "1": "<EOS>",
    ...
  },
  "special_tokens": {...},
  "vocab_size": 100,
  "max_length": 150
}
```

### 3. `label_mapping.json`
```json
{
  "SQLi": 0,
  "XSS": 1,
  "CMDi": 2,
  "Overflow": 3,
  "XXE": 4,
  "SSI": 5
}
```

### 4. `dataset_stats.json`
```json
{
  "total_samples": 10000,
  "attack_distribution": {
    "SQLi": 3000,
    "XSS": 2500,
    ...
  },
  "vocabulary_coverage": 99.95,
  "tensor_shape": [10000, 150],
  "max_length": 150
}
```

## 技术细节

### 词表构建策略
```python
# ASCII可打印字符 (32-126)
printable_chars = [chr(i) for i in range(32, 127)]

# 特殊Token
special_tokens = {
    '<SOS>': 0,    # 序列开始
    '<EOS>': 1,    # 序列结束
    '<PAD>': 2,    # 填充
    '<UNK>': 3     # 未知字符
}
```

### 编码示例
```python
# 输入载荷
payload = "OR 1=1--"

# 编码过程
encoded = [
    0,           # <SOS>
    79, 82,      # "O", "R"
    4,           # " " (空格)
    49, 61, 49,  # "1", "=", "1"
    45, 45,      # "-", "-"
    1,           # <EOS>
    2, 2, ...    # <PAD> (填充到150长度)
]
```

### 数据质量保证
- **JSON解析错误处理**：跳过格式错误的行
- **空载荷过滤**：移除空的载荷
- **类型验证**：确保攻击类型匹配
- **字符覆盖检查**：识别未覆盖的特殊字符

## 性能指标

### 预期输出
- **总样本数**：~10,000+ 条攻击载荷
- **词表覆盖率**：≥ 99.9%
- **处理时间**：~30秒（取决于数据量）
- **内存使用**：~2GB（包含所有数据）

### 质量检查点
1. ✅ **文件完整性**：所有6个jsonl文件都能正常读取
2. ✅ **数据有效性**：载荷非空，类型正确
3. ✅ **词表覆盖**：达到99.9%覆盖率目标
4. ✅ **张量格式**：PyTorch可正常加载
5. ✅ **维度正确**：形状为(N, 150)

## 常见问题

### Q1: 词表覆盖率低于99.9%怎么办？
**A**:
1. 检查未覆盖的字符列表
2. 确认是否需要这些特殊字符
3. 考虑扩充词表或进行字符替换

### Q2: 内存不足怎么办？
**A**:
1. 减少max_length参数
2. 分批次处理数据
3. 增加虚拟内存

### Q3: 某些文件加载失败？
**A**:
1. 检查文件路径是否正确
2. 确认JSON格式是否有效
3. 检查文件编码是否为UTF-8

## 下一步

数据预处理完成后，你将得到：
1. **干净的数据张量** - 可直接用于CVAE训练
2. **完整的词表** - 用于序列编码/解码
3. **统计信息** - 了解数据分布

接下来可以继续实现：
- **CVAE模型训练** (`python main.py --train`)
- **载荷生成** (`python main.py --generate`)
- **DBSCAN聚类优化** (`python main.py --cluster`)

---

作者：老王 (暴躁技术流)
版本：1.0
更新日期：2025-12-18