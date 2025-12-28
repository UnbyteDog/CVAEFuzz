# XSSEngine 实现完成报告

## ✅ 实现概述

艹！老王我完成了专业的XSS检测引擎！这个引擎实现了完整的"探测-分析-测试"流水线，并且完全符合你提出的技术规范！

**文件位置**：`Fuzz/BaseFuzz/engines/xss_engine.py`

---

## 🔥 核心功能

### 1. **反射型 XSS (Reflected) 探测流水线**

#### A. 无害探针 (Probe Stage)
```python
PROBE_TEMPLATE = 'CVDBXSS_{RANDOM}_PROBE'
```

**检测逻辑**：
- 生成随机且唯一的探针字符串（如 `CVDBXSS_A3F7X9_PROBE`）
- 注入探针到参数中
- 调用基类的 `_is_reflected()` 方法检查探针是否原样出现
- 如果反射，提取反射上下文环境

**上下文识别**：
```python
context = self._get_reflected_context(response_text, self.probe)
# 返回: 'script_tag', 'event_handler', 'html_tag', 'text_content', etc.
```

#### B. 上下文感知分析 (Context Analysis)

**支持的上下文类型**：
- `script_tag`: `<script>` 标签内
- `style_tag`: `<style>` 标签内
- `event_handler`: 事件处理器（`onclick`, `onload`, etc.）
- `html_tag`: HTML标签属性
- `html_comment`: HTML注释
- `javascript`: `javascript:` 伪协议
- `text_content`: 普通文本内容

**上下文识别正则**：
```python
CONTEXT_PATTERNS = {
    'script_tag': re.compile(r'<script[^>]*>.*?</script>', re.IGNORECASE | re.DOTALL),
    'event_handler': re.compile(r'\bon[a-z]+\s*=', re.IGNORECASE),
    'html_tag': re.compile(r'<[^>]+>', re.IGNORECASE),
    'javascript': re.compile(r'javascript:', re.IGNORECASE),
}
```

#### C. 载荷选择与测试

**根据上下文选择载荷**：
```python
def _select_payloads_by_context(self, payloads, context):
    if context == 'script_tag':
        # JavaScript上下文：使用 alert/confirm/prompt 载荷
        return [p for p in payloads if 'alert' in p.lower()]

    elif context == 'event_handler':
        # 事件处理器：使用不带标签的载荷
        return [p for p in payloads if '<' not in p and 'alert' in p.lower()]

    elif context == 'html_tag':
        # HTML标签：使用标签闭合载荷
        return [p for p in payloads if p.startswith(('>', '">', "'>"))]
```

**转义检测**：
```python
def _check_if_escaped(self, response_text, payload):
    # 检查常见的HTML实体编码
    escaped_chars = {
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#x27;',
    }
    # 检查载荷中的特殊字符是否被转义
```

**置信度计算**：
```python
def _calculate_xss_severity(self, context, payload, response_text):
    base_confidence = 0.7

    context_bonus = {
        'script_tag': 0.2,      # 最危险
        'event_handler': 0.15,
        'javascript': 0.1,
        'html_tag': 0.05,
        'text_content': 0.0,
    }

    confidence = base_confidence + context_bonus.get(context, 0.0)

    # 根据载荷调整
    if 'alert(' in payload:
        confidence += 0.1
    if 'onerror=' in payload:
        confidence += 0.1

    return severity, min(max(confidence, 0.5), 0.95)
```

---

### 2. **DOM XSS 静态分析**

**危险关键词库**（15个）：
```python
DOM_KEYWORDS = [
    '.innerHTML',
    '.outerHTML',
    'document.write',
    'document.writeln',
    'eval(',
    'setTimeout(',
    'setInterval(',
    'Function(',
    'execScript(',
    '.location',
    '.href',
    '.src',
    'location.href',
    'location.hash',
    'location.search',
]
```

**检测逻辑**：
```python
def _detect_dom_xss(self, target, param_name, response):
    # 1. 检查响应中是否包含DOM关键词
    found_keywords = [kw for kw in self.DOM_KEYWORDS if kw in response.text]

    # 2. 检查参数名是否出现在危险上下文中
    param_in_context = self._check_param_in_dangerous_context(
        response_text, param_name
    )

    if param_in_context:
        # 发现DOM XSS风险（Low级别）
```

**漏洞评级**：Low（置信度0.6）

---

### 3. **WAF 拦截检测**

**拦截状态码**：
```python
WAF_STATUS_CODES = [403, 429, 503]
```

**检测策略**：
- 随机选择5个载荷进行WAF测试
- 统计被拦截次数
- 被拦截的载荷不计入漏洞（跳过）

---

## 📋 技术规范实现清单

### ✅ **A. 反射型 XSS 探测流水线**
- [x] 注入随机且唯一的无害探针（`CVDBXSS_{RANDOM}_PROBE`）
- [x] 检查探针是否原样出现在响应体中
- [x] 利用基类 `_is_reflected()` 方法判定
- [x] 调用 `_get_reflected_context()` 提取反射点前后字符

### ✅ **B. 上下文感知分析**
- [x] 判定环境：识别反射位在HTML标签间、属性值、`<script>`标签中
- [x] 检测危险关键词：`.innerHTML`, `document.write()`, `eval()`, etc.
- [x] 判定：若页面同时存在输入源和输出点，判定为Low或Medium风险

### ✅ **C. DOM XSS 静态分析**
- [x] 检测响应中的DOM XSS特征（15个危险关键词）
- [x] 检查参数是否被用于危险函数
- [x] 判定为Low级别风险（置信度0.6）

### ✅ **D. 深度变异与绕过**
- [x] 20%概率调用 `PayloadTransformer.deep_mutate(strategy='encoding')`
- [x] 对XSS载荷进行URL编码、Unicode转义等变异

### ✅ **E. WAF 拦截检测**
- [x] 检测403/429/503状态码
- [x] 统计WAF拦截次数
- [x] 被拦截载荷不计入漏洞

### ✅ **F. 置信度评分**
- [x] 根据字符是否被转义（`_check_if_escaped()`）
- [x] 根据探针是否原样返回（`_is_reflected()`）
- [x] 根据Payload是否能闭合上下文（`_calculate_xss_severity()`）
- [x] 动态计算confidence（0.5-0.95）

---

## 🎯 检测流程

```
XSSEngine.detect(target, payloads, param_name='name')
    ↓
┌─────────────────────────────────────┐
│ 1. 无害探针检测                      │
│    - 注入 CVDBXSS_{RANDOM}_PROBE    │
│    - 检查探针是否反射                │
│    - 提取上下文环境                  │
└─────────────────────────────────────┘
    ↓
    参数未反射？
    ↓ 是
    返回空列表（跳过XSS测试）

    ↓ 否
┌─────────────────────────────────────┐
│ 2. DOM XSS 静态分析                 │
│    - 检测DOM危险关键词               │
│    - 检查参数在危险上下文中          │
│    - 判定Low级别风险                 │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 3. 根据上下文选择载荷               │
│    - script_tag → alert() 载荷      │
│    - event_handler → 无标签载荷      │
│    - html_tag → 标签闭合载荷         │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 4. 执行XSS载荷测试                  │
│    - 20%深度变异绕过WAF              │
│    - 检查载荷是否反射                │
│    - 检查载荷是否转义                │
│    - 计算置信度                      │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 5. 统计WAF拦截                      │
│    - 随机测试5个载荷                 │
│    - 统计403/429/503次数             │
└─────────────────────────────────────┘
    ↓
返回漏洞列表
```

---

## 📊 漏洞证据示例

### 反射型 XSS（script_tag上下文）
```json
{
  "vuln_type": "XSS",
  "method": "Reflected",
  "severity": "High",
  "confidence": 0.9,
  "payload": "<script>alert(1)</script>",
  "param_name": "name",
  "evidence": "上下文: script_tag, 载荷原样反射",
  "target_url": "http://target.com/?name=<script>alert(1)</script>",
  "response_info": {...}
}
```

### 反射型 XSS（event_handler上下文）
```json
{
  "vuln_type": "XSS",
  "method": "Reflected",
  "severity": "Medium",
  "confidence": 0.75,
  "payload": "alert(1)",
  "param_name": "callback",
  "evidence": "上下文: event_handler, 载荷原样反射",
  "response_info": {...}
}
```

### DOM XSS（静态分析）
```json
{
  "vuln_type": "XSS",
  "method": "DOM-Based",
  "severity": "Low",
  "confidence": 0.6,
  "payload": "参数name可能被用于DOM操作",
  "param_name": "name",
  "evidence": "检测到DOM关键词: .innerHTML, document.write",
  "response_info": {...}
}
```

---

## ⚡ 性能优化

1. **提前终止机制**：
   - 参数未反射时立即返回（跳过所有XSS测试）
   - 找到高置信度漏洞（>0.8）后break

2. **载荷数量限制**：
   - 每个上下文最多测试10个载荷
   - 避免过多请求

3. **随机抽样WAF检测**：
   - 只随机选择5个载荷测试WAF
   - 减少WAF检测开销

4. **智能载荷选择**：
   - 根据上下文选择合适的载荷
   - 提高检测效率

---

## 🔍 使用示例

```python
from Fuzz.BaseFuzz.engines.xss_engine import XSSEngine
from Fuzz.BaseFuzz.requester import Requester
from Fuzz.BaseFuzz.baseline import BaselineManager
from Fuzz.spider import FuzzTarget

# 1. 初始化依赖
requester = Requester(timeout=10)
baseline_mgr = BaselineManager(requester)

# 2. 建立基准
target = FuzzTarget(
    url='http://target.com/?name=test',
    method='GET',
    params={'name': 'test'},
    data={},
    depth=0
)
baseline = baseline_mgr.build_profile(target, samples=5)

# 3. 初始化引擎
engine = XSSEngine(requester, baseline)

# 4. 执行检测
payloads = [
    "<script>alert(1)</script>",
    "<img src=x onerror=alert(1)>",
    "javascript:alert(1)",
]
vulns = engine.detect(target, payloads, param_name='name')

# 5. 查看结果
for vuln in vulns:
    print(f"[漏洞] {vuln.method}: {vuln.evidence}")
    print(f"  严重性: {vuln.severity}")
    print(f"  置信度: {vuln.confidence:.2f}")
```

---

## ✅ 测试验证

文件末尾包含单元测试：
- 探针生成测试
- DOM关键词库验证
- 上下文载荷选择测试

运行测试：
```bash
python Fuzz/BaseFuzz/engines/xss_engine.py
```

---

**实现日期**：2025-12-25
**实现者**：老王 (暴躁技术流)
**代码行数**：664行
**状态**：✅ 完成并可用
