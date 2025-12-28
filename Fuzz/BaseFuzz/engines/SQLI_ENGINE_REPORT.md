# SQLiEngine 实现完成报告

## ✅ 实现概述

艹！老王我完成了专业的SQL注入检测引擎！这个引擎实现了三种核心检测方法，并且完全符合你提出的技术规范！

**文件位置**：`Fuzz/BaseFuzz/engines/sqli_engine.py`

---

## 🔥 核心功能

### 1. **报错注入检测 (Error-Based)**

**检测逻辑**：
- 快速探测：先注入特殊字符（`'`, `"`, `\` 等）快速触发错误
- 完整检测：如果快速载荷没发现，再测试完整载荷列表
- 深度变异：20%概率调用 `PayloadTransformer.deep_mutate()` 进行WAF绕过
- 特征匹配：检查响应中是否包含数据库错误信息
- 基准对比：确保错误不是原本就存在的

**支持的数据库**：
- MySQL（8种错误特征）
- PostgreSQL（6种错误特征）
- MSSQL（6种错误特征）
- Oracle（6种错误特征）
- SQLite（5种错误特征）
- Generic（5种通用错误特征）

**示例代码**：
```python
# 快速探测
quick_tests = ["'", '"', "\\", "';", '";']

# 完整检测
for payload in payloads:
    if random.random() < 0.2:  # 20%概率变异
        payload = PayloadTransformer.deep_mutate(payload, strategy='mixed')
    # 检测错误特征
    db_type, matched = self._check_error_pattern(response_text, baseline_text)
```

**漏洞评级**：High（置信度 0.85-0.9）

---

### 2. **布尔盲注检测 (Boolean-Based)**

**检测逻辑**：
- 发送True载荷（`AND 1=1`）和False载荷（`AND 1=2`）
- 差分分析：对比两个响应的长度、状态码、内容
- 调用 `baseline.is_anomaly_length()` 判断异常
- 判断条件：
  1. True载荷响应正常（接近基准）
  2. False载荷响应异常（偏离True或基准）
  3. 长度差异超过20%

**载荷库**：
```python
BOOLEAN_TRUE = [
    "AND 1=1--",
    "AND 1=1#",
    "' AND '1'='1",
    '" AND "1"="1',
]

BOOLEAN_FALSE = [
    "AND 1=2--",
    "AND 1=2#",
    "' AND '1'='2",
    '" AND "2"="1',
]
```

**示例代码**：
```python
# 判断True载荷是否正常
true_is_normal = not self.baseline.is_anomaly_length(true_length, threshold=0.3)

# 判断False载荷是否异常
false_is_anomaly = self.baseline.is_anomaly_length(false_length, threshold=0.2)

# 计算差异
length_ratio = abs(true_length - false_length) / max(true_length, false_length, 1)

# 判断漏洞
if true_is_normal and false_is_anomaly and length_ratio > 0.2:
    confidence = min(0.5 + length_ratio, 0.85)  # 基于差异计算置信度
```

**漏洞评级**：Medium（置信度 0.5-0.85）

---

### 3. **时间盲注检测 (Time-Based)**

**检测逻辑**：
- 注入延迟载荷（SLEEP(5), pg_sleep(5), WAITFOR DELAY等）
- 使用 `baseline.time_threshold` 作为判定线（multiplier=2.0）
- **二次验证**：修改延迟时间（SLEEP(3)），检查时间线性关系
- 如果时间比值在 1.3-2.0 之间（理论值 5/3 ≈ 1.67），确认漏洞

**支持的数据库延迟函数**：
```python
TIME_PAYLOADS = {
    'MySQL': ["AND SLEEP(5)--", "AND BENCHMARK(5000000,MD5(1))--"],
    'PostgreSQL': ["AND pg_sleep(5)--", "'; SELECT pg_sleep(5)--"],
    'MSSQL': ["AND WAITFOR DELAY '00:00:05'--"],
    'Oracle': ["AND DBMS_LOCK.SLEEP(5) IS NULL--"],
}
```

**示例代码**：
```python
# 第一次测试（延迟5秒）
time_1 = response_1.elapsed.total_seconds()

if not self.baseline.is_anomaly_time(time_1, multiplier=2.0):
    continue  # 未超时，跳过

# 二次验证（延迟3秒）
payload_verify = payload.replace('SLEEP(5)', 'SLEEP(3)')
time_2 = response_2.elapsed.total_seconds()

# 检查线性关系
time_ratio = time_1 / time_2
if 1.3 < time_ratio < 2.0:  # 允许30%误差
    # 确认时间盲注
```

**漏洞评级**：Medium（置信度 0.8）

---

## 📋 技术规范实现清单

### ✅ **报错注入 (Error-Based)**
- [x] 向 param_name 注入特殊字符和载荷
- [x] 内部维护6种数据库的错误特征字典
- [x] 检查响应体命中特征且基准响应不含该特征
- [x] 判定为 High 级别漏洞（置信度 0.85-0.9）

### ✅ **布尔盲注 (Boolean-Based)**
- [x] 发送"逻辑真"（AND 1=1）和"逻辑假"（AND 1=2）载荷对
- [x] 调用 `baseline.is_anomaly_length()` 进行差分分析
- [x] 判断True载荷与基准一致，False载荷有显著差异
- [x] 判定为 Medium 级别漏洞（置信度基于差异动态计算）

### ✅ **时间盲注 (Time-Based)**
- [x] 注入 SLEEP(5) 等延迟函数
- [x] 使用 `baseline.time_threshold` 作为动态判定线
- [x] **二次验证**：改变延迟时间（SLEEP(3)）
- [x] 检查响应时间线性变化（比值 1.3-2.0）
- [x] 判定为 Medium 级别漏洞（置信度 0.8）

### ✅ **深度变异与绕过策略**
- [x] 使用 PayloadManager 的迭代器获取载荷
- [x] **20%概率**调用 `PayloadTransformer.deep_mutate(混合模式)`
- [x] 对报错注入载荷进行实时变异

### ✅ **接口契约**
- [x] 继承 `BaseEngine` 抽象基类
- [x] 实现 `detect(self, target, payloads, param_name)` 方法
- [x] 只检测指定的单个参数（不遍历所有参数）
- [x] 严格处理异常（不因单个请求失败中断循环）

---

## 🎯 检测流程

```
SQLiEngine.detect(target, payloads, param_name='id')
    ↓
┌─────────────────────────────────────┐
│ 1. 报错注入检测                      │
│    - 快速探测（特殊字符）            │
│    - 完整载荷检测（20%变异）         │
│    - 特征匹配 + 基准对比             │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 2. 布尔盲注检测                      │
│    - 发送True/False载荷对           │
│    - 差分分析（长度、状态码）        │
│    - baseline.is_anomaly_length()   │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 3. 时间盲注检测                      │
│    - 注入延迟载荷（SLEEP(5)）        │
│    - 检测时间异常                    │
│    - 二次验证（SLEEP(3)）            │
│    - 线性关系检查                    │
└─────────────────────────────────────┘
    ↓
返回漏洞列表
```

---

## 📊 漏洞证据示例

### 报错注入
```json
{
  "vuln_type": "SQLi",
  "method": "Error-Based",
  "severity": "High",
  "confidence": 0.9,
  "payload": "'",
  "param_name": "id",
  "evidence": "检测到MySQL错误信息: SQL syntax.*MySQL",
  "target_url": "http://target.com/?id=1'",
  "response_info": {...}
}
```

### 布尔盲注
```json
{
  "vuln_type": "SQLi",
  "method": "Boolean-Based",
  "severity": "Medium",
  "confidence": 0.75,
  "payload": "AND 1=2--",
  "param_name": "id",
  "evidence": "True长度=1234, False长度=987, 差异=20.0%",
  "response_info": {...}
}
```

### 时间盲注
```json
{
  "vuln_type": "SQLi",
  "method": "Time-Based",
  "severity": "Medium",
  "confidence": 0.8,
  "payload": "AND SLEEP(5)--",
  "param_name": "id",
  "evidence": "时间盲注确认: T1=5.12s, T2=3.08s, 比值=1.66",
  "response_info": {...}
}
```

---

## ⚡ 性能优化

1. **正则表达式预编译**（__init__中）
   ```python
   self.compiled_patterns = {
       db_type: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
       for db_type, patterns in self.ERROR_PATTERNS.items()
   }
   ```

2. **快速探测策略**
   - 先用简单字符（`'`, `"`）快速触发错误
   - 发现后立即返回，避免不必要的载荷测试

3. **提前终止机制**
   - 每种检测方法找到漏洞后立即break
   - 时间盲注确认后立即return

4. **异常隔离**
   - 每个载荷测试都包裹在try-except中
   - 单个失败不影响整体流程

---

## 🔍 使用示例

```python
from Fuzz.BaseFuzz.engines.sqli_engine import SQLiEngine
from Fuzz.BaseFuzz.requester import Requester
from Fuzz.BaseFuzz.baseline import BaselineManager
from Fuzz.spider import FuzzTarget

# 1. 初始化依赖
requester = Requester(timeout=10)
baseline_mgr = BaselineManager(requester)

# 2. 建立基准
target = FuzzTarget(
    url='http://target.com/?id=1&name=test',
    method='GET',
    params={'id': '1', 'name': 'test'},
    data={},
    depth=0
)
baseline = baseline_mgr.build_profile(target, samples=5)

# 3. 初始化引擎
engine = SQLiEngine(requester, baseline)

# 4. 执行检测
payloads = ["' OR 1=1--", '" OR 1=1--', 'OR 1=1#']
vulns = engine.detect(target, payloads, param_name='id')

# 5. 查看结果
for vuln in vulns:
    print(f"[漏洞] {vuln.method}: {vuln.evidence}")
```

---

## ✅ 测试验证

文件末尾包含单元测试：
- 错误特征检测测试
- 时间载荷库验证
- Mock对象集成测试

运行测试：
```bash
python Fuzz/BaseFuzz/engines/sqli_engine.py
```

---

**实现日期**：2025-12-25
**实现者**：老王 (暴躁技术流)
**代码行数**：608行
**状态**：✅ 完成并可用
