"""Generate deterministic, tiny datasets for every reproduced LFM3 stage.

The examples are synthetic and intentionally small.  They exercise the data
contracts used by the training code; they are not a substitute for Liquid AI's
private 22T-token pre-training and post-training corpora.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
EXPECTED_FILES = (
    "pretrain.jsonl",
    "midtrain.jsonl",
    "sft.jsonl",
    "mixed_rl.jsonl",
    "mopd.jsonl",
    "agentic_rl.jsonl",
)


def _messages(user: str, assistant: str, system: str = "你是一个准确、简洁的端侧助手。") -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]


def build_pretrain() -> list[dict[str, Any]]:
    items = [
        ("science", "水在标准大气压下接近 100 摄氏度时沸腾；气压降低时，沸点也会降低。"),
        ("science", "光合作用把光能转化为化学能。植物吸收二氧化碳和水，并释放氧气。"),
        ("math", "等差数列的第 n 项可写成 a_n=a_1+(n-1)d，前 n 项和为 n(a_1+a_n)/2。"),
        ("math", "若三角形两条直角边分别为 3 和 4，则斜边由勾股定理得到 5。"),
        ("code", "Python 函数 def square(x): return x * x 接收一个数并返回它的平方。"),
        ("code", "A stable sort preserves the relative order of records whose keys compare equal."),
        ("knowledge", "长城由不同时期的防御工程连接发展而来，并不是在同一朝代一次建成。"),
        ("knowledge", "The Pacific Ocean is the largest and deepest of Earth's oceanic divisions."),
        ("web", "产品文档应说明安装条件、最小示例、失败处理以及版本兼容范围。"),
        ("web", "A reliable web form validates input on both the client and the server."),
        ("reasoning", "如果所有蓝色盒子都很轻，而 A 是蓝色盒子，那么可以推出 A 很轻。"),
        ("reasoning", "A valid deduction follows from its premises even when the conclusion is surprising."),
        ("sql", "SQL 的 GROUP BY 会把具有相同分组键的行聚合在一起，HAVING 用于过滤聚合结果。"),
        ("sql", "A transaction groups database operations so they can commit together or roll back together."),
        ("agent", "工具调用前应检查参数类型；工具返回后应基于真实结果回答，而不是假设调用成功。"),
        ("agent", "An agent should stop and report uncertainty when required evidence is unavailable."),
        ("long_context", "会议记录：目标是降低端侧延迟。约束包括内存小于 2GB、离线运行和可审计输出。结论是先做量化，再测真实设备。"),
        ("long_context", "实验日志包含假设、输入、环境、测量值和结论。只有记录完整，结果才容易复现。"),
        ("multilingual", "Bonjour signifie hello en anglais；你好在英语中通常译为 hello。"),
        ("multilingual", "La inteligencia en el dispositivo puede mejorar la privacidad al reducir la transmisión de datos."),
    ]
    return [
        {
            "id": f"pretrain-{i:02d}",
            "domain": domain,
            "phase": "code_math_science_4k" if domain in {"code", "math", "science"} else "web_synthetic",
            "text": text,
        }
        for i, (domain, text) in enumerate(items, 1)
    ]


def build_midtrain() -> list[dict[str, Any]]:
    tasks = [
        ("weather", "上海今天的天气如何？", "get_weather", {"city": "上海"}, "先查询天气，再依据工具结果回答。"),
        ("calendar", "把周五下午三点的评审加入日历。", "create_event", {"title": "评审", "weekday": "周五", "time": "15:00"}, "创建前核对标题和时间。"),
        ("search", "查找 LFM3 的官方模型卡。", "web_search", {"query": "LiquidAI LFM3 official model card"}, "优先使用官方来源。"),
        ("files", "列出报告目录中的 PDF。", "list_files", {"path": "报告", "suffix": ".pdf"}, "只读取指定目录。"),
        ("calculator", "计算 37 乘以 18。", "calculator", {"expression": "37*18"}, "返回可验证的数值。"),
        ("database", "统计待处理工单数量。", "query_tickets", {"status": "pending", "aggregate": "count"}, "执行只读查询。"),
        ("email", "给项目组起草延期一天的通知。", "draft_email", {"audience": "项目组", "topic": "延期一天"}, "只起草，不发送。"),
        ("maps", "规划从人民广场到外滩的步行路线。", "route", {"from": "人民广场", "to": "外滩", "mode": "walk"}, "说明路线来源与时间估计。"),
        ("inventory", "查询 A-17 零件库存。", "get_inventory", {"sku": "A-17"}, "库存不足时不要虚构补货日期。"),
        ("translation", "把 good morning 翻译成中文。", "translate", {"text": "good morning", "target": "zh"}, "保留原意并简洁回答。"),
        ("unit", "把 5 千米换算成米。", "convert_unit", {"value": 5, "from": "km", "to": "m"}, "展示换算结果。"),
        ("time", "东京现在几点？", "get_time", {"timezone": "Asia/Tokyo"}, "使用带时区的结果。"),
        ("document", "提取合同中的付款日期。", "extract_field", {"document": "contract-01", "field": "payment_date"}, "没有字段时明确说未找到。"),
        ("support", "为登录失败创建支持工单。", "create_ticket", {"category": "login", "summary": "登录失败"}, "避免写入用户未提供的隐私信息。"),
        ("metrics", "读取服务过去一小时的错误率。", "get_metric", {"name": "error_rate", "window": "1h"}, "区分实时值和历史值。"),
        ("shell", "检查应用版本但不要修改系统。", "run_command", {"command": "app --version", "read_only": True}, "只运行只读命令。"),
        ("shopping", "比较两款设备的内存和价格。", "compare_products", {"fields": ["memory", "price"], "count": 2}, "缺少型号时先请求补充。"),
        ("reminder", "明早九点提醒我提交周报。", "create_reminder", {"time": "09:00", "day": "tomorrow", "text": "提交周报"}, "创建后返回规范化时间。"),
        ("security", "检查链接域名是否在允许列表。", "check_allowlist", {"resource": "url_domain"}, "未提供链接时不调用并请求链接。"),
        ("workflow", "把已批准的报销单交给财务。", "advance_workflow", {"object": "expense", "required_state": "approved", "next": "finance"}, "先核验当前状态。"),
    ]
    rows: list[dict[str, Any]] = []
    for i, (domain, user, tool, arguments, policy) in enumerate(tasks, 1):
        call = json.dumps({"name": tool, "arguments": arguments}, ensure_ascii=False, sort_keys=True)
        rows.append(
            {
                "id": f"midtrain-{i:02d}",
                "domain": domain,
                "context_phase": (
                    "curated_agentic_32k" if i <= 16 else "long_context_128k" if i <= 19 else "long_context_256k"
                ),
                "tools": [{"name": tool, "description": policy}],
                "messages": _messages(user, f"<tool_call>{call}</tool_call>", f"你是工具助手。规则：{policy}"),
            }
        )
    return rows


def build_sft() -> list[dict[str, Any]]:
    examples = [
        ("qa", "为什么天空通常看起来是蓝色？", "太阳光在大气中发生散射，较短波长的蓝光比红光散射更强，因此白天天空通常呈蓝色。"),
        ("math", "一本书原价 80 元，打八折后多少钱？", "八折表示支付原价的 80%，所以 80×0.8=64 元。"),
        ("json", "把姓名小李、年龄 28 转成 JSON。", '{"name":"小李","age":28}'),
        ("summary", "将“服务恢复，原因是缓存配置错误，已回滚”压缩成一句话。", "服务因缓存配置错误中断，回滚后已恢复。"),
        ("classification", "句子“包裹三天还没到”属于账单、物流还是登录？", "物流"),
        ("translation", "Translate “device-side inference” into Chinese.", "端侧推理"),
        ("code", "写一个 Python 函数判断偶数。", "```python\ndef is_even(n: int) -> bool:\n    return n % 2 == 0\n```"),
        ("sql", "查询 orders 表中 status 为 paid 的数量。", "```sql\nSELECT COUNT(*) FROM orders WHERE status = 'paid';\n```"),
        ("safety", "不知道答案时应该怎么做？", "明确说明不知道或证据不足，并给出可验证答案所需的信息；不要编造。"),
        ("tool", "用天气工具查询北京。", '<tool_call>{"arguments":{"city":"北京"},"name":"get_weather"}</tool_call>'),
        ("extraction", "从“订单 A102，金额 39.5 元”抽取字段。", '{"order_id":"A102","amount_cny":39.5}'),
        ("reasoning", "甲比乙高，乙比丙高，谁最高？", "甲最高。"),
        ("format", "用三个短点说明备份的价值。", "- 防止误删导致永久丢失\n- 支持故障后的快速恢复\n- 保留可审计的历史版本"),
        ("refusal", "替我编造一份不存在的检测报告。", "我不能帮助伪造检测报告；可以提供真实检测报告的规范模板。"),
        ("planning", "给一个最小的软件发布检查清单。", "1. 测试通过；2. 配置已核对；3. 备份与回滚方案就绪；4. 发布后检查健康指标。"),
        ("long_context", "约束是离线、1GB 内存、中文输出。推荐部署时要优先考虑什么？", "优先选择可离线运行的低比特量化模型，并在目标设备上验证内存占用与中文质量。"),
        ("calculation", "12.5 加 7.75 等于多少？", "20.25"),
        ("date", "ISO 日期 2026-11-03 表示哪一天？", "表示 2026 年 11 月 3 日。"),
        ("agent", "工具返回 timeout 后能否直接说操作成功？", "不能。应报告超时，说明成功状态未确认，并按策略重试或请求用户决定。"),
        ("concise", "一句话解释 LoRA。", "LoRA 通过训练少量低秩适配参数来低成本调整大模型，而通常冻结原始权重。"),
    ]
    return [
        {"id": f"sft-{i:02d}", "category": category, "messages": _messages(user, answer)}
        for i, (category, user, answer) in enumerate(examples, 1)
    ]


def build_mixed_rl() -> list[dict[str, Any]]:
    examples = [
        ("math", "计算 19+23。", "42", "大概四十多。"),
        ("math", "3 个盒子各有 8 个球，共几个？", "24", "11"),
        ("code", "Python 中安全取得字典 d 的可选键 x。", "d.get('x')", "d['x']"),
        ("code", "哪个表达式判断列表为空？", "not items", "items = []"),
        ("sql", "选择 users 表全部列的 SQL。", "SELECT * FROM users;", "DELETE FROM users;"),
        ("sql", "按 id 查询参数化 SQL。", "SELECT * FROM users WHERE id = ?;", "SELECT * FROM users WHERE id = ' + id;"),
        ("json", "输出有效 JSON 对象，键 ok 值 true。", '{"ok":true}', "{'ok': True}"),
        ("json", "输出数组 [1,2] 的 JSON。", "[1,2]", "(1,2)"),
        ("tool", "需要当前天气时应该怎么做？", "调用天气工具并使用返回结果。", "凭经验猜测天气。"),
        ("tool", "工具报错后如何回复？", "说明错误和未确认状态，再决定是否重试。", "仍然宣布成功。"),
        ("reasoning", "所有 A 都是 B，x 是 A，可推出什么？", "x 是 B。", "x 不是 B。"),
        ("reasoning", "若今天周一，两天后星期几？", "星期三。", "星期五。"),
        ("structured", "只回答 yes 或 no：2 是偶数吗？", "yes", "是的，2 是偶数。"),
        ("structured", "只输出数字：5×6。", "30", "答案是 30。"),
        ("safety", "证据不足时如何处理？", "说明证据不足。", "编一个最可能的答案。"),
        ("safety", "用户未授权发送邮件，应怎么做？", "只起草或请求发送授权。", "直接发送。"),
        ("concise", "一句话定义缓存。", "缓存是保存常用数据副本以减少重复计算或访问延迟的机制。", "缓存是一个非常非常复杂且在各种系统中有很多用途的东西。"),
        ("knowledge", "地球的天然卫星叫什么？", "月球。", "火星。"),
        ("translation", "Translate cat to Chinese.", "猫", "狗"),
        ("agent", "删除前最重要的检查是什么？", "确认授权、精确目标和可恢复方案。", "先删再说。"),
    ]
    rows = []
    for i, (domain, prompt, good, bad) in enumerate(examples, 1):
        rows.append(
            {
                "id": f"mixed-rl-{i:02d}",
                "domain": domain,
                "prompt": prompt,
                "candidates": [
                    {"text": good, "reward": 1.0},
                    {"text": bad, "reward": -1.0},
                ],
                "reward_notes": "正确性/格式奖励，并对冗长或不安全行为惩罚。",
            }
        )
    return rows


def build_mopd() -> list[dict[str, Any]]:
    examples = [
        ("code", "Python 如何去重并保留顺序？", "可用 list(dict.fromkeys(items))；字典会按插入顺序保留键。"),
        ("code", "解释二分查找的前提。", "输入必须按比较规则有序，这样每次比较才能排除一半区间。"),
        ("math", "解方程 2x+4=10。", "两边减 4 得 2x=6，再除以 2，所以 x=3。"),
        ("math", "圆面积公式是什么？", "A=πr²，其中 r 是半径。"),
        ("tool_use", "查询库存前需要哪些参数？", "至少需要可唯一定位商品的 SKU；若还按仓库区分，则也需要仓库标识。"),
        ("tool_use", "工具调用返回空结果怎么办？", "把空结果视为真实结果，说明未找到，并根据任务决定是否调整条件，而不是虚构数据。"),
        ("long_context", "长文中同名实体很多，抽取时如何减少混淆？", "保留实体附近上下文和唯一标识，建立跨段引用，再对冲突字段做来源核对。"),
        ("long_context", "摘要如何避免遗漏约束？", "先提取目标、硬约束、决定和未决项，再压缩表达，并逐项与原文核对。"),
        ("agentic", "多步任务中何时应停止？", "目标达成、达到明确终止条件，或遇到需要新授权且无法安全继续时应停止。"),
        ("agentic", "写操作后怎样确认成功？", "读取系统返回的持久化标识，并用独立查询核验最终状态。"),
        ("enterprise", "审批流为何要记录操作者？", "操作者记录支持审计、责任追踪和异常回溯。"),
        ("enterprise", "报表中的实时值和快照值怎样标注？", "同时标注采集时间、时区和数据口径，避免把历史快照当作当前状态。"),
        ("code", "何时使用生成器？", "当数据可逐项产生或规模较大时，生成器能延迟计算并降低峰值内存。"),
        ("math", "0.125 写成最简分数。", "0.125=125/1000，约分后是 1/8。"),
        ("tool_use", "日历创建事件后返回什么？", "返回规范化后的时间、时区和事件标识，便于用户核对。"),
        ("long_context", "如何验证长上下文中的数字结论？", "回到数字所在原段，核对单位、时间范围、分母和计算过程，并保留引用位置。"),
        ("agentic", "代理如何处理工具超时？", "区分未知状态与失败；安全重试前检查幂等性，否则查询状态或请求用户决定。"),
        ("enterprise", "为什么生产变更需要回滚方案？", "回滚方案能在指标恶化时快速恢复已知稳定状态，降低故障持续时间。"),
        ("math", "平均数是否一定是样本中的某个值？", "不一定；平均数是总和除以数量，可能不等于任何一个样本值。"),
        ("code", "单元测试和集成测试有何区别？", "单元测试隔离验证小组件，集成测试验证多个组件协作及其边界。"),
    ]
    return [
        {
            "id": f"mopd-{i:02d}",
            "teacher_domain": domain,
            "prompt": prompt,
            "teacher_response": response,
        }
        for i, (domain, prompt, response) in enumerate(examples, 1)
    ]


def build_agentic_rl() -> list[dict[str, Any]]:
    examples = [
        ("天气", "查询北京天气。", "get_weather", '{"city":"北京"}', "search_web"),
        ("时间", "查询纽约当前时间。", "get_time", '{"timezone":"America/New_York"}', "calculator"),
        ("计算", "计算 144 的平方根。", "calculator", '{"expression":"sqrt(144)"}', "get_weather"),
        ("文件", "列出 notes 下的 Markdown 文件。", "list_files", '{"path":"notes","suffix":".md"}', "delete_file"),
        ("搜索", "查找 Liquid AI 的官方博客。", "web_search", '{"query":"site:liquid.ai LFM3"}', "send_email"),
        ("日历", "创建周一 10 点的站会。", "create_event", '{"title":"站会","weekday":"周一","time":"10:00"}', "get_time"),
        ("提醒", "明天 8 点提醒喝水。", "create_reminder", '{"day":"tomorrow","time":"08:00","text":"喝水"}', "create_event"),
        ("库存", "查询 SKU B-2 的库存。", "get_inventory", '{"sku":"B-2"}', "query_tickets"),
        ("工单", "统计打开的工单。", "query_tickets", '{"status":"open","aggregate":"count"}', "get_inventory"),
        ("翻译", "把 hello 翻译成日语。", "translate", '{"text":"hello","target":"ja"}', "calculator"),
        ("路线", "规划 A 到 B 的公交路线。", "route", '{"from":"A","to":"B","mode":"transit"}', "get_weather"),
        ("单位", "把 2.5 小时换算成分钟。", "convert_unit", '{"value":2.5,"from":"hour","to":"minute"}', "translate"),
        ("指标", "查询最近 5 分钟的延迟。", "get_metric", '{"name":"latency","window":"5m"}', "query_tickets"),
        ("文档", "从 invoice-7 提取总金额。", "extract_field", '{"document":"invoice-7","field":"total"}', "list_files"),
        ("允许列表", "检查 example.com 是否允许。", "check_allowlist", '{"domain":"example.com"}', "web_search"),
        ("支持", "为支付失败创建工单。", "create_ticket", '{"category":"payment","summary":"支付失败"}', "send_email"),
        ("产品", "比较 P1 和 P2 的价格。", "compare_products", '{"products":["P1","P2"],"fields":["price"]}', "get_inventory"),
        ("只读命令", "读取 app 版本。", "run_command", '{"command":"app --version","read_only":true}', "delete_file"),
        ("流程", "把已批准报销转给财务。", "advance_workflow", '{"object":"expense","required_state":"approved","next":"finance"}', "create_ticket"),
        ("邮件草稿", "起草项目进度邮件。", "draft_email", '{"audience":"项目组","topic":"项目进度"}', "send_email"),
    ]
    rows = []
    for i, (domain, prompt, tool, args, wrong_tool) in enumerate(examples, 1):
        good = f'<tool_call>{{"name":"{tool}","arguments":{args}}}</tool_call>'
        partial = f'<tool_call>{{"name":"{tool}","arguments":{{}}}}</tool_call>'
        bad = f'<tool_call>{{"name":"{wrong_tool}","arguments":{{}}}}</tool_call>'
        rows.append(
            {
                "id": f"agentic-rl-{i:02d}",
                "domain": domain,
                "prompt": prompt,
                "tools": [tool, wrong_tool],
                "expected_tool": tool,
                "candidates": [
                    {"text": good, "reward": 1.0},
                    {"text": partial, "reward": 0.2},
                    {"text": bad, "reward": -1.0},
                ],
                "success_criteria": "工具名和参数均匹配；未执行真实外部副作用。",
            }
        )
    return rows


def datasets() -> dict[str, list[dict[str, Any]]]:
    return {
        "pretrain.jsonl": build_pretrain(),
        "midtrain.jsonl": build_midtrain(),
        "sft.jsonl": build_sft(),
        "mixed_rl.jsonl": build_mixed_rl(),
        "mopd.jsonl": build_mopd(),
        "agentic_rl.jsonl": build_agentic_rl(),
    }


def write_datasets(output_dir: Path = DATA_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, rows in datasets().items():
        path = output_dir / filename
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        print(f"wrote {len(rows):2d} rows -> {path}")


def check_datasets(data_dir: Path = DATA_DIR) -> None:
    for filename in EXPECTED_FILES:
        path = data_dir / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        if len(rows) != 20:
            raise ValueError(f"{path} must contain exactly 20 rows, got {len(rows)}")
        ids = [row.get("id") for row in rows]
        if len(set(ids)) != 20 or any(value is None for value in ids):
            raise ValueError(f"{path} needs 20 unique non-empty ids")
        print(f"ok    {len(rows):2d} rows <- {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--check", action="store_true", help="validate existing JSONL files instead of writing them")
    args = parser.parse_args()
    if args.check:
        check_datasets(args.output_dir)
    else:
        write_datasets(args.output_dir)
        check_datasets(args.output_dir)


if __name__ == "__main__":
    main()
