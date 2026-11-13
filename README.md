# Claude 注水攻防挑战：模型指纹识别

用尽量少的请求识别 10 个匿名模型 ID 背后的真实模型。只依赖 Python 3 标准库，全部在本地运行。

**当前状态（11-13）**：参考库已通过 ppapi（`https://app.ppapi.us/v1`）采集，26 个模型、638 条有效应答。官方线路还没有采集。

在留出配置上的结果（每个 ID 1 次请求，共 10 次）：

| 条件 | 模拟赛平均分（满分 200） | 首选完全正确 |
|---|---|---|
| 原始参考库 | 186.9，最差一局 140 | 116/136，85% |
| 隔离疑似掉包的记录后 | 190.4，最差一局 150 | 117/130，90% |

第二行把被标记的测试样本也去掉了，所以偏乐观。详见 `plan/2026-11-13-real-data-results.md`。

**这个网关的三个特点**（比赛如果走它，同样适用）：

- 首字节前只等约 30 秒，超时返回 504。所以首轮卷子 `battery_v2` 不含计算题；含计算题的 `battery_v1` 和区分卷在这里对推理模型不可靠。
- 每分钟最多 50 次请求，超过返回 429。采集用 `--workers 4` 或更低。
- 高负载时，标着不同 GPT 型号的请求里有一部分应答像 GPT-5.6 Luna。用 `python3 -m reference.audit` 检查，加 `--write` 隔离。

## 工作原理

1. 每个匿名 ID 先收到一份打包探针卷（一次请求，按 JSON 作答）。
2. 响应被拆成特征：能力题对错、知识截止、行为偏好、随机序列、文风、token 数、速度、响应字段。
3. 特征与参考库逐个候选比对，得到后验概率。
4. 后验低于阈值的 ID 再收到一份区分卷，卷子按“最能分开当前前几名候选”来选。
5. 按期望得分决定每个 ID 是在第一题选中，还是在第二题填名称和版本。

## 目录

| 路径 | 内容 |
|---|---|
| `candidates.py` | 候选模型清单（52 个，分三层），含两条线路上的模型 ID |
| `config.py` | 请求参数、信号权重 |
| `client/` | 请求客户端、计数账本 |
| `probes/` | 探针卷生成器；`frozen/` 是冻结后的卷子，采集和比赛都读它 |
| `features/` | 响应解析与特征提取 |
| `classify/` | 参考库与后验（`library.py`）、选卷与答题决策（`decide.py`） |
| `reference/` | 参考库采集脚本、system prompt 配置；`data/` 存放采集结果 |
| `sim/` | 合成模型、模拟赛评估、本地模拟网关 |
| `run_contest.py` | 比赛当天入口 |
| `tests/` | 测试 |

## 使用步骤

所有命令在项目根目录运行。

### 1. 配置密钥

```bash
export PPAPI_API_KEY=...
export ANTHROPIC_API_KEY=...
```

线路地址在 `endpoints.json`。其他厂商有官方密钥的，在这个文件里加一条线路，并在 `candidates.py` 里给对应模型补上该线路的模型 ID。

### 2. 冒烟测试（66 次请求）

```bash
python3 -m reference.collect --smoke --routes ppapi,anthropic --plan   # 只看计划，不发请求
python3 -m reference.collect --smoke --routes ppapi,anthropic
```

每个模型发 1 次，用来确认请求参数不会被任何候选模型拒绝。**这一步必须先做**：比赛中被拒的请求也计数。

### 3. 采集参考库

| 阶段 | 命令 | 请求数 |
|---|---|---|
| 主卷，ppapi 上密钥可用的候选 | `python3 -m reference.collect --routes ppapi --workers 4` | 726（已完成） |
| 主卷，官方线路的 Claude | `python3 -m reference.collect --routes anthropic --tiers 1 --depth light` | 180 |
| 区分卷，ppapi 的 Claude | `python3 -m reference.collect --routes ppapi --tiers 1 --papers disc_capability,disc_knowledge,disc_behavior` | 756 |

请求数包含冒烟测试已发的部分，已存在的文件会跳过，所以中断后可以直接重跑。加 `--plan` 只看计划，加 `--limit N` 限制本次发送量。

采集会产生真实的 API 费用。冒烟测试的记录里有每次请求的 `usage`，可以据此估算总费用后再决定采集范围。

### 3.5 参考库体检

```bash
python3 -m reference.audit            # 只看报告
python3 -m reference.audit --write    # 把“标签是 A 却高把握像 B”的记录写入 reference/quarantine.json
```

被隔离的记录不会进参考库，也不会进模拟赛。被标记不等于确认掉包，先看报告再决定。

### 4. 模拟赛

```bash
python3 -m sim.evaluate --world replay --split dev --calibrate    # 校准并比较策略
python3 -m sim.evaluate --world replay --split test               # 最终估计，只跑一次
python3 -m sim.evaluate --world replay --lib-routes ppapi --ep-routes anthropic   # 跨线路
```

输出每种局面下的平均分、请求数、名称和版本准确率、第一题命中情况，以及后验的校准情况。

### 5. 彩排

```bash
python3 -m sim.gateway --world replay --split test --port 8765
CONTEST_API_KEY=x python3 run_contest.py --base-url http://127.0.0.1:8765/v1 --target claude-opus-5.5 --run rehearsal
```

答案在 `runs/gateway_truth.json`，跑完再看。

### 6. 比赛当天

```bash
export CONTEST_API_KEY=...
python3 run_contest.py --base-url https://主办方地址/v1 --target claude-opus-5.5 --dry-run
python3 run_contest.py --base-url https://主办方地址/v1 --target claude-opus-5.5
```

- `--target` 用 `candidates.py` 里的 key，`python3 candidates.py` 可以列出全部。
- 所有响应和账本存在 `runs/<run>/`。用同一个 `--run` 重跑会从断点继续，已发过的请求不会再发。
- 第一发如果被拒，程序立即停止，只损失 1 次请求。
- `--policy single` 每个 ID 只发 1 次；默认 `sequential`。

## 请求规则

- 请求只带 `model`、`messages`、`max_tokens`、`stream`、`stream_options`。不带 `temperature`、`top_p`、`n`、`logprobs`、`response_format`：Claude 4.7 及之后的模型在 OpenAI 兼容接口上会拒绝小于 1 的 `temperature`。
- 只有连接没建立时才重试（连接被拒、DNS、TLS 握手）。请求一旦发出，任何 HTTP 响应、读超时、断流都不重试，并记入账本。
- 账本在请求发出前落盘，程序崩溃也不会少记。

## 测试

```bash
python3 -m unittest tests.test_all
python3 -m sim.evaluate --world synthetic --policies single,sequential,split
```

`--world synthetic` 用的是按候选清单虚构的统计模型，不是真模型。它只能说明代码能跑通，它的分数不代表真实准确率。

## 已知的限制

- 参考库的标签是“ppapi 称为 X 的模型”，没有用官方线路核对过。
- 区分卷还没有采集参考数据，所以 `sequential` 策略目前和 `single` 等价。
- Opus 4.6、Opus 4.7、Sonnet 4.6 在 11-13 上午起网关返回“没有可用通道”，每个只有 12 条样本。
- `reference/configs.py` 里的验证用 system prompt 和参考库的是同一个人写的，估出来的分数会偏高。建议由另一位组员重写 dev 和 test 两组。
- 候选清单以外的模型只能被标记为“疑似库外模型”，填的是最接近的候选。
- 官方已下线、只在 Bedrock 保留的 Opus 4.1、Opus 4、Sonnet 4 在两条采集线路上都拿不到，没有进清单。
- 名称和版本的填写格式（如 “Claude / Opus 5.5”）是按题面例子推的，非 Claude 模型的格式没有向主办方确认。
- 多个 ID 背后是同一模型时，目前各自独立判断，没有合并证据。
- 不支持 HTTP 代理。
