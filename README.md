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

## 架构

系统分成两条流水线，共用同一套探针卷、特征提取和参考库：**离线**采集参考库，**在线**（比赛当天）对 10 个匿名 ID 做识别。

```
                       ┌────────────── 离线：建库 ──────────────┐
 candidates.py ──┐     │                                        │
 reference/      │     │  reference.collect ──► reference/data/ │
 configs.py ─────┼────►│  （候选 × 卷子 × system prompt）        │
 probes/frozen ──┘     │        │                               │
                       │        ▼                               │
                       │  reference.audit ──► quarantine.json   │
                       └────────┬───────────────────────────────┘
                                │ reference.store 读取、转成样本
                                ▼
                       features.extract（响应 → 特征）
                                │
                                ▼
                       classify.library.Library（参考库 + 后验）
                                ▲                    │
                ┌───────────────┘                    │
                │                                    ▼
 ┌──────────── 在线：比赛 ────────────────────────────────────────┐
 │ run_contest.py                                                 │
 │   └─ classify.decide.Session                                   │
 │        1. 每个 ID 发首轮卷子 ─► client.api.HttpTransport ─► 网关 │
 │        2. extract → Library.posterior                          │
 │        3. 后验 < 阈值的 ID：choose_paper 选区分卷再发            │
 │        4. answers()：按期望得分决定「选中」还是「填名称/版本」    │
 │   └─ client.ledger.Ledger（请求前落盘，硬上限）                 │
 └────────────────────────────────────────────────────────────────┘

 sim/：用 ReplayWorld（回放真实参考数据）或 SyntheticWorld（合成）
      替换网关，同一套 Session 代码跑模拟赛；sim.gateway 起本地 HTTP 网关用于彩排。
```

### 分层与职责

| 层 | 模块 | 职责 | 依赖 |
|---|---|---|---|
| 数据定义 | `candidates.py`、`config.py` | 候选清单（答案空间）、请求参数 `REQUEST_PROFILE`、信号权重、路径 | 无 |
| 探针 | `probes/papers.py` | 生成并冻结打包卷子；每份卷子是一个 JSON 作答的问卷，含能力、知识、行为、随机序列、文风和分词器探针 | `config` |
| 传输 | `client/api.py`、`client/ledger.py` | 流式请求、解析 SSE（首字节时间、usage、推理字段）、重试规则；账本计数 | `config` |
| 特征 | `features/extract.py` | 把一条响应变成 `{卷:特征名: (类型, 信号组, 值)}` | `probes` |
| 分类 | `classify/library.py` | 按候选聚合特征统计，计算对数似然和后验，估计一份卷子能分开两个候选的程度 | `candidates`、`config` |
| 决策 | `classify/decide.py` | `Session`：发首轮、选区分卷、预算控制、按期望得分出答案 | 以上全部 |
| 参考库 | `reference/` | 采集、落盘、体检、system prompt 配置 | `client`、`probes`、`features` |
| 评估 | `sim/` | 构造局面、回放、打分、校准、本地网关 | `classify`、`reference` |
| 入口 | `run_contest.py` | 装配上述各层，写出 `answers.json` | 全部 |

依赖方向是自上而下，`client` 和 `classify` 不互相引用；`Session` 只要求传输对象有 `ask(mid, paper, name)` 方法，所以真实 HTTP 传输、模拟赛的 `EpisodeTransport` 和彩排网关可以互换。

### 数据流

1. **探针**：`probes.papers --freeze` 把卷子写入 `probes/frozen/*.json`。采集、模拟和比赛都读这份冻结文件，所以提示词字节完全一致，`prompt_tokens` 才能跨环境比较。
2. **采集**：`reference.collect` 对每个候选、每份卷子、每组 system prompt（`reference/configs.py` 的 ref/dev/test 三个划分）发请求，原始响应存到 `reference/data/<线路>/<模型>/<卷>/<配置>__<序号>.json|.sse`。
3. **特征**：`extract` 提取 11 个信号组（`tok` 分词器、`cap` 能力、`know` 知识、`beh` 行为、`rand` 随机序列、`style` 文风、`fmt` 格式、`reason` 推理量、`self` 自述、`speed` 速度、`meta` 响应字段）。特征类型有 `bin`、`bag`、`lognum`、`text`、`ptok` 五种，各有自己的似然模型。
4. **后验**：`Library` 对每个候选算 `log 先验 + 温度 × Σ 组权重 × 组对数似然`，再归一化。候选清单外加一个 `OTHER` 伪候选，用全体样本的背景分布表示“库外模型”。稀疏的候选统计会向背景分布收缩。
5. **决策**：`Session.answers()` 比较两个期望得分：选为指定模型 `20p − 10(1−p)`，填名称和版本 `10·P(名称对) + 10·P(版本对)`，取大者。
6. **追加请求**：首轮后验低于 `stop_p` 的 ID，用 `choose_paper` 在区分卷里选 Bhattacharyya 距离加权后增益最大的一份，增益低于 `MIN_GAIN` 就不再发。

### 关键设计决定

- **请求计数是硬约束**。`Ledger` 在请求发出前就把名额写盘（fsync），崩溃也不会少记；`Session` 用 `run_dir` 里已存在的记录断点续跑，已付费的请求不重发。
- **请求参数固定**。所有环境共用 `REQUEST_PROFILE`，不带 `temperature` 等参数，避免被部分模型拒绝（被拒的请求也计数）。第一发被拒就抛 `AbortRun` 停止。
- **不使用响应里的 `model` 字段**。采集线路会回显真实模型名，用它会看起来“很准”但在比赛里没有意义。
- **参考库按配置汇总**。每个候选的统计汇集了多种 system prompt 下的响应，伪装带来的差异只会拓宽分布，不会让匹配失效。
- **网关重算 usage 时自动降级**。`_check_usage` 发现所有 ID 的 `prompt_tokens` 几乎相同，就关闭 `tok` 组。
- **库的标签可能不干净**。`reference.audit` 用 5 折交叉检验找出“标签是 A 却高把握像 B”的记录，写入 `quarantine.json`，`reference.store` 加载时跳过。
- **权重可覆盖**。`weights.json` 只写要改的键，会合并进 `config.DEFAULT_WEIGHTS`；`sim.evaluate --calibrate` 用来调这些值。

## 目录

### 文件结构图

```
ClaudeChallenge/
├── run_contest.py            # 比赛当天入口：装配各层，写出 answers.json
├── candidates.py             # 候选模型清单（52 个，三层）与两条线路的模型 ID
├── config.py                 # 路径、REQUEST_PROFILE、信号权重、读取密钥
├── endpoints.json            # 线路地址与密钥环境变量名（不含密钥）
├── endpoints.example.json    # endpoints.json 模板
├── weights.json              # 覆盖 config.DEFAULT_WEIGHTS 的权重
│
├── probes/                   # 探针卷
│   ├── papers.py             #   生成、冻结、加载卷子
│   └── frozen/               #   冻结后的卷子（采集与比赛都读它）
│       ├── battery_v1.json   #     含计算题的主卷
│       ├── battery_v2.json   #     首轮主卷（不含计算题）
│       ├── half_a.json / half_b.json                       # split 策略的两半
│       └── disc_capability / disc_knowledge / disc_behavior.json   # 区分卷
│
├── client/                   # 传输层
│   ├── api.py                #   流式请求、SSE 解析、重试规则、列模型
│   └── ledger.py             #   请求账本（发出前落盘，硬上限）
│
├── features/
│   └── extract.py            # 响应 → 特征 {卷:特征名: (类型, 信号组, 值)}
│
├── classify/                 # 识别核心
│   ├── library.py            #   参考库、似然、后验、区分卷增益
│   └── decide.py             #   Session：发卷、选卷、预算、答题决策
│
├── reference/                # 参考库的采集与维护
│   ├── collect.py            #   采集脚本（冒烟、续跑、--plan）
│   ├── audit.py              #   标签体检，疑似掉包记录
│   ├── store.py              #   读取记录、转成库样本
│   ├── configs.py            #   system prompt 配置（ref / dev / test）
│   ├── quarantine.json       #   被隔离的记录
│   └── data/<线路>/<模型>/<卷>/<配置>__<序号>.json|.sse   # 原始响应
│
├── sim/                      # 模拟与评估
│   ├── evaluate.py           #   ReplayWorld、局面构造、打分、校准
│   ├── synthetic.py          #   SyntheticWorld（合成模型）
│   └── gateway.py            #   本地模拟网关（彩排用）
│
├── tests/
│   └── test_all.py           # 单元测试
├── runs/                     # 运行产物：<run>/ledger.jsonl、响应、answers.json
├── plan/                     # 计划与结果文档（按日期命名）
├── CLAUDE.md                 # 项目说明（给 Claude Code）
└── README.md
```

### 路径说明

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
CONTEST_API_KEY=x python3 run_contest.py --base-url http://127.0.0.1:8765/v1 --run rehearsal
```

答案在 `runs/gateway_truth.json`，跑完再看。网关每次启动都会覆盖这个文件。

### 6. 比赛当天

比赛走 PP（`https://app.ppapi.us/v1`），用主办方发的密钥。默认参数就是比赛设置：`/v1/messages` 协议、指定模型 Opus 5、`sequential` 策略、参考库用同一线路的 `ppapi_msg` 数据。

**前一天**：确认 `git status` 干净，当天跑的就是彩排过的代码。

**1. 设置密钥**（`read -s` 不回显，不进命令历史）

```bash
read -s CONTEST_API_KEY && export CONTEST_API_KEY
```

**2. 核对 ID 名单**。默认 `model-a` 到 `model-j`。不一样就加 `--ids model-a,model-b,...`（逗号分隔，不加空格）。

**3. 干跑**（不发请求）

```bash
python3 run_contest.py --base-url https://app.ppapi.us/v1 --run contest --dry-run
```

确认输出里有：`library: 23 models`、`target in library: True`、`protocol: anthropic`、`would send 20 first-round requests`。

**4. 正式运行**（`caffeinate` 防止 Mac 休眠）

```bash
caffeinate -is python3 run_contest.py --base-url https://app.ppapi.us/v1 --run contest
```

请求逐条发送。首轮每个 ID 发主卷和语气卷各 1 次，共 20 次；之后对把握不够的 ID 补发，总数最多 40 次，彩排里约 23 到 25 次（约 0.25 美元）。账本硬上限 100（`--cap`），比赛允许 200。

**5. 填答卷**。结果打印成表，并存到 `runs/contest/answers.json`。

- “第一题”列为“选”的 ID：第一题勾选。
- 其余 ID：按“第二题（答卷选项）”列填写。文字和答卷下拉选项逐字一致，包括选项本身的错别字 “Cluade”。
- 备注“把握低”“无参考数据，按家族推断”的 ID 最可能填错，讲解时可以说明。

**出问题时**

| 情况 | 怎么办 |
|---|---|
| 中途断网或程序崩溃 | 用同一个 `--run contest` 重跑。已发过的请求从磁盘读取，不会再发 |
| `ABORTED: first request was rejected (HTTP 4xx)` | 请求参数或密钥被拒，只损失 1 次。先看报错内容，不要直接重跑 |
| 第一发是 504 或 503（网关超时） | 同一个 `--run` 重跑会读到这条已计数的失败记录并再次停止，不发新请求。改用 `--run contest2`；新目录的账本从 0 开始，实际已用次数要再加 1 |
| 401 或 403 | 密钥不对或没导出 |

**要向主办方确认**

- 比赛密钥在 PP 上能不能用 `/v1/messages`。只能用 OpenAI 兼容接口时加 `--protocol openai`，参考库会换成旧接口的 `ppapi` 数据，那条路径在新规则下没有重新评估过。
- 10 个 ID 的确切名字。
- 失败的请求（504、429）算不算次数。

其他参数：`--target` 用 `candidates.py` 里的 key（`python3 candidates.py` 列出全部）；`--policy single` 每个 ID 只发首轮 2 次，不补发。

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
