# 证据复验实验 v0.0

本版跑通受控实验。模型在两次长度触发的压缩后，决定是否读取证据、查看当前代码、执行局部检查、重新考虑候选或提交。运行器发放预定义候选，提交时导出选中的 E1 工作区，评测器从独立副本运行终态检查。适合测证据使用和预算策略。自动写补丁、任意 shell、SlopCodeBench/SWE-Milestone 的自由写码闭环和 SWE-rebench 轨迹审计属于后续阶段。

## BYL 上的 Qwen3.5-9B 与批次预检

在本机修改代码并同步后，在 BYL 的项目目录执行。脚本通过 Qwen3.5 条件生成类载入官方 BF16 快照，生成时只提供文本，计数和推理共用关闭思考模式的 chat template。

```bash
cd /mnt/sata1/beihang-cuda_backup
conda activate /mnt/sata1/beihang-cuda_backup/.conda/v00
export MODEL_DIR="$PWD/models/Qwen3.5-9B"
export TMPDIR="$PWD/.conda/tmp"
mkdir -p "$TMPDIR"

# 无需 Torch 或 GPU。校验文件头、索引和每个分片的末尾。
python code/version0_0/run.py model-check --model-dir "$MODEL_DIR"

# 项目内的 tokenizer 副本可独立于服务器上的权重下载执行预检。
python code/version0_0/run.py preflight --tokenizer-dir results/pilot/tokenizer \
  --case results/pilot/case-*/case.json \
  --output results/pilot-check
```

GPU 验证前在这个 Conda 环境内安装与驱动匹配的 PyTorch CUDA wheel，再安装 `requirements-model.txt`。使用 BYL 启动器会在启动前两次检查整机占用。只有至少两张卡空闲且其中有 A800 时才会启动。使用唯一输出目录。

```bash
python code/version0_0/byl_gpu.py model-smoke --model-dir "$MODEL_DIR"
python code/version0_0/byl_gpu.py run --model-dir "$MODEL_DIR" \
  --case results/pilot/case-fail-to-pass/case.json results/pilot/case-independent-blocker/case.json \
  --output results/pilot-9b --arms summary logs cache eager selective
```

批次运行共用一次模型加载，每个用例保留独立的证据、预算和输出。查看 `batch_summary.json`、各组 `events.jsonl` 和 `summary.json`。事件中区分复验建议、模型动作与执行结果。例行组对已观测但待确认的检查执行复验，按需组显示因独立阻断或更优候选而可推迟的检查。两组共用时效性判断。

### 提交前验证对照

用 `--arms eager selective eager-visible guarded` 单独运行这组对照。默认仍是原来的五组。

- `eager-visible` 保持例行复验，直接把各候选当前的公开测试状态放进 E1 决策输入。模型仍负责最终选择。
- `verified-visible` 在 E1 决策前运行与提交检查相同的公开验证策略，把当前证据与待查项展示给模型，最终候选仍由模型选定。
- `verified-explicit` 沿用 `verified-visible` 的验证策略，在展示时给当前通过和失败证据分别命名，只列出缺少当前结果的测试名称。历史结果仍可通过证据工具检索。
- `verified-scoped` 在 `verified-explicit` 的基础上给旧工作摘要标注时间范围，提醒模型用当前证据判断 E1 可行性。验证策略和最终提交权相同。
- `guarded` 在模型提出提交时才处理可能改变选择的公开检查。已有独立失败证据的候选直接排除。其余按任务已声明的候选优先级评估，优先重查过期的失败证据，一旦找到通过所有公开检查的候选就提交，并记录原始提议与最终结果。预算耗尽时，可提交已验证的备选并记录尚未确定的高优先级候选。当前公开测试不足以支持任何候选时停止提交。模型主动结束时保留其决定。
- 这些对照基于当前快照重建候选状态。提交检查只访问公开测试和实际执行记录，独立判分继续使用单独的测试。比较时同时报告模型提议、提交前执行次数、未解决候选及最终选择。控制器按明确优先级更换候选属于运行时效果。
- 任务未声明候选优先级时，提交检查只验证模型提出的候选。
- `guarded-pruned` 额外处理 E1 的显式测试请求。如果同一候选仍被另一条有效的公开失败证据排除，运行器会跳过其待查项，并把阻断证据返回给模型。跳过的请求单独记录，不计为物理测试。

把上一轮的 `batch_summary.json` 传给 `report_pilot.py` 的 `--baseline`，可以按用例比较新增组和历史对照。报告会核对基线覆盖和参考结果，并分别统计模型提议、运行器改选、测试执行和漏检。

`decision_cases.py` 从同一个已核验的 tqdm #1 快照创建两种文件内容不同的实现，控制用户优先级、候选标签、持续阻断和反向回归。生成时对作用于候选的完整工作区做哈希检查。使用不同输入执行独立集成验收。六个条件都来自同一个缺陷，用于诊断，不应计为六个独立缺陷。

```bash
python code/version0_0/decision_cases.py prepare \
  --upstream-case results/pilot/case-tqdm-fail-to-pass --output results/pilot-distinct
python code/version0_0/decision_cases.py audit --case results/pilot-distinct/case-*/case.json
python code/version0_0/run.py preflight --tokenizer-dir "$MODEL_DIR" \
  --case results/pilot-distinct/case-*/case.json --arms guarded --output results/pilot-distinct-check
python code/version0_0/byl_gpu.py run --model-dir "$MODEL_DIR" \
  --case results/pilot-distinct/case-*/case.json \
  --arms selective eager-visible verified-visible guarded --output results/pilot-distinct-8k
```

扩展的先导用例使用 `pilot_cases.py` 从上游的固定提交归档生成，归档的项目、编号和提交哈希见该文件。PySnooper 用例的测试依赖列在 `requirements-pilot.txt`。扩展用例在 E1 重新从当前快照读取源材料，仍按实际 token 长度触发压缩。统计时把同一批次所有组的失败和预算耗尽一并计入。例如使用 `python code/version0_0/report_pilot.py results/pilot-9b/batch_summary.json --output results/pilot-report.json` 生成配对报告。

### 从同一检查点续跑

`--shared-boundary` 让每个用例先以 `cache` 策略运行一次公共前缀，在第二次长度压缩后保存 `boundary.json`。各组续跑前独立读入同一文件，并核对 SHA-256。文件记录已消耗的模型和测试预算、证据及压缩文本。分支沿用这些预算消耗，总墙钟纳入公共前缀，报告也单列实际续跑时间。`prefix/events.jsonl` 保存分支之前的事件。

`--change-notice silent` 保留第二阶段和当前仓库材料，隐藏明确的切换通知以及决策输入里的 `epoch` 字段。模型仍能从代码、证据工具和公开检查观察变化。带预验的分支会主动展示当前测试证据，无预验的 `cache` 分支用于观察模型自主查证。公开检查和独立判分仍实际在 E1 执行。

`--change-notice paired` 与 `--shared-boundary` 一起使用。每个策略运行两个分支，共用静默生成的第二次压缩检查点。明示分支在首次续跑时收到一条工作区进入最终快照的通知。报告逐对检查去除通知后的首个决策输入是否一致。该通知也适用于文件未变的快照；生成用例时可用 `--scenarios no-change relevant-stable` 分别测试文件未变和相关文件只改注释的情况。

### 比较提交阶段的复验调度

`guarded-eager` 和 `guarded` 共用指纹、候选优先级和提交检查。例行策略在收到提交提议后先复验所有过期的已观测项，然后进入同一提交检查。按需策略优先查可能影响当前选择的候选，用仍有效的失败证据排除被独立条件阻断的候选。提交事件记录进入调度前的证据、提议和剩余测试次数的摘要哈希。可借此检查两组是否得到相同的实际输入。

`gate_sweep.py` 提供无模型的配对预算扫描。在各条件下固定 E0 已通过全部公开检查的提议，运行真实验证器和独立验收。零预算也记录在案。报告先在同一 BugsInPy 缺陷的衍生用例内求均值，再对缺陷求均值。GPU 续跑用于观察模型会提出什么提交，CPU 预算扫描量化给定同一提议时调度的影响。两类结果都保留原始测试记录。

```bash
python code/version0_0/decision_cases.py prepare-tqdm-regressions \
  --source-directory results/pilot-expanded-verified --output results/pilot-scheduler
python code/version0_0/gate_sweep.py \
  --case results/pilot-scheduler/case-*/case.json results/pilot-notice-pairs/case-*/case.json \
  --budgets 0 1 2 3 12 --output results/pilot-scheduler-budget
```

```bash
python code/version0_0/decision_cases.py prepare \
  --upstream-case results/pilot/case-tqdm-fail-to-pass --output results/pilot-shared-boundary \
  --reread-after-change --scenarios prefer-native adapter-regression
python code/version0_0/decision_cases.py prepare-writer \
  --upstream-case results/pilot-expanded-verified/case-pysnooper-3 --output results/pilot-shared-boundary
python code/version0_0/byl_gpu.py run --model-dir "$MODEL_DIR" \
  --shared-boundary --change-notice silent --case results/pilot-shared-boundary/case-*/case.json \
  --arms cache verified-explicit verified-scoped guarded --output results/pilot-shared-silent
```

## 编写与验证计划

| 步骤 | 交付 | 要发现的问题 |
| --- | --- | --- |
| 1 | 生成 E0/E1 和候选 A/B，预跑两环境下的所有候选检查 | 参考结果是否可重现，旧测试是否覆盖当前主张，环境变化是否真的翻转结果 |
| 2 | 用 chat template 计数，默认工作窗口 8,192、压缩阈值 6,144、单次输出预留 1,024 | 原始工具观测是否离开逐字历史，第二阶段够不够触发，摘要与动作 JSON 是否正常 |
| 3 | 同一初始证据与预算运行五组，记录每次物理执行和缓存命中 | 有独立阻断时能否跳过复查，旧失败恢复时能否改变选择，缓存开销是否抵消收益 |
| 4 | 独立判分并保留参考矩阵、事件和成本 | 终态通过是否藏住了漏检和不必要的复验，能否看到模型能力与预算限制 |
| 5 | 用本地 7B 先跑合成双案例及 BugsInPy 一例，再扩到按仓库分割的 6–10 个缺陷 | 估计吞吐、可重现率、模型格式错误率和有效配对数 |

## CPU 冒烟

从项目根目录执行。CPU 流程只使用 Python 3.10+ 标准库。下列输出目录需是新路径。

```bash
python code/version0_0/run.py prepare-demo --scenario independent-blocker --output results/v00-demo-case
python code/version0_0/run.py run --scripted --case results/v00-demo-case/case.json --output results/v00-demo-run
python -m unittest discover -s code/version0_0 -v
```

其他合成条件为 `unrelated`、`relevant-stable`、`fail-to-pass` 和 `pass-to-fail`。脚本后端用于自检，默认按正则切分计数，输出由固定规则生成。安装模型依赖后，可给 `--scripted` 加 `--tokenizer-dir <模型快照目录>`，只加载 tokenizer 来检查 8K 或 16K 的实际打包长度。模型行为结论需要真实权重运行。

## BugsInPy 首例

使用 [BugsInPy tqdm #1 元数据](https://github.com/soarsmu/BugsInPy/tree/master/projects/tqdm/bugs/1) 指定的两次提交。A 使用当前仓库行为，B 覆盖为固定版本的实现。本版用一个派生的编号回归断言和单独的集成断言，固定 progress wrapper 来复现错误。原始验证器使用 pytest 和历史 Python，派生版本的判分应在论文中注明。三种变化条件共用同一个缺陷，仅算一个缺陷样本。

```bash
git clone --filter=blob:none https://github.com/tqdm/tqdm.git "$HOME/v00-tqdm"
python code/version0_0/run.py prepare-tqdm --repo "$HOME/v00-tqdm" \
  --scenario fail-to-pass --output results/v00-tqdm-case
python code/version0_0/run.py check --case results/v00-tqdm-case/case.json
python code/version0_0/run.py run --scripted \
  --case results/v00-tqdm-case/case.json --output results/v00-tqdm-scripted
```

也可以提供 GitHub 的源码 tar 包，使用 `--buggy-archive ... --fixed-archive ...`。两次提交分别为 `8cc777fe8401a05d07f2c97e65d15e4460feab88` 和 `c0dcf39b046d1b4ff6de14ac99ad9a1b10487512`。使用 `--scenario unrelated` 或 `relevant-stable` 生成配对对照。扩展数据时，复制生成的 `case.json` 结构，补齐两快照、候选覆盖文件、可信测试命令、各测试的具体依赖和彼此独立的提交验收输入。可用 `--test-python` 指定旧仓库专用解释器。若快照使用不同的预配环境，显式填写 `epoch_environment_ids`，并由测试命令通过工作区配置选择对应环境。测试命令应在退出码为 0 时以最后一行 JSON 报告 `passed`、`assertion_failed` 或 `not_collected`。执行验收只面向经审查的仓库与命令，临时工作区用于隔离各次结果。

## 普通 Ubuntu 上运行单卡

将 `code/version0_0` 放到自己的目录。用 `nvidia-smi` 确认机器、你的 GPU 分配和空闲显存。创建独立 Python 3.11 环境，按已安装驱动选择 [PyTorch CUDA wheel](https://pytorch.org/get-started/locally/)，再安装模型依赖。固定模型 revision（Hugging Face 提交哈希），把完整权重放在个人目录或使用已有的只读模型缓存。运行器从该目录离线加载。

```bash
python3.11 -m venv .venv-v00
source .venv-v00/bin/activate
# 先安装与你的驱动相匹配的 PyTorch CUDA wheel。
python -m pip install -r code/version0_0/requirements-model.txt
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'

# 已获分配的物理 GPU 编号也可以是其他值。
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=2
hf download Qwen/Qwen2.5-Coder-7B-Instruct --revision c03e6d358207e414f1eca0bb1891e29f1db0e242 \
  --local-dir "$HOME/models/qwen25-coder-7b"
python code/version0_0/run.py run \
  --model-dir "$HOME/models/qwen25-coder-7b" --device cuda:0 \
  --case results/v00-tqdm-case/case.json --output results/v00-tqdm-7b \
  --arms summary logs cache eager selective
```

共享机器上串行运行。默认每个组最多 110,000 个累计模型 token、12 次物理测试、90 秒测试进程墙钟和每阶段 8 次模型动作。模型加载和最终验收成本另统计。可用 `tmux` 维持 SSH 断开后的实验会话，按服务器管理规则使用已分配的 GPU。一次跑一个 case，使用唯一输出目录。完成一个小批量后再根据实测 A800 占卡时长决定规模，50 A800 GPU·h 是先导的总规划上限。

| 模型 | 单卡用途 | BF16 权重估算 |
| --- | --- | --- |
| [Qwen2.5-Coder-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct) | 首轮固定模型，测工具协议和证据选择 | 约 15 GB |
| [Qwen2.5-Coder-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct) | 检查 7B 的能力下限，适合先做迁移对照 | 约 30 GB |
| [Qwen3-Coder-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct) | 后续迁移对照。MoE 的全部专家权重占显存 | 约 61 GB |

权重估算按每参数 2 字节，不含 KV cache、临时张量和运行时分配。30B 先测实际峰值，当前后端使用 BF16 单卡和贪心解码。8K 运行稳定后可用 `--window 16384 --trigger 12288 --reserve 1024` 作为独立实验条件。源文件材料及总模型 token 档位需要随窗口重新预检。

## 结果怎么读

- `summary` 只保留模型摘要。 `logs` 工具提供历史测试结果。 `cache` 增加按声明依赖指纹复用。 `eager` 在 E1 主动复查失效的已知检查。 `selective` 给出已有独立阻断、仍可能影响决定的检查和测试成本提示，模型仍可自行决定下一步。
- 每组共享 E0 初始观测与同一按文件排序的材料来源。第二个阈值有意保留第一阶段造成的历史差异，因此实际注入的文本跨度可能不同。 `prefix_span` 和 `compaction` 事件保留边界信息。长程材料用于制造上下文压力，真实训练或正式实验还应验证每段材料与任务相关性。
- `reference.json` 是独立真值矩阵。每组的 `events.jsonl` 包含压缩、动作、缓存、实际执行和 token 收支。 `summary.json` 中分别记录当前候选是否通过、是否选到优先可行候选、复用了多少次、哪些旧测试翻转后的检查被执行。显式引用旧证据的统计只覆盖模型在 JSON 中报告的引用。
- 当前指纹基于声明的依赖和环境标识。临时副本内容、声明依赖之外的隐式系统状态和随机性需要在扩展数据时补充审计。 `declared_unchanged_but_flipped` 可发现参考矩阵中已经出现的依赖遗漏。
- 下一批数据先从 [BugsInPy 复现版](https://github.com/reproducing-research-projects/BugsInPy) 筛选，也可接 [PyBugHive](https://github.com/pybughive/pybughive)。后续闭环优先用 [SWE-Milestone](https://swe-milestone.com/)，用 [SWE-rebench 公开轨迹](https://huggingface.co/datasets/nebius/SWE-rebench-leaderboard-trajectories) 审计现实中的重复验证。

本版另建小型运行器，参考了仓库 12 号 Session 的压缩与恢复边界设计。受控协议把逐请求 token 计量、不可变的候选快照和测试子进程结果放在同一个实验入口。
