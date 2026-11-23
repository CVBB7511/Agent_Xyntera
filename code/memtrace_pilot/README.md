# MemTrace × Qwen3.5-9B 配对先导

## 256K 恢复短回放

当前默认窗口为 262,144 tokens。256K 的结果单独写在 `results/memtrace-pilot/context-262144/`，原有 128K 文件可用 `--context-window 131072` 查看。短回放读取上次 SWE-Milestone 失败时保存的检查点，并用 Qwen tokenizer 对重建的请求计数。若超预算立即结束。加 `--generate` 时，CPU 预检通过后最多启动一次推理来检查召回工具调用。

```bash
cd /mnt/sata1/beihang-cuda_backup
bash code/memtrace_pilot/refresh-memtrace.sh
.conda/v00/bin/python code/memtrace_pilot/pilot.py recovery-check --generate
```

省略 `--generate` 只运行 CPU 预检。正式运行前如需测单卡接近完整窗口的推理，用 `pilot.py smoke --context-window 262144`。恢复短回放与正式基准各自记录结果。

`recovery-check --generate` 最多占卡 180 秒。短回放、推理预检和正式运行均要求启动前至少有两张卡的整卡显存占用低于 75%，选择显存占用比例最低且有分配余量的 A800。PyTorch 的分配上限为单卡显存的 70% 减去启动前的已用显存。运行中只采样所选卡的整卡显存，达到 75% 时结束自己的任务。

显存查询超时或返回空值会自动重试，连续三次无有效读数时结束本次任务。单题重跑可在 `run --task ...` 加 `--retry-monitor-interruption`，归档因监控中断且尚未提交的尝试后重新运行。

## 256K Dask 单题配对

```bash
cd /mnt/sata1/beihang-cuda_backup
p=.conda/v00/bin/python
"$p" -u code/memtrace_pilot/pilot.py smoke --context-window 262144 --smoke-input-tokens 147456 &&
  "$p" -u code/memtrace_pilot/pilot.py run --context-window 262144 \
    --benchmark SWE-EVO --task dask__dask_2022.9.2_2022.10.0 --time-limit 3600
```

预检输入约 144K tokens。两组各最多运行一小时，评分结果、轨迹和每组的整卡显存回执写到 `results/memtrace-pilot/context-262144/`。

已有 Dask 补丁时，可直接在 BYL 项目目录补跑官方评分。项目 Docker 缓存中的镜像按冻结摘要验明后直接复用。评分使用 CPU。

```bash
.conda/v00/bin/python -u code/memtrace_pilot/pilot.py grade --context-window 262144 \
  --benchmark SWE-EVO --task dask__dask_2022.9.2_2022.10.0
```

## BYL 手动续跑原有 128K SWE-Milestone

在本地项目根目录用 PowerShell 同步修复，然后登录 BYL。同步脚本使用现有 SSH 密钥。

```powershell
./code/memtrace_pilot/byl.ps1 sync
ssh beihang@10.132.80.126
```

在 BYL 执行以下命令。刷新脚本只用项目已有的 Conda 环境离线重装 MemTrace。

```bash
cd /mnt/sata1/beihang-cuda_backup
bash code/memtrace_pilot/refresh-memtrace.sh && {
  log="results/memtrace-pilot/milestone-$(date +%Y%m%d-%H%M%S).log"
  nohup .conda/v00/bin/python -u code/memtrace_pilot/pilot.py run \
    --benchmark SWE-Milestone --context-window 131072 --retry-context-overflow \
    >"$log" 2>&1 </dev/null &
  echo "PID=$! log=$log"
}
```

记下日志路径即可退出 SSH。`--retry-context-overflow` 只允许归档本次已确认的旧 MemTrace 故障，原记录在 `results/memtrace-pilot/attempt-history/`。已评分和已留存官方提交的尝试保留。启动每次推理前自动核对 GPU 规则，空闲卡不足时进程退出。稍后可用同一命令续跑。

如已有保存的提交但缺少评分，登录 BYL 后单独运行评分。`grade` 只使用存量补丁和里程碑快照，`report` 生成汇总，两者均无需 GPU。

```bash
cd /mnt/sata1/beihang-cuda_backup
p=.conda/v00/bin/python
"$p" code/memtrace_pilot/pilot.py grade --benchmark SWE-Milestone --context-window 131072
"$p" code/memtrace_pilot/pilot.py report --context-window 131072
```

把 `--benchmark` 改为 `DeepSWE` 或 `SWE-EVO` 可只处理该基准。省略它会处理全部。评分结束后，可在本地用 `./code/memtrace_pilot/byl.ps1 fetch` 取回结果。新试验使用修补后的 MemTrace build，历史成绩保留原运行的源码标识。

## 从头配置

在本地项目根目录运行以下 PowerShell 命令。推理、评分和数据暂存位于 BYL 项目数据盘。运行编号、诊断小仓库和正式任务分别存放，正式任务以冻结任务和实验组为键，重复启动会检查已有尝试。

```powershell
./code/memtrace_pilot/byl.ps1 sync
./code/memtrace_pilot/byl.ps1 bootstrap
./code/memtrace_pilot/byl.ps1 prepare
./code/memtrace_pilot/byl.ps1 preflight
./code/memtrace_pilot/byl.ps1 smoke
./code/memtrace_pilot/byl.ps1 run
./code/memtrace_pilot/byl.ps1 grade
./code/memtrace_pilot/byl.ps1 report
./code/memtrace_pilot/byl.ps1 fetch
```

在 `results/memtrace-pilot` 中，`inputs` 保存上游版本与镜像，`controls` 保存评分器对空提交和参考提交的判断，`runs` 保存正式尝试及最终补丁，`gpu-time.json` 逐次累计占卡时间。`performance.csv` 按任务列出成绩，`summary.csv` 按基准和实验组列出分母、时间和原论文的全量参考值。空单元格表示该值尚不可得。

SWE-EVO 保留 Conan 和 Dask 两个原始任务位。增补按实例 ID 检查各仓库首个候选，顺序和资格记录在 `inputs/evo-screening.json`。SWE-Milestone 预检逐节点比较空提交和参考提交的官方 Score，并要求至少一个节点的参考提交严格 Resolved。参考提交 Score 提升但未严格 Resolved 的节点写入预检告警。

审核旧证据时，先对照 `trace/actions.jsonl` 和轨迹，检查命令是否真的执行了测试、测试是否完整、先后两次验证是否覆盖同一断言，以及输入与依赖是否改变。把已确认的机会数、复用、重复执行、相关变化后复验和无法判定数连同事件索引记录在该次运行的 `audit.json`，再重新生成 `evidence.csv`。命令关键词命中数仅供定位。

基准来源和评分规则见 [MemTrace](https://github.com/Homy-Xu/MemTrace)、[DeepSWE](https://github.com/datacurve-ai/deep-swe)、[SWE-EVO](https://github.com/SWE-EVO/SWE-EVO) 和 [SWE-Milestone](https://github.com/DeepCommit-ai/SWE-Milestone)。MemTrace 上游源码及许可证位于本地论文代码目录，部署时单独同步。
