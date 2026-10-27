# VABench T2AV 评测接入与 A3 运行指南

本接入评测已生成的音视频。固定版本为 `tanABCC/VABench@12b9617a900a9eca1691da6b7115ce3d58f53a16`，默认使用 778 个可映射 T2AV Prompt，另有固定 10 条 Mini。不提供模型生成调用，不将 10 条 Mini 的分数当作完整集的估计。

当前交付包含轻量测试结果。GPU 15 维真实模型分数、GPU 官方 <1% 对齐及 A3/NPU 算子兼容性均待服务器实测。`transfer_to_npu` 是迁移入口，不是 A3 认证结果。

单条视频可以先做独立冒烟测试，使用 `prepare --subset smoke --sample-id '官方idx'`，然后 `run --dimensions second_clap`。该模式从固定778题中精确选择1题，报告标记 `evaluation_scope=smoke`，不改变Mini10/778，也不代表完整评测通过。仅有4秒小猫视频时，可先验证媒体处理与CLAP链路；同步类指标留待更合适时长样本。

```bash
python -m ais_bench.benchmark.vabench.cli prepare --source-root /data/VABench --subset smoke --sample-id 'kitten curled up meowing' --videos /data/gererated --output /data/prepared_kitten_smoke
python -m ais_bench.benchmark.vabench.cli run --source-root /data/VABench --cache-dir /data/vabench-cache --prepared /data/prepared_kitten_smoke --device npu:0 --dimensions second_clap --output /data/runs/kitten_smoke_clap
```

AISBench主命令的对应配置为 `ais_bench/configs/vabench_examples/eval_vabench_smoke.py`，填写服务器的三个路径后运行 `ais_bench ais_bench/configs/vabench_examples/eval_vabench_smoke.py -m eval`。

## 1. 数据和算法的来源

源码目录由 `--source-root` 指定，必须包含 `vabench/`、`mapping/`、`json_files/` 和第三方依赖。协议内保存 39 个原始数据文件的内容指纹；JSON/mapping 改变会报错，不进行模糊匹配。

原始 `json_text` 为 848 条，只有 778 条与 mapping 精确匹配；另外 70 条不自动补名。JSON 里每条均声明全部维度，本实现保留实际代码协议，不擅自套用论文的语音/虚拟场景过滤假设。若需要另一套语义子集，应单独命名和验收。

Mini 选择依据在 `ais_bench/benchmark/vabench/protocol.json` 的 `mini` 中。选取了新闻主播、快速说唱、小猫、钢琴、自然夜景、咖啡馆、玻璃弹珠、微波炉精确同步、复杂声音场景及虚拟世界，共 10 条，包含明显的语音/唇同步候选场景。是否实际有人脸及指标是否有效仍取决于生成视频。

## 2. A3 环境准备

以下为 Linux shell 示例。先使用本机 A3 驱动/CANN 所匹配的 PyTorch、torchvision、torchaudio 和 torch_npu 镜像/环境。不要直接在 A3 安装官方 `requirement.txt`，其中含 NVIDIA CUDA 组件及固定 GPU torch。确认版本组合的依据是 [Ascend 官方安装说明](https://github.com/Ascend/pytorch/blob/master/README.md)。

```bash
npu-smi info
python -c "import torch, torch_npu; print(torch.__version__, torch_npu.__version__); print(torch.npu.is_available(), torch.npu.device_count())"
ffmpeg -version
ffprobe -version
```

FFmpeg 必须是完整版本，支持 MP4 解码和 `pcm_s16le` 音频输出；官方 LSA 还会使用 H.264 编码。某些桌面软件附带的裁剪版缺少音频编码器，不能用于评测。预处理可通过 `VABENCH_FFMPEG` / `VABENCH_FFPROBE` 指定完整可执行文件；模型后端调用的 `ffmpeg` 仍需在 PATH 指向相同的完整版本。

AISBench 主环境与官方后端可以分开，后者推荐 Python 3.11。AISBench LocalRunner 本身也要能识别 torch/torch_npu。后端通过 `--python` 或配置 `BACKEND_PYTHON` 指定，不需要在模型环境安装完整 AISBench；子进程会加入本接入包的 PYTHONPATH。

在匹配 A3 软件栈的模型环境中安装额外依赖，建议先把既有 torch 三件套和 torch_npu 版本写入 pip constraints，再安装 `requirements/vabench.txt`。这是依赖起点，Synchformer 的附加依赖仍应按其随附说明补齐；安装后运行 `pip check`。不要为满足其他包自动替换已验证的 torch/CANN 组合。

```bash
# 在 AISBench 源码根目录；切换到已匹配 CANN 的模型环境。
python -c "import importlib.metadata as m; from pathlib import Path; names=['torch','torchvision','torchaudio','torch-npu']; Path('/tmp/vabench-torch-constraints.txt').write_text('\n'.join(n+'=='+m.version(n) for n in names)+'\n')"
python -m pip install -r requirements/vabench.txt -c /tmp/vabench-torch-constraints.txt
python -m pip install git+https://github.com/openai/CLIP.git -c /tmp/vabench-torch-constraints.txt
python -m pip install git+https://github.com/facebookresearch/ImageBind.git -c /tmp/vabench-torch-constraints.txt
python -m pip check
```

若环境已有 onnxruntime-gpu，应避免与 onnxruntime 同时安装。A3 推荐 CPU ONNX；DNSMOS 默认显式 CPU，LSA 中 InsightFace 在 NPU/CPU 模式也显式使用 CPU provider，SyncNet/S3FD 的 PyTorch 模型仍按请求设备运行。

## 3. 准备官方源码及本地缓存

已有本地源码可直接复制到服务器。若重新获取，固定提交：

```bash
git clone https://github.com/tanABCC/VABench.git /data/VABench
git -C /data/VABench checkout 12b9617a900a9eca1691da6b7115ce3d58f53a16
export VABENCH_ROOT=/data/VABench
export VABENCH_CACHE_DIR=/data/vabench-cache
```

提前下载 [官方模型缓存](https://huggingface.co/datasets/tanABBCC/VABENCH_CACHE_DIR)，不在评分期间自动下载。可在联网准备机使用下列命令，再复制到服务器。此操作下载数十 GB 的权重，应预留空间。

```bash
hf download tanABBCC/VABENCH_CACHE_DIR --repo-type dataset --revision aa0c3da973d86f2994e9c8860f28785ce0cc6152 --local-dir /data/vabench-cache
```

缓存的关键路径如下（不是只有 Qwen 权重）：

```text
vabench-cache/
  dns_mos/{sig_bak_ovr.onnx,model_v8.onnx}
  nisqa/nisqa_mos_only.tar
  audiobox_aesthetics/checkpoint.pt
  viclip/{ViCLIP-L_InternVid-FLT-10M.pth,bpe_simple_vocab_16e6.txt.gz}
  clap/laion__clap-htsat-unfused/...
  imagebind/imagebind_huge.pth
  synchformer/configs/cfg-24-01-04T16-39-21.yaml
  synchformer/checkpoints/24-01-04T16-39-21.pt
  synchformer/Synchformer/...                       # 第三方源码
  syncnet2/syncnet_v2.model
  Qwen2.5_omni_7B/...
  checkpoints/auxiliary/sfd_face.pth                # 可能需要额外准备
  checkpoints/auxiliary/models/buffalo_l/*.onnx     # InsightFace 检测/关键点
```

还需要两处第三方源码。Synchformer 优先使用缓存随附版本，避免仓库最新版与 checkpoint 不匹配；DNSMOS 脚本来自微软官方仓库的固定提交。

```bash
ln -s /data/vabench-cache/synchformer/Synchformer /data/VABench/third_party/Synchformer
mkdir -p /data/VABench/third_party/DNS-Challenge/DNSMOS
curl -fL https://raw.githubusercontent.com/microsoft/DNS-Challenge/591184a9fcb2cbdec02520fed81a32bbbf9d73ff/DNSMOS/dnsmos_local.py -o /data/VABench/third_party/DNS-Challenge/DNSMOS/dnsmos_local.py
```

LSA 的额外人脸模型不一定在缓存中。`sfd_face.pth` 可从 [LatentSync 官方资源](https://huggingface.co/ByteDance/LatentSync-1.5/tree/main/auxiliary) 准备；`buffalo_l` 从 [InsightFace 官方模型资源](https://github.com/deepinsight/insightface/tree/master/python-package#model-zoo) 准备，解压到上述目录。GPU 参考与 NPU 候选必须使用相同文件，报告会记录其内容 SHA256。准备步骤增加依赖文件但不改写官方评分算法。

运行预检，保留结果：

```bash
python -m ais_bench.benchmark.vabench.cli doctor --source-root /data/VABench --cache-dir /data/vabench-cache > /data/vabench-doctor.json
```

`doctor` 列出缺失模块和资源、torch 设备状态、npu-smi。最终兼容性仍以实际维度执行结果为准。

## 4. 导出题目、准备 MP4、提取 WAV

在 AISBench 源码根目录执行轻量 CLI，不需要加载评分模型：

```bash
python -m ais_bench.benchmark.vabench.cli catalog --source-root /data/VABench --subset mini --output /data/mini_prompts.json
python -m ais_bench.benchmark.vabench.cli catalog --source-root /data/VABench --subset standard --output /data/t2av_778_prompts.json
```

清单 `entries[].metadata.prompt_en` 是模型生成提示词，`entries[].idx` 是结果文件基名。按选定清单生成 MP4 后，放在同一根目录下，可平铺或分场景子目录：

```text
/data/generated/
  news anchor delivers evening news.mp4
  kitten curled up meowing.mp4
  ...
```

每个 idx 必须恰好找到一个 MP4，必须含音轨。工具不自动把无声视频补成静音。准备目录应为新目录；已完成目录不可覆盖。更改视频后使用新目录重新准备。

```bash
python -m ais_bench.benchmark.vabench.cli prepare --source-root /data/VABench --subset mini --videos /data/generated --output /data/prepared_mini
```

正式集改为 `--subset standard` 并使用另一个目录。准备阶段保留原 MP4 路径，生成未裁剪、未主动重采样/混声的 PCM16 WAV；各专家继续按官方代码做自己的预处理。小于 5 秒的样本在 manifest 标记，可能影响同步指标，应在模型产物规格中检查。

`manifest.json` 包含媒体 SHA256、协议指纹与题目映射；运行前重查文件内容。MP4/WAV 及 JSON 路径为绝对路径，移动到另一台服务器后应在当地重新 prepare，相同内容仍会得到可比较的输入签名。请保留原 MP4，准备目录不是独立的媒体副本。

## 5. 运行 A3 评测

先跑专家单维以定位环境问题，再跑 Mini 15 维。NPU 模式在每个独立后端进程启用 `torch_npu.contrib.transfer_to_npu`；MLLM 设定明确的 NPU device_map，不会在报错时重试 CPU。

```bash
python -m ais_bench.benchmark.vabench.cli run --source-root /data/VABench --cache-dir /data/vabench-cache --prepared /data/prepared_mini --device npu:0 --dimensions second_clap --output /data/runs/npu_clap --options ais_bench/configs/vabench_examples/options.json
python -m ais_bench.benchmark.vabench.cli run --source-root /data/VABench --cache-dir /data/vabench-cache --prepared /data/prepared_mini --device npu:0 --output /data/runs/npu_mini --options ais_bench/configs/vabench_examples/options.json
```

若模型环境与 AISBench 不同，添加 `--python /opt/venvs/vabench/bin/python`。`options.json` 还支持 `dim_params`（官方维度参数）、`device_overrides`（显式专家设备）、`timeout`（每后端进程秒数）。如确认某专家需 CPU，在 options 中配置，例如 `"first_nisqa": "cpu"`。不能为七个 MLLM 分别配置不同设备；它们在一个后端进程中共享 Omni。NPU Omni 当前显式映射单卡；多卡 NPU 分片未实现，若显存不足将记录错误，不能宣称已支持。

与 AISBench 主流程集成：编辑 `ais_bench/configs/vabench_examples/eval_vabench.py` 的 `PREPARED_DIR`、`SOURCE_ROOT`、`CACHE_DIR`、`DEVICE`、`BACKEND_PYTHON`。`NUM_DEVICES` 控制 LocalRunner 预留卡数，NPU 单卡模式设 1。GPU 参考默认保留 Omni 的 device_map=auto，多卡时按需要提高预留卡数。默认一个 AISBench 任务、内部专家子进程串行，MLLM 七维合并，避免重复并发加载大模型。

```bash
ais_bench ais_bench/configs/vabench_examples/eval_vabench.py -m eval
```

LocalRunner 在启动后端前设置设备可见性。多模型横比可以增加 models 中不同 abbr，并为各模型配置其对应 prepared 目录的 model_dataset_combinations；不能将同一个视频目录错误地当成多个模型生成结果。

## 6. 官方基线和 <1% 对齐

需要同一批 MP4、相同生成 WAV、官方源码/第三方依赖、权重文件与评分参数。当前没有 GPU 服务器，本步骤由后续验证执行。

```bash
# 在 GPU 环境；与 A3 使用相同的 options 和模型内容。
python -m ais_bench.benchmark.vabench.cli run --source-root /data/VABench --cache-dir /data/vabench-cache --prepared /data/prepared_mini --device cuda:0 --mode official --output /data/runs/gpu_official --options ais_bench/configs/vabench_examples/options.json
python -m ais_bench.benchmark.vabench.cli run --source-root /data/VABench --cache-dir /data/vabench-cache --prepared /data/prepared_mini --device cuda:0 --mode adapter --output /data/runs/gpu_aisbench --options ais_bench/configs/vabench_examples/options.json
python -m ais_bench.benchmark.vabench.cli compare --reference /data/runs/gpu_official/report.json --candidate /data/runs/gpu_aisbench/report.json --output /data/runs/gpu_alignment.json
```

`official` 模式执行原 VABench 类，修正错误 mapping 路径，并提供同一组离线资源；这是“带工程运行修正的官方基线”，不是声称未修复的官方 CLI 原样可用。使用相同的设备放置修正以便比较。按场景和维度运行，官方类会释放 Omni，因此基线可能明显慢于适配模式。

对齐公式为 `abs(mean_aisbench - mean_official) / abs(mean_official) < 0.01`。默认当 `abs(mean_official) <= 1e-6` 时改用 `abs(diff) <= 1e-6`；可通过 `--near-zero` / `--absolute` 明确调整，报告记录最终阈值。官方没有统一总分，本工具不平均十五个不同量纲的维度来验收。

比较工具还核对样本 ID/状态、输入哈希、协议、官方源码、接入代码、权重和评分参数指纹。15 维缺任一维则不能通过全量验收。LSA 允许双方相同缺失集合，但至少需要一个有效样本；其他维度缺失不通过。后端报错后返回的默认 0/1 等数字仍保存在 raw 里，但标记失败，不进入有效均值；另存 `official_numeric_mean` 仅供审计。相同的失败不能让“对齐”通过。

GPU 对齐完成后，再将 candidate 指向 NPU 报告作跨硬件检查。单维排障可为 compare 指定 `--dimensions second_clap`，但单维通过不代表 15 维通过。正式验收应在 778 集上执行，Mini 用于快速验证。

## 7. 输出、失败排查与结果回收

```text
run/
  report.json             # 原始记录、15维/场景均值、缺失失败、设备观测、版本和哈希
  input_manifest.json     # 本次引用的输入记录
  metrics.csv             # 原始量纲指标，不是百分制总分
  scenes.csv              # 场景切片
  samples.csv             # 逐题状态和分数
  partial_report.json     # 每个完成任务后的阶段结果
  job_00/request.json
  job_00/response.json
  job_00/process.log
  job_00/维度/场景/backend.log
```

`requested_device` 是配置值；实际证据见 `model_factory_observations`、`device_observations` 与 `observed_omni_devices`。未观测到设备时不能将请求值冒充实际设备。InsightFace 与 DNSMOS 单独记录 ONNX provider。后端错误会尽量继续其他维度并保存报告，CLI 最终返回非零；AISBench 也保留报告后标记任务错误。

每个维度/场景有独立 cwd，避免官方 LSA 的 `temp/` 与 `detect_results/` 相互覆盖。没有跨任务断点续跑；重试请使用新输出目录，避免旧结果与新配置混合。若作业被强杀，partial_report/partial.json 用于定位，不能当完整报告。

后续 A3 验证请保留 doctor JSON、完整 run 目录、`pip freeze`、设备可见性设置和驱动/CANN 版本。优先检查：

- 缺文件：根据 doctor/后端报告准备对应源码、权重、tokenizer 与人脸模型。
- 设备不支持：确认 CANN/torch_npu 组合，查看 process.log 中算子错误；仅对已确认的专家增加 CPU override。
- Omni OOM：先确认视频像素/时长与显存。修改 video_max_pixels 会影响评分，必须对 GPU 官方基线使用同一设置并重新对齐。
- LSA 缺结果：结合 backend.log 核实无人脸、无可用轨迹或执行错误，不能自动补 0。

## 8. 轻量测试

在 AISBench 开发环境执行，无需评分模型或 torch：

```bash
python -m pytest tests/UT/vabench -q
```

包含真实 FFmpeg 预处理、真实子进程协议、真实官方 VABench 编排类（以模拟评分模型替代推理）、配置加载和分区、汇总及误差校验。轻量测试证明集成逻辑，不证明模型分数正确或 A3 算子均可用。

参考：[官方 VABench](https://github.com/tanABCC/VABench)、[Transformers 4.57 Omni](https://huggingface.co/docs/transformers/v4.57.0/en/model_doc/qwen2_5_omni)、[Ascend 迁移说明](https://www.hiascend.com/developer/techArticles/20260227-1?envFlag=1)。
