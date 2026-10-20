# AISBench 配置：基于 Harbor 的 Terminal-Bench 4.0 测评
#
# 与 harbor_terminal_bench_3_task.py 的差异（TB 3.0 -> TB 4.0）：
#   1. 数据集：本地 path 指向 v4.0.0 tag 物化出的任务目录（terminal-bench-4/tasks/），
#      66 个任务与 dataset.toml 清单完全一致（tag 快照已核验）。
#      注意：不要使用本目录下无 .git 的 terminal-bench/（滚动 main，与 4.0 tag 有偏差）。
#      任务内容保持官方原样（不要修改任何任务文件，否则镜像内容寻址缓存失效、
#      且与官方 digest 口径不可比）。
#   2. GPU 任务（jax-speedrun-gpu / fp8-rmsnorm-gemm / math-eval-grader）默认
#      不排除：docker 环境下 harbor 对 gpus>0 的任务在环境创建阶段立即抛
#      RuntimeError，trial 按失败计 0 分，报告中注明"因 GPU 类型任务导致失败"。
#      其镜像已由 01_prebuild_pack.sh 临时去除 gpus 声明预构建入包（task.toml
#      已恢复官方原样）。若要改回静默排除，把 exclude_task_names 恢复为任务名
#      列表即可。
#   3. 多容器任务：11 个（3.0 的 erp-procurement-planning 已在 4.0 移除），
#      sidecar 离线处理方式与 3.0 相同（见 tb3-adapt/ 脚本，TASKS_DIR 换成 4.0 的 tasks 目录即可复用）。
#   4. harbor 版本：v4.0.0 tag 内没有 .github/harbor-version pin（0.23.1.dev 是
#      main 分支的 CI 文件，不随 4.0 发布），AISBench/harbor fork 0.21.0 可直接使用；
#      fork 的 VerifierConfig 已支持 4.0 任务用到的 [[verifier.collect]]（5 个任务）
#      和 artifact service 字段。若 oracle 自检暴露解析问题，再考虑 rebase 到 0.23.x。
#   5. 旧约束不变：不兼容 harbor==0.6.1 路线；verifier 全部独立（separate），
#      每任务 env + verifier 两张镜像。
#   6. 模型接入（claude-code + DeepSeek 官方 API 场景）：claude-code 只从环境变量
#      读取密钥与端点（ANTHROPIC_API_KEY / ANTHROPIC_BASE_URL），通过 agent_env
#      注入（见下方 models 块）；api_base 是 terminus-2 的参数，对 claude-code 无效。
import os

from mmengine.config import read_base
from ais_bench.benchmark.tasks.custom_tasks.harbor_task import HarborTask
from ais_bench.benchmark.tasks.base import EmptyTask
from ais_bench.benchmark.summarizers.harbor import HarborSummarizer

with read_base():
    from ais_bench.benchmark.configs.summarizers.example import summarizer

models = [
    dict(
        abbr="claude-code",
        agent_name="claude-code",  # -a/--agent: Agent名称 (terminus-2, claude-code, openhands等)
        model_names=["deepseek-flash"],  # 传给 ANTHROPIC_MODEL，由 DeepSeek 的 Anthropic 兼容端点解析
        agent_kwargs={  # --ak/--agent-kwarg: Agent额外参数
            # 注意：api_base 是 terminus-2 的参数，claude-code 不读取；
            # claude-code 的端点地址由下方 agent_env 的 ANTHROPIC_BASE_URL 控制
            "model_info": {  # 模型token限制和成本信息
                "max_input_tokens": 128000,
                "max_output_tokens": 4096,
                "input_cost_per_token": 0.0,
                "output_cost_per_token": 0.0,
            },
            "llm_call_kwargs": {  # LLM调用参数
                "max_tokens": 4096,  # 最大输出token数
                # "temperature": 0.7,
                # "top_p": 0.9,
                # "top_k": 50,
            },
        },
        agent_env={  # --ae/--agent-env: 传递给agent的环境变量（claude-code 只认环境变量）
            # key 优先从环境变量 DEEPSEEK_API_KEY 读取（避免明文写进配置/git），
            # 未设置时需将下方占位符替换为真实 key
            "ANTHROPIC_API_KEY": os.environ.get("DEEPSEEK_API_KEY", "sk-替换为你的DeepSeek密钥"),
            "ANTHROPIC_BASE_URL": "https://api.deepseek.com/anthropic",  # DeepSeek 的 Anthropic 兼容端点
        },
    )
]

datasets = []

sub_tasks = ["terminal-bench-4"]
for task in sub_tasks:
    datasets.append(
        dict(
            abbr=f'harbor_{task}',
            args=dict(
                n_attempts=1,  # -k/--n-attempts: 每个trial的尝试次数（pass@k 时改为 5）
                timeout_multiplier=1.0,  # --timeout-multiplier: 超时倍数（所有超时乘以此系数）
                agent_timeout_multiplier=None,  # --agent-timeout-multiplier: Agent执行超时倍数（覆盖timeout-multiplier）
                verifier_timeout_multiplier=None,  # --verifier-timeout-multiplier: 验证器超时倍数
                agent_setup_timeout_multiplier=None,  # --agent-setup-timeout-multiplier: Agent设置超时倍数
                environment_build_timeout_multiplier=None,  # --environment-build-timeout-multiplier: 环境构建超时倍数
                debug=False,  # --debug: 启用调试日志
                n_concurrent_trials=5,  # -n/--n-concurrent: 并发运行的trial数量
                quiet=False,  # -q/--quiet: 静默模式
                max_retries=0,  # -r/--max-retries: 最大重试次数
                retry_include_exceptions=None,  # --retry-include: 需要重试的异常类型列表
                retry_exclude_exceptions=[  # --retry-exclude: 不需要重试的异常类型列表
                    # "AgentTimeoutError",
                    # "VerifierTimeoutError",
                    # "RewardFileNotFoundError",
                    "RewardFileEmptyError",
                    "VerifierOutputParseError",
                ],
                environment_type="docker",  # -e/--env: 环境类型 (docker, daytona, e2b, modal)
                environment_force_build=False,  # --force-build/--no-force-build: 是否强制重建环境
                environment_delete=False,  # --delete/--no-delete: 完成后是否删除环境
                path="/path/to/terminal-bench-4/tasks/",  # -p/--path: 本地数据集路径（v4.0.0 tag 物化出的 tasks 目录，需按实际路径修改）
                dataset_name_version=None,  # -d/--dataset: 远程数据集名称@版本（联网时可用 terminal-bench/terminal-bench@4.0）
                task_names=None,  # --include-task-name: 包含的任务名称（支持glob模式）
                exclude_task_names=None,  # --exclude-task-name: 排除的任务名称。
                # TB 4.0 的 3 个 H100 任务（jax-speedrun-gpu / fp8-rmsnorm-gemm /
                # math-eval-grader）默认不排除：docker 环境下立即失败（RuntimeError），
                # 计 0 分并在报告注明 GPU 原因；如需静默排除改回任务名列表。
                n_tasks=None,  # --n-tasks: 最大任务数量（快速跑通流程可设置几条）
                disable_verification=False,  # --disable-verification: 禁用验证器
                verifier_env=None,  # --ve/--verifier-env: 验证器环境变量
                yes=True,  # -y/--yes: 自动确认环境变量提示
                env_file=None,  # --env-file: .env文件路径
            ),
        )
    )

infer = dict(
    runner=dict(
        task=dict(type=EmptyTask)
    ),
)

eval = dict(
    runner=dict(
        task=dict(type=HarborTask)
    ),
)

summarizer = dict(
    attr="accuracy",
    type=HarborSummarizer,
)
