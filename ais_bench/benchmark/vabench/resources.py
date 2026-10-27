"""Explicit local resource checks; never download weights during scoring."""
from pathlib import Path

from .data import digest, object_digest


WEIGHTS = {
    'first_dnsmos': {'dns_mos_primary': 'dns_mos/sig_bak_ovr.onnx',
                     'dns_mos_p808': 'dns_mos/model_v8.onnx'},
    'first_nisqa': {'nisqa_model_path': 'nisqa/nisqa_mos_only.tar'},
    'first_audiobox': {'first_audiobox_model_path': 'audiobox_aesthetics/checkpoint.pt'},
    'second_viclip': {'second_viclip_model_name': 'viclip/ViCLIP-L_InternVid-FLT-10M.pth'},
    'second_clap': {'second_clap_model_name': 'clap/laion__clap-htsat-unfused'},
    'second_imagebind': {'second_imagebind_model_name': 'imagebind/imagebind_huge.pth'},
    'second_lsa': {'syncnet2_model_path': 'syncnet2/syncnet_v2.model'},
    'second_desync': {'cfg_path': 'synchformer/configs/cfg-24-01-04T16-39-21.yaml',
                      'ckpt_path': 'synchformer/checkpoints/24-01-04T16-39-21.pt'},
}


def submodules(dimension, cache):
    return {key: str(Path(cache).resolve() / path)
            for key, path in WEIGHTS.get(dimension, {}).items()}


def required_paths(dimensions, cache):
    paths = {p for d in dimensions for p in submodules(d, cache).values()}
    if any(d.startswith(('third_', 'fourth_')) for d in dimensions):
        paths.add(str(Path(cache).resolve() / 'Qwen2.5_omni_7B'))
    if 'second_viclip' in dimensions:
        paths.add(str(Path(cache).resolve() / 'viclip/bpe_simple_vocab_16e6.txt.gz'))
    if 'second_lsa' in dimensions:
        paths.add(str(Path(cache).resolve() / 'checkpoints/auxiliary/sfd_face.pth'))
        paths.add(str(Path(cache).resolve() / 'checkpoints/auxiliary/models/buffalo_l'))
    return sorted(paths)


def source_signature(source_root):
    root = Path(source_root).resolve()
    # Path.rglob does not traverse directory symlinks on all Python versions.
    # Synchformer is commonly linked from the official cache, so enumerate it
    # as an explicit root as well and deduplicate paths.
    files = sorted(set([*root.glob('vabench/*.py'), *root.glob('third_party/**/*.py'),
                        *(root / 'third_party/Synchformer').glob('**/*.py')]))
    if not files:
        raise FileNotFoundError(f'Official source missing: {root}')
    return object_digest({p.relative_to(root).as_posix(): digest(p) for p in files})


def weight_fingerprints(dimensions, cache):
    """Hash contents, not paths or mtimes, for cross-server comparisons."""
    cache = Path(cache).resolve()
    found, missing = {}, []
    for name in required_paths(dimensions, cache):
        path = Path(name)
        if not path.exists():
            missing.append(name)
            continue
        files = sorted(p for p in path.rglob('*') if p.is_file()
                       and '.cache' not in p.parts and p.suffix not in ('.lock', '.incomplete')) if path.is_dir() else [path]
        if not files:
            missing.append(name)
        for file in files:
            found[file.relative_to(cache).as_posix()] = digest(file)
    return found, missing


def preflight(source_root, cache, dimensions):
    root = Path(source_root).resolve()
    missing = [p for p in required_paths(dimensions, cache) if not Path(p).exists()]
    if 'first_dnsmos' in dimensions:
        file = root / 'third_party/DNS-Challenge/DNSMOS/dnsmos_local.py'
        if not file.is_file():
            missing.append(str(file))
    if 'second_desync' in dimensions:
        for file in ('scripts/train_utils.py', 'dataset/transforms.py'):
            path = root / 'third_party/Synchformer' / file
            if not path.is_file():
                missing.append(str(path))
    for dimension in dimensions:
        path = root / 'vabench' / f'{dimension}.py'
        if not path.is_file():
            missing.append(str(path))
    return missing
