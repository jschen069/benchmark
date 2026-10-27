"""CPU-only contract tests: real FFmpeg, real subprocess and data audits."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from ais_bench.benchmark.vabench import DIMENSIONS, MLLM_DIMENSIONS
from ais_bench.benchmark.vabench.data import (
    PROTOCOL_PATH, catalog, digest, media_tool, prepare, read_json, verify_prepared, write_json)
from ais_bench.benchmark.vabench.engine import run, successful
from ais_bench.benchmark.vabench.results import (
    build_report, compare, normalize_records, summarize_dimension)
from ais_bench.benchmark.vabench.worker import has_backend_error


SOURCE = Path(__file__).resolve().parents[4] / 'VABench'


def test_pinned_catalog():
    if not SOURCE.exists():
        pytest.skip('Local official checkout is not available')
    standard, mini = catalog(SOURCE), catalog(SOURCE, 'mini')
    assert len(standard) == 778 and len(mini) == 10
    assert len({e['idx'] for e in standard}) == 778
    assert {e['idx'] for e in mini} <= {e['idx'] for e in standard}
    assert len({e['category'] for e in mini}) == 10
    assert {'easy_human_language', 'hard_human_language', 'hard_Exact_Time_Synchronization'} <= {
        e['category'] for e in mini}


def test_pinned_data_rejects_drift(tmp_path):
    root = tmp_path / 'source'
    root.mkdir()
    with pytest.raises(ValueError, match='differs from pinned'):
        catalog(root)


def test_explicit_single_sample_smoke_catalog():
    if not SOURCE.exists():
        pytest.skip('Local official checkout is not available')
    entries = catalog(SOURCE, 'smoke', sample_id='kitten curled up meowing')
    assert len(entries) == 1 and entries[0]['idx'] == 'kitten curled up meowing'
    assert len(catalog(SOURCE, 'mini')) == 10
    assert len(catalog(SOURCE, 'standard')) == 778
    with pytest.raises(ValueError, match='requires one sample_id'):
        catalog(SOURCE, 'smoke')
    with pytest.raises(ValueError, match='requires one sample_id'):
        catalog(SOURCE, 'smoke', sample_id='not-an-official-prompt')
    with pytest.raises(ValueError, match='only supported'):
        catalog(SOURCE, 'mini', sample_id='kitten curled up meowing')


@pytest.fixture
def entry():
    return dict(idx='中文 sample with spaces', category='scene', metadata=dict(
        prompt_en='A test tone with a visual scene.', prompt_audio='A tone.',
        prompt_vision='A color image.', dimension=list(DIMENSIONS),
        auxiliary_info_audio=['Is a tone audible?'], auxiliary_info_vision=['Is color visible?']))


@pytest.fixture
def video(tmp_path, entry):
    if not shutil.which(media_tool('ffmpeg')) or not shutil.which(media_tool('ffprobe')):
        pytest.skip('FFmpeg and ffprobe are required for media tests')
    folder = tmp_path / 'videos'
    folder.mkdir()
    file = folder / (entry['idx'] + '.mp4')
    subprocess.run([
        media_tool('ffmpeg'), '-v', 'error', '-y', '-f', 'lavfi', '-i',
        'color=c=blue:s=64x64:r=24:d=0.3', '-f', 'lavfi', '-i',
        'sine=frequency=440:sample_rate=48000:duration=0.3',
        '-c:v', 'mpeg4', '-c:a', 'aac', '-shortest', str(file)], check=True)
    return file


def test_real_media_preparation(tmp_path, entry, video):
    out = tmp_path / 'prepared'
    manifest = prepare([entry], video.parent, out)
    assert manifest['subset_size'] == 1
    assert manifest['records'][0]['short_video'] is True
    verified, grouped = verify_prepared(out)
    assert verified['input_signature'] == manifest['input_signature']
    assert grouped['scene'][0]['prompt_en'] == entry['metadata']['prompt_en']
    wave = Path(manifest['records'][0]['audio_path'])
    import wave as wave_module
    with wave_module.open(str(wave)) as audio:
        assert audio.getframerate() == 48000
        assert audio.getsampwidth() == 2
    with pytest.raises(FileExistsError):
        prepare([entry], video.parent, out)
    wave.write_bytes(b'changed')
    with pytest.raises(ValueError, match='Input changed'):
        verify_prepared(out)


def test_smoke_cli_labels_single_sample(tmp_path, entry, video, monkeypatch):
    import ais_bench.benchmark.vabench.cli as cli
    monkeypatch.setattr(cli, 'catalog', lambda *a, **kw: [entry])
    out = tmp_path / 'smoke'
    assert cli.main(['prepare', '--source-root', 'fixture', '--subset', 'smoke',
                     '--sample-id', entry['idx'], '--videos', str(video.parent),
                     '--output', str(out)]) == 0
    manifest, grouped = verify_prepared(out)
    assert manifest['evaluation_scope'] == 'smoke'
    assert manifest['subset_size'] == 1
    report = build_report(manifest, {}, {})
    assert report['evaluation_scope'] == 'smoke'
    assert 'not Mini10/778' in report['scope_note']
    with pytest.raises(ValueError, match='exactly one'):
        prepare([entry, entry], video.parent, tmp_path / 'invalid', subset='smoke')


def test_missing_ambiguous_and_silent_input(tmp_path, entry, video):
    with pytest.raises(ValueError, match='Missing MP4'):
        prepare([dict(entry, idx='missing')], video.parent, tmp_path / 'out1')
    extra = video.parent / 'duplicate'
    extra.mkdir()
    shutil.copyfile(video, extra / video.name)
    with pytest.raises(ValueError, match='ambiguous MP4'):
        prepare([entry], video.parent, tmp_path / 'out2')
    (extra / video.name).unlink()
    silent = video.parent / 'silent.mp4'
    subprocess.run([media_tool('ffmpeg'), '-v', 'error', '-y', '-i', str(video),
                    '-an', '-c:v', 'copy', str(silent)], check=True)
    with pytest.raises(ValueError, match='must contain video and audio'):
        prepare([dict(entry, idx='silent')], video.parent, tmp_path / 'out3')


def _report(score=0.0, status='ok', dim='second_clap'):
    records = [dict(idx='x', category='scene', status=status, score=score, raw={})]
    return build_report(dict(input_signature='same', records=[{'idx': 'x'}]),
                        {dim: {'records': records}},
                        {k: 'same' for k in ('protocol_sha256', 'source_signature', 'integration_signature',
                                             'weights_signature', 'scoring_signature')})


def test_comparison_zero_negative_and_threshold():
    reference = _report(0)
    assert compare(reference, _report(0), ['second_clap'])['passed']
    assert not compare(reference, _report(0.01), ['second_clap'])['passed']
    assert compare(_report(-0.5), _report(-0.502), ['second_clap'])['passed']
    # Boundary uses strict < 1%, rather than <= 1%.
    assert not compare(_report(100), _report(101), ['second_clap'])['passed']
    assert not compare(_report(), _report())['passed']  # fifteen dims required by default
    with pytest.raises(ValueError, match='Invalid tolerance'):
        compare(reference, reference, relative=float('inf'))


@pytest.mark.parametrize('mutation', ['missing', 'failed', 'weights', 'media', 'duplicate', 'nan'])
def test_alignment_cannot_pass_invalid_runs(mutation):
    left, right = _report(0.5), _report(0.5)
    if mutation == 'missing':
        right['details']['second_clap']['records'][0]['status'] = 'missing'
    elif mutation == 'failed':
        for report in (left, right):
            report['details']['second_clap']['records'][0]['status'] = 'backend_error'
    elif mutation == 'weights':
        right['provenance']['weights_signature'] = 'different'
    elif mutation == 'media':
        right['input_signature'] = 'different'
    elif mutation == 'duplicate':
        right['details']['second_clap']['records'] *= 2
    elif mutation == 'nan':
        right['details']['second_clap']['records'][0]['score'] = float('nan')
    assert not compare(left, right, ['second_clap'])['passed']


def test_statuses_and_weighted_mean():
    expected = [dict(idx='x', category='a'), dict(idx='y', category='b')]
    rows = normalize_records('second_lsa', [{'video_path': 'x.mp4', 'video_results': 0}], expected)
    assert [r['status'] for r in rows] == ['ok', 'missing']
    summary = summarize_dimension(rows)
    assert summary['mean'] == 0 and summary['missing'] == 1
    rows = [dict(idx=str(i), category=cat, status='ok', score=score)
            for i, (cat, score) in enumerate([('a', 0), ('a', 0), ('b', 3)])]
    assert summarize_dimension(rows)['mean'] == 1  # not (0+3)/2
    assert normalize_records('second_clap', [{'video_path': 'x.mp4', 'video_results': 0}],
                             expected, backend_errors=True)[0]['status'] == 'backend_error'
    assert has_backend_error('Error processing clip: failed')
    assert has_backend_error('Traceback (most recent call last):')
    assert not has_backend_error('Evaluation completed normally')
    assert not has_backend_error('人脸检测失败或评估失败: example.mp4')


def test_explicit_cpu_onnx_provider_and_observation(monkeypatch):
    from types import ModuleType
    from ais_bench.benchmark.vabench.worker import instrument_models
    class Session:
        def __init__(self):
            self.providers = ['CUDAExecutionProvider']
        def set_providers(self, providers):
            self.providers = providers
        def get_providers(self):
            return self.providers
    class Scorer:
        def __init__(self):
            self.session = Session()
    module = ModuleType('vabench.first_dnsmos')
    module._load_dns_mos_model = lambda: Scorer()
    observations = []
    instrument_models(module, observations, 'cpu')
    scorer = module._load_dns_mos_model()
    assert scorer.session.providers == ['CPUExecutionProvider']
    assert observations[0]['observed_onnx_providers'] == ['CPUExecutionProvider']


def test_lsa_missing_sets_must_match_and_not_be_empty():
    left = _report(2.0, dim='second_lsa')
    left['details']['second_lsa']['records'].append(dict(idx='y', category='scene',
                                                       status='missing', score=None))
    left['sample_ids'] = ['x', 'y']
    assert compare(left, copy.deepcopy(left), ['second_lsa'])['passed']
    right = copy.deepcopy(left)
    right['details']['second_lsa']['records'][1]['status'] = 'ok'
    right['details']['second_lsa']['records'][1]['score'] = 2.0
    assert not compare(left, right, ['second_lsa'])['passed']


def test_real_worker_subprocess_and_mllm_grouping(tmp_path, entry, video, monkeypatch):
    """A fake scorer tests process/protocol plumbing, not model accuracy."""
    import ais_bench.benchmark.vabench.engine as engine
    prepared = tmp_path / 'prepared'
    prepare([entry], video.parent, prepared, subset='smoke')
    source = tmp_path / 'fake_source'
    package = source / 'vabench'
    package.mkdir(parents=True)
    (source / 'torch.py').write_text('def device(x): return x\n', encoding='utf-8')
    (package / '__init__.py').write_text('', encoding='utf-8')
    (package / 'omni_runtime.py').write_text(
        'def release_omni(): pass\n'
        'def get_omni_model_and_processor(*a, **kw): raise RuntimeError("unused")\n', encoding='utf-8')
    dims = ['third_alignment', 'fourth_qa_audio']
    for dim in dims:
        (package / f'{dim}.py').write_text(
            'import json\n'
            f'def compute_{dim}(file, device, submodules, **kw):\n'
            '    items = json.load(open(file, encoding="utf-8"))\n'
            '    return None, [{"video_path": x["video_list"][0], "video_results": 3.0} for x in items]\n',
            encoding='utf-8')
    cache = tmp_path / 'cache'
    checkpoint = cache / 'Qwen2.5_omni_7B'
    checkpoint.mkdir(parents=True)
    (checkpoint / 'dummy').write_text('test fixture, not a model')
    monkeypatch.setattr(engine, 'catalog', lambda *args: [entry])
    report = run(prepared, source, cache, tmp_path / 'run', dimensions=dims,
                 device='cpu', python=sys.executable)
    assert successful(report)
    assert report['evaluation_scope'] == 'smoke'
    assert report['metrics']['third_alignment']['mean'] == 3
    assert len(list((tmp_path / 'run').glob('job_*'))) == 1
    request = read_json(tmp_path / 'run/job_00/request.json')
    assert request['dimensions'] == dims
    assert compare(report, report, dims)['passed']

    # Exercise the real upstream VABench orchestration class, with the same
    # fake compute_* scorers. Only model inference is substituted.
    if SOURCE.exists():
        (package / '__init__.py').write_text(
            (SOURCE / 'vabench/__init__.py').read_text(encoding='utf-8'), encoding='utf-8')
        (source / 'torch.py').write_text(
            'def device(x): return x\n'
            'class cuda:\n'
            '    @staticmethod\n'
            '    def is_available(): return False\n', encoding='utf-8')
        (package / 'utils.py').write_text(
            'import json\n'
            'def init_submodules(*a, **kw): return {}\n'
            'def load_json(p):\n'
            '    with open(p, encoding="utf-8") as f: return json.load(f)\n'
            'def save_json(obj, p):\n'
            '    with open(p, "w", encoding="utf-8") as f: json.dump(obj, f)\n', encoding='utf-8')
        mapping_dir = source / 'mapping'
        mapping_dir.mkdir()
        (mapping_dir / 'final_idx_to_prompt.csv').write_text(
            'idx\tprompt\n' + entry['idx'] + '\t' + entry['metadata']['prompt_en'] + '\n',
            encoding='utf-8')
        # The source signature changed to install the real orchestrator;
        # recompute both modes against this identical source snapshot.
        adapter = run(prepared, source, cache, tmp_path / 'adapter_real_class',
                      dimensions=dims, device='cpu')
        official = run(prepared, source, cache, tmp_path / 'official_real_class',
                       dimensions=dims, device='cpu', mode='official')
        assert successful(official)
        assert compare(official, adapter, dims)['passed']
