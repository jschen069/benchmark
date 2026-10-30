"""Fixed display protocol and incomplete-run semantics, without model inference."""
import copy
import csv

import pytest

from ais_bench.benchmark.vabench import DIMENSIONS
from ais_bench.benchmark.vabench.data import digest, read_json, write_json
from ais_bench.benchmark.vabench.engine import successful
from ais_bench.benchmark.vabench.results import (
    build_report, export_csv, format_summary, normalize_records, refresh_report)
from ais_bench.benchmark.vabench.scoring import RULES, normalize_score


def fixture_inputs():
    manifest = dict(input_signature='synthetic_fixture', evaluation_scope='mini',
                    records=[dict(idx='a', category='one'), dict(idx='b', category='two')])
    details = {}
    for dim in DIMENSIONS:
        low, high, _ = RULES[dim]
        score = (low + high) / 2
        raw = [dict(video_path=r['idx'] + '.mp4', video_results=score,
                    reasoning='Synthetic test fixture, not model output') for r in manifest['records']]
        details[dim] = dict(records=normalize_records(dim, raw, manifest['records']), requested_device='fixture')
    return manifest, details


@pytest.mark.parametrize('dimension', DIMENSIONS)
def test_fixed_endpoints_and_midpoints(dimension):
    low, high, reverse = RULES[dimension]
    assert normalize_score(dimension, low)['normalized_score'] == (100 if reverse else 0)
    assert normalize_score(dimension, high)['normalized_score'] == (0 if reverse else 100)
    assert normalize_score(dimension, (low + high) / 2)['normalized_score'] == 50
    assert normalize_score(dimension, None)['normalized_score'] is None
    assert normalize_score(dimension, float('nan'))['normalized_score'] is None


def test_all_fifteen_valid_means_produce_equal_weight_formal_total():
    manifest, details = fixture_inputs()
    report = build_report(manifest, details, {})
    assert report['total']['status'] == 'formal' and report['total']['score'] == 50
    assert report['total']['valid_dimensions'] == 15
    assert all(v == 1 / 15 for v in report['total']['weights'].values())
    assert successful(report)
    assert report['normalization_protocol']['official_total'] is False
    assert '正式总分: 50.00/100' in format_summary(report)
    # Change one normalized metric from 50 to 100; total rises by 50/15.
    for row in details['fourth_qa_audio']['records']:
        row['score'] = 1
    assert build_report(manifest, details, {})['total']['score'] == pytest.approx(50 + 50 / 15)


def test_lsa_omission_keeps_mean_and_prevents_formal_total():
    manifest, details = fixture_inputs()
    # An absent official record is aligned to the original input denominator.
    details['second_lsa']['records'].pop()
    report = build_report(manifest, details, {})
    item = report['metrics']['second_lsa']
    assert item['mean'] == 5 and item['coverage_percent'] == 50
    assert item['expected'] == 2 and item['scored'] == item['omitted'] == 1
    assert report['total']['status'] == 'preview' and report['total']['score'] == 50
    assert not successful(report)
    details['second_lsa']['records'].clear()
    report = build_report(manifest, details, {})
    assert report['total']['score'] is None and report['total']['status'] == 'unavailable'
    assert report['total']['valid_dimensions'] == 14


def test_single_metric_smoke_success_does_not_fabricate_a_fifteen_metric_total():
    manifest, details = fixture_inputs()
    report = build_report(manifest, {'second_clap': details['second_clap']}, {}, ['second_clap'])
    assert successful(report)
    assert len(report['metrics']) == 15 and len(report['details']) == 1
    assert report['total']['score'] is None
    assert report['metrics']['first_dnsmos']['status'] == 'not_evaluated'


def test_mean_before_clipping_and_out_of_anchor_samples():
    manifest, details = fixture_inputs()
    for row, value in zip(details['first_nisqa']['records'], [-1, 5]):
        row['score'] = value
    item = build_report(manifest, details, {})['metrics']['first_nisqa']
    assert item['mean'] == 2 and item['normalized_score'] == 25
    assert item['out_of_bounds_count'] == 1 and item['mean_out_of_bounds'] is False
    assert item['scenes']['one']['normalized_score'] == 0


def test_overall_mean_weights_videos_instead_of_scenes():
    entries = [dict(idx='a', category='one'), dict(idx='b', category='one'),
               dict(idx='c', category='two')]
    raw = [dict(video_path=r['idx'] + '.mp4', video_results=s) for r, s in zip(entries, [1, 1, 5])]
    details = {'first_nisqa': dict(records=normalize_records('first_nisqa', raw, entries))}
    report = build_report(dict(input_signature='fixture', records=entries), details, {})
    metric = report['metrics']['first_nisqa']
    assert metric['mean'] == pytest.approx(7 / 3)
    assert metric['normalized_score'] == pytest.approx(100 / 3)
    assert metric['scenes']['one']['normalized_score'] == 0
    assert metric['scenes']['two']['normalized_score'] == 100


def test_failure_scores_stay_null_and_original_fallback_stays_auditable():
    expected = [dict(idx='a', category='scene')]
    raw = [dict(video_path='a.mp4', video_results=1.0,
                reasoning='No reasoning provided (empty or OOM/parse fallback).')]
    row = normalize_records('third_alignment', raw, expected)[0]
    assert row['status'] == 'parse_error' and row['score'] is None
    assert row['original_score'] == 1 and row['raw'] == raw[0]
    qa = [dict(video_path='a.mp4', video_results=0, answers=[
        dict(answer='no', reason='OOM_ERROR')])]
    row = normalize_records('fourth_qa_audio', qa, expected)[0]
    assert row['status'] == 'backend_error' and row['score'] is None
    # Legitimate zero QA and low macro grades remain valid.
    assert normalize_records('fourth_qa_audio', [dict(video_path='a.mp4', video_results=0)], expected)[0]['status'] == 'ok'
    assert normalize_records('third_alignment', [dict(video_path='a.mp4', video_results=1,
                             reasoning='Valid low grade')], expected)[0]['status'] == 'ok'


def test_failed_sample_is_excluded_without_changing_remaining_mean():
    manifest, details = fixture_inputs()
    row = details['second_desync']['records'][0]
    row.update(status='backend_error', original_score=1.0, score=1.0)
    report = build_report(manifest, details, {})
    assert report['details']['second_desync']['records'][0]['score'] is None
    assert report['metrics']['second_desync']['mean'] == 1.0
    assert report['metrics']['second_desync']['failed'] == 1
    assert report['total']['status'] == 'preview'


def test_unattributed_backend_errors_cannot_certify_a_formal_total():
    manifest, details = fixture_inputs()
    details['second_clap']['errors'] = [{'error': 'Unattributed backend failure'}]
    report = build_report(manifest, details, {})
    assert report['metrics']['second_clap']['coverage_percent'] == 100
    assert report['metrics']['second_clap']['backend_issue_count'] == 1
    assert report['total']['status'] == 'preview'
    assert not successful(report)


def test_csv_preserves_original_means_and_adds_normalized_and_total_outputs(tmp_path):
    manifest, details = fixture_inputs()
    report = build_report(manifest, details, {})
    export_csv(report, tmp_path)
    with (tmp_path / 'metrics.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 15 and all(float(r['normalized_score']) == 50 for r in rows)
    with (tmp_path / 'total.csv').open(encoding='utf-8-sig', newline='') as stream:
        row = next(csv.DictReader(stream))
    assert row['status'] == 'formal' and float(row['score']) == 50
    assert '正式总分' in (tmp_path / 'summary.md').read_text(encoding='utf-8')


def test_summarize_cli_reuses_old_raw_records_and_does_not_modify_input(tmp_path, capsys):
    from ais_bench.benchmark.vabench.cli import main
    manifest, details = fixture_inputs()
    original = build_report(manifest, details, {})
    original['schema_version'] = 1
    original.pop('total')
    source = tmp_path / 'old_report.json'
    write_json(source, original)
    signature = digest(source)
    output = tmp_path / 'new_summary'
    assert main(['summarize', '--report', str(source), '--output', str(output)]) == 0
    report = read_json(output / 'report.json')
    assert report['total']['score'] == 50 and digest(source) == signature
    assert report['postprocessing']['model_inference_rerun'] is False
    assert report['postprocessing']['source_report_sha256'] == signature
    assert '正式总分' in capsys.readouterr().out
    with pytest.raises(FileExistsError):
        main(['summarize', '--report', str(source), '--output', str(output)])


def test_old_false_success_is_reclassified_without_mutating_original_report():
    manifest, details = fixture_inputs()
    old = build_report(manifest, details, {})
    old['schema_version'] = 1
    row = old['details']['third_alignment']['records'][0]
    row.update(status='ok', score=1.0)
    row.pop('original_score')
    row['raw'].update(video_results=1.0,
                      reasoning='No reasoning provided (empty or OOM/parse fallback).')
    snapshot = copy.deepcopy(old)
    refreshed = refresh_report(old)
    assert old == snapshot
    row = refreshed['details']['third_alignment']['records'][0]
    assert row['score'] is None and row['original_score'] == 1.0
    assert row['status'] == 'parse_error'
    assert refreshed['total']['status'] == 'preview'


def test_aisbench_table_shows_fifteen_metrics_and_one_total(tmp_path):
    from mmengine.config import ConfigDict
    from ais_bench.benchmark.summarizers.vabench import VABenchSummarizer
    manifest, details = fixture_inputs()
    report = build_report(manifest, details, {})
    write_json(tmp_path / 'results/test/vabench.json', report)
    summarizer = VABenchSummarizer(ConfigDict(models=[dict(abbr='test')],
        datasets=[dict(abbr='vabench', path='fixture')], work_dir=str(tmp_path)))
    raw, parsed, metrics, modes = summarizer._pick_up_results()
    assert len(parsed['test']) == 16
    assert parsed['test']['vabench/total']['normalized_score'] == 50
    assert parsed['test']['vabench/total']['status'] == '正式总分'
    table = summarizer._format_table(parsed, metrics, modes)
    assert len(table) == 17 and table[-1][4] == '50.00'
    summarizer.summarize(output_path=str(tmp_path / 'summary/fixture.txt'), time_str='fixture')
    for name in ('fixture.csv', 'fixture.md'):
        text = (tmp_path / 'summary' / name).read_text(encoding='utf-8')
        assert 'vabench/total' in text and '正式总分' in text


def test_duplicate_and_unexpected_sample_records_cannot_certify_total():
    manifest, details = fixture_inputs()
    details['first_dnsmos']['records'].append(copy.deepcopy(details['first_dnsmos']['records'][0]))
    with pytest.raises(ValueError, match='duplicate/unexpected'):
        build_report(manifest, details, {})
