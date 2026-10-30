"""Official raw scores plus fixed display scores, coverage and an AISBench total."""
import csv
from pathlib import Path
from statistics import mean

from . import DIMENSIONS
from .data import object_digest
from .scoring import RULES, aggregate, finite_number, normalize_score, protocol


def record_failure(dimension, score, raw):
    raw = raw or {}
    reasons = [str(raw.get('reasoning', ''))]
    answers = raw.get('answers')
    if isinstance(answers, list):
        reasons.extend(str(a.get('reason', '')) for a in answers if isinstance(a, dict))
    if any('OOM_ERROR' in reason.upper() for reason in reasons):
        return 'backend_error'
    if not finite_number(score):
        return 'invalid_score'
    if dimension.startswith('third_'):
        if any('empty or oom/parse fallback' in reason.lower() for reason in reasons):
            return 'parse_error'
        if not 1 <= score <= 5:
            return 'invalid_score'
    if dimension.startswith('fourth_'):
        if not 0 <= score <= 1:
            return 'invalid_score'
        if answers is not None and (not isinstance(answers, list) or not answers or any(
                not isinstance(a, dict) or a.get('answer') not in ('yes', 'no')
                or not str(a.get('reason', '')).strip() for a in answers)):
            return 'parse_error'
    return None


def normalize_records(dimension, raw, expected, backend_errors=False):
    """Keep missing/failed outputs distinct from legitimate numerical zero."""
    lookup = {}
    for row in raw:
        path = row.get('video_path') or row.get('audio_path')
        if not path:
            raise ValueError(f'{dimension}: result without media path')
        idx = Path(path).stem
        if idx in lookup:
            raise ValueError(f'{dimension}: duplicate result {idx}')
        lookup[idx] = row
    if set(lookup) - {e['idx'] for e in expected}:
        raise ValueError(f'{dimension}: unexpected result sample')
    result = []
    for entry in expected:
        row = lookup.get(entry['idx'])
        score = row.get('video_results') if row else None
        status = ('backend_error' if backend_errors else 'missing' if row is None
                  else record_failure(dimension, score, row) or 'ok')
        result.append(dict(idx=entry['idx'], category=entry['category'], status=status,
                           score=score if status == 'ok' else None,
                           original_score=score if finite_number(score) else None, raw=row))
    return result


def summarize_dimension(records, dimension=None):
    def stats(rows):
        valid = [r['score'] for r in rows if r['status'] == 'ok' and finite_number(r.get('score'))]
        originals = [r.get('original_score', (r.get('raw') or {}).get('video_results', r.get('score')))
                     for r in rows]
        originals = [v for v in originals if finite_number(v)]
        expected = len(rows)
        report = dict(expected=expected, scored=len(valid),
                      missing=sum(r['status'] == 'missing' for r in rows),
                      failed=sum(r['status'] not in ('ok', 'missing', 'not_evaluated') for r in rows),
                      not_evaluated=sum(r['status'] == 'not_evaluated' for r in rows),
                      mean=mean(valid) if valid else None,
                      official_numeric_mean=mean(originals) if originals else None,
                      coverage=len(valid) / expected if expected else 0.0,
                      coverage_percent=100 * len(valid) / expected if expected else 0.0)
        report['omitted'] = report['missing'] if dimension == 'second_lsa' else 0
        report['status'] = ('complete' if expected and len(valid) == expected else
                            'not_evaluated' if report['not_evaluated'] == expected else 'incomplete')
        if dimension is not None:
            report.update(normalize_score(dimension, report['mean']))
            low, high, _ = RULES[dimension]
            report['out_of_bounds_count'] = sum(v < low or v > high for v in valid)
        return report
    report = stats(records)
    report['scenes'] = {cat: stats([r for r in records if r['category'] == cat])
                        for cat in sorted({r['category'] for r in records})}
    return report


def build_report(manifest, dimensions, provenance, requested_dimensions=None):
    scope = manifest.get('evaluation_scope', 'legacy')
    expected = {r['idx']: r for r in manifest['records']}
    if len(expected) != len(manifest['records']):
        raise ValueError('Duplicate input sample IDs')
    details, metrics = {}, {}
    for dimension in DIMENSIONS:
        value = dimensions.get(dimension)
        rows = value.get('records', []) if value is not None else []
        lookup = {r['idx']: r for r in rows}
        if len(lookup) != len(rows) or set(lookup) - set(expected):
            raise ValueError(f'{dimension}: duplicate/unexpected sample records')
        aligned = []
        for idx, entry in expected.items():
            if idx not in lookup:
                aligned.append(dict(idx=idx, category=entry.get('category', 'unknown'), score=None,
                                    original_score=None, raw=None,
                                    status='not_evaluated' if value is None else 'missing'))
                continue
            row = dict(lookup[idx])
            row.setdefault('category', entry.get('category', 'unknown'))
            row.setdefault('original_score', (row.get('raw') or {}).get('video_results', row.get('score')))
            if row['status'] == 'ok':
                row['status'] = record_failure(dimension, row.get('score'), row.get('raw')) or 'ok'
            if row['status'] != 'ok':
                row['score'] = None
            aligned.append(row)
        metrics[dimension] = summarize_dimension(aligned, dimension)
        if value is not None:
            metrics[dimension]['backend_issue_count'] = len(value.get('errors', []))
            if value.get('errors') and metrics[dimension]['status'] == 'complete':
                metrics[dimension]['status'] = 'incomplete'
            details[dimension] = dict(value, records=aligned)
    selected = list(dimensions) if requested_dimensions is None else list(requested_dimensions)
    normalization = protocol()
    return dict(schema_version=2, input_signature=manifest['input_signature'], evaluation_scope=scope,
                scope_note=('Single-sample smoke test; not Mini10/778 benchmark acceptance.'
                            if scope == 'smoke' else ''),
                sample_ids=sorted(expected), requested_dimensions=selected,
                provenance=provenance, details=details, metrics=metrics,
                directions={d: 'lower' if d == 'second_desync' else 'higher' for d in DIMENSIONS},
                normalization_protocol=normalization,
                normalization_protocol_sha256=object_digest(normalization), total=aggregate(metrics))


def refresh_report(report):
    """Reaggregate existing AISBench per-sample outputs without scoring again."""
    if not report.get('input_signature') or not report.get('sample_ids') or not isinstance(report.get('details'), dict):
        raise ValueError('An AISBench report with input_signature, sample_ids and per-sample details is required')
    categories = {}
    for value in report['details'].values():
        for row in value.get('records', []):
            idx, cat = row['idx'], row.get('category', 'unknown')
            if idx in categories and categories[idx] != cat:
                raise ValueError(f'Inconsistent sample category: {idx}')
            categories[idx] = cat
    manifest = dict(input_signature=report['input_signature'],
                    evaluation_scope=report.get('evaluation_scope', 'legacy'),
                    records=[dict(idx=idx, category=categories.get(idx, 'unknown')) for idx in report['sample_ids']])
    result = build_report(manifest, report['details'], report.get('provenance', {}),
                          report.get('requested_dimensions', list(report['details'])))
    if report.get('postprocessing'):
        result['postprocessing'] = report['postprocessing']
    return result


def display_report(report):
    if report.get('input_signature') and report.get('sample_ids'):
        return refresh_report(report)
    # Metric-only legacy artifacts can display native means, but cannot certify a total.
    metrics = {}
    expected = max((m.get('expected', 0) for m in report.get('metrics', {}).values()), default=0)
    for dimension in DIMENSIONS:
        data = dict(report.get('metrics', {}).get(dimension, {}))
        data.setdefault('mean', None)
        for field in ('scored', 'missing', 'failed', 'not_evaluated'):
            data.setdefault(field, 0)
        data.setdefault('expected', expected)
        data.setdefault('scenes', {})
        data['omitted'] = data['missing'] if dimension == 'second_lsa' else 0
        data['coverage_percent'] = 100 * data['scored'] / data['expected'] if data['expected'] else 0.0
        data['status'] = 'incomplete'
        data.update(normalize_score(dimension, data['mean']))
        data.setdefault('out_of_bounds_count', 0)
        metrics[dimension] = data
    result = dict(report, metrics=metrics, directions={
        d: 'lower' if d == 'second_desync' else 'higher' for d in DIMENSIONS})
    result['total'] = aggregate(metrics)
    result['total'].update(score=None, status='unavailable', label='总分不可用',
                           note='Metric-only legacy report lacks auditable per-sample records')
    return result


def format_summary(report):
    rows = [f'VABench: scope={report.get("evaluation_scope", "legacy")}, '
            f'samples={len(report.get("sample_ids", []))}', '',
            '| 指标 | 原始均分 | 归一化分(0–100) | 有效/输入 | LSA省略 | 缺失(含省略) | 失败 | 覆盖率 | 状态 |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for dimension, values in report['metrics'].items():
        raw = '—' if values['mean'] is None else f'{values["mean"]:.5f}'
        score = values.get('normalized_score')
        normalized = '—' if score is None else f'{score:.2f}'
        rows.append(f'| {dimension} | {raw} | {normalized} | {values["scored"]}/{values["expected"]} '
                    f'| {values.get("omitted", 0)} | {values["missing"]} | {values["failed"]} '
                    f'| {values["coverage_percent"]:.1f}% | {values["status"]} |')
    total = report['total']
    value = '—' if total['score'] is None else f'{total["score"]:.2f}/100'
    rows += ['', f'{total["label"]}: {value} '
             f'(有效指标 {total["valid_dimensions"]}/{total["expected_dimensions"]})',
             '总分采用固定归一化后的15项等权平均；这是AISBench展示协议，只代表当前选定样本。']
    if report.get('scope_note'):
        rows += [report['scope_note']]
    return '\n'.join(rows) + '\n'


def export_csv(report, output):
    output = Path(output)
    columns = ['mean', 'normalized_score', 'expected', 'scored', 'omitted', 'missing', 'failed',
               'not_evaluated', 'coverage_percent', 'out_of_bounds_count', 'mean_out_of_bounds', 'status']
    for filename, groups in [('metrics.csv', False), ('scenes.csv', True)]:
        with (output / filename).open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['dimension'] + (['category'] if groups else ['direction']) + columns)
            for dim, values in report['metrics'].items():
                entries = values['scenes'].items() if groups else [(report['directions'][dim], values)]
                for category, data in entries:
                    writer.writerow([dim, category] + [data.get(k) for k in columns])
    with (output / 'samples.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['dimension', 'idx', 'category', 'status', 'score', 'original_score', 'requested_device'])
        for dim, values in report['details'].items():
            for row in values['records']:
                writer.writerow([dim] + [row.get(k) for k in ('idx', 'category', 'status', 'score', 'original_score')]
                                + [values.get('requested_device', 'unobserved')])
    with (output / 'total.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['score', 'status', 'label', 'valid_dimensions', 'expected_dimensions'])
        writer.writerow([report['total'][k] for k in (
            'score', 'status', 'label', 'valid_dimensions', 'expected_dimensions')])
    (output / 'summary.md').write_text(format_summary(report), encoding='utf-8')


def compare(reference, candidate, dimensions=None, relative=0.01,
            near_zero=1e-6, absolute=1e-6):
    """Same media AND protocol, source, weights, preprocessing, status sets.

    Different devices are allowed: this is also the GPU-vs-NPU audit tool.
    Each dataset mean is sample-weighted, never an average of scene means.
    """
    if not all(finite_number(v) for v in (relative, near_zero, absolute)) or relative <= 0 or near_zero < 0 or absolute < 0:
        raise ValueError('Invalid tolerance')
    dimensions = list(DIMENSIONS if dimensions is None else dimensions)
    if not dimensions or len(set(dimensions)) != len(dimensions):
        raise ValueError('Nonempty, unique dimensions required')
    problems = []
    for field in ('input_signature', 'sample_ids'):
        if not reference.get(field) or reference.get(field) != candidate.get(field):
            problems.append(f'{field} mismatch or missing')
    for key in ('protocol_sha256', 'source_signature', 'integration_signature',
                'weights_signature', 'scoring_signature'):
        left = reference.get('provenance', {}).get(key)
        right = candidate.get('provenance', {}).get(key)
        if not left or left != right:
            problems.append(f'{key} mismatch or missing')
    reports = {}
    for dim in dimensions:
        left = reference.get('details', {}).get(dim)
        right = candidate.get('details', {}).get(dim)
        issues = []
        a = b = None
        if not left or not right:
            issues.append('dimension missing')
        else:
            lrows, rrows = left['records'], right['records']
            ls = {r['idx']: r['status'] for r in lrows}
            rs = {r['idx']: r['status'] for r in rrows}
            if len(ls) != len(lrows) or len(rs) != len(rrows):
                issues.append('duplicate sample result')
            if ls != rs or sorted(ls) != reference.get('sample_ids', []):
                issues.append('sample/status set mismatch')
            if any(s not in ('ok', 'missing') for s in list(ls.values()) + list(rs.values())):
                issues.append('backend failure/invalid score')
            if dim != 'second_lsa' and ('missing' in ls.values() or 'missing' in rs.values()):
                issues.append('unexpected missing score')
            if not any(s == 'ok' for s in ls.values()):
                issues.append('no valid scores')
            if any(not finite_number(r.get('score')) for r in lrows + rrows if r['status'] == 'ok'):
                issues.append('nonfinite score')
            if not issues:
                a = mean(r['score'] for r in lrows if r['status'] == 'ok')
                b = mean(r['score'] for r in rrows if r['status'] == 'ok')
        delta = abs(a - b) if a is not None else None
        error = delta / abs(a) if a is not None and abs(a) > near_zero else None
        passed = (delta is not None and not issues and
                  (delta <= absolute if error is None else error < relative))
        reports[dim] = dict(passed=passed, reference_mean=a, candidate_mean=b,
                            absolute_error=delta, relative_error=error,
                            rule='absolute' if error is None else 'relative', issues=issues)
    return dict(passed=not problems and all(r['passed'] for r in reports.values()),
                protocol_issues=problems, dimensions=reports,
                thresholds=dict(relative_strictly_less_than=relative,
                                near_zero_at_most=near_zero, absolute_at_most=absolute))
