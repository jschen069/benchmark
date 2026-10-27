"""Scores retain native units. No synthetic cross-dimension accuracy."""
import math
import csv
from pathlib import Path
from statistics import mean

from . import DIMENSIONS


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def normalize_records(dimension, raw, expected, backend_errors=False):
    """Keep missing/failed results separate from genuine numerical zero."""
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
        # Official code sometimes catches an exception and emits a fallback
        # number. If logs report an error, preserve all numbers but refuse to
        # certify this invocation as a clean scoring run.
        reason = str((row or {}).get('reasoning', '')).upper()
        if backend_errors or 'OOM_ERROR' in reason:
            status = 'backend_error'
        elif row is None:
            status = 'missing'
        elif not finite_number(score):
            status = 'invalid_score'
        else:
            status = 'ok'
        result.append(dict(idx=entry['idx'], category=entry['category'],
                           status=status, score=score if finite_number(score) else None,
                           raw=row))
    return result


def summarize_dimension(records):
    def stats(rows):
        valid = [r['score'] for r in rows if r['status'] == 'ok']
        raw = [r['score'] for r in rows if finite_number(r.get('score'))]
        return dict(expected=len(rows), scored=len(valid),
                    missing=sum(r['status'] == 'missing' for r in rows),
                    failed=sum(r['status'] not in ('ok', 'missing') for r in rows),
                    mean=mean(valid) if valid else None,
                    official_numeric_mean=mean(raw) if raw else None)
    report = stats(records)
    report['scenes'] = {category: stats([r for r in records if r['category'] == category])
                        for category in sorted({r['category'] for r in records})}
    return report


def build_report(manifest, dimensions, provenance):
    scope = manifest.get('evaluation_scope', 'legacy')
    return dict(schema_version=1, input_signature=manifest['input_signature'],
                evaluation_scope=scope,
                scope_note=('Single-sample smoke test; not Mini10/778 benchmark acceptance.'
                            if scope == 'smoke' else ''),
                sample_ids=sorted(r['idx'] for r in manifest['records']),
                provenance=provenance, details=dimensions,
                metrics={d: summarize_dimension(v['records']) for d, v in dimensions.items()},
                directions={d: 'lower' if d == 'second_desync' else 'higher'
                            for d in dimensions})


def export_csv(report, output):
    output = Path(output)
    with (output / 'metrics.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['dimension', 'direction', 'mean', 'expected', 'scored', 'missing', 'failed'])
        for dim, values in report['metrics'].items():
            writer.writerow([dim, report['directions'][dim]] + [values[k] for k in (
                'mean', 'expected', 'scored', 'missing', 'failed')])
    with (output / 'samples.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['dimension', 'idx', 'category', 'status', 'score', 'requested_device'])
        for dim, values in report['details'].items():
            for row in values['records']:
                writer.writerow([dim] + [row[k] for k in ('idx', 'category', 'status', 'score')]
                                + [values['requested_device']])
    with (output / 'scenes.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['dimension', 'category', 'mean', 'expected', 'scored', 'missing', 'failed'])
        for dim, values in report['metrics'].items():
            for category, scene in values['scenes'].items():
                writer.writerow([dim, category] + [scene[k] for k in (
                    'mean', 'expected', 'scored', 'missing', 'failed')])


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
