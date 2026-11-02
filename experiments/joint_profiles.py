"""Preloaded cost, admission and residual profiles for joint execution.

Times persisted by calibration are seconds. The selector receives integer
nanoseconds; no CUDA or tensor imports occur in this module.
"""
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from statistics import median
from typing import Mapping
from types import MappingProxyType


NS = 1_000_000_000


def seconds_ns(value):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError('service time must be finite and nonnegative')
    return math.ceil(value * NS)


def nearest_rank(values, quantile=.90):
    ordered = sorted(values)
    if not ordered or not 0 < quantile <= 1:
        raise ValueError('quantile requires nonempty samples and 0 < q <= 1')
    if not all(math.isfinite(x) for x in ordered):
        raise ValueError('nonfinite sample')
    return ordered[math.ceil(quantile * len(ordered)) - 1]


def summarize_samples(values):
    values = tuple(values)
    return {'p50': median(values), 'p90': nearest_rank(values), 'samples': len(values)}


def group_keys(group):
    mode, route, total, cache_class = tuple(group)
    return ((mode, route, total, cache_class), (mode, route, total), (mode, route), ())


@dataclass(frozen=True)
class ResidualTable:
    entries: tuple = ()
    min_sources: int = 5
    min_requests: int = 20

    def lookup(self, group):
        entries = {tuple(e['key']): e for e in self.entries}
        for key in group_keys(group):
            entry = entries.get(key)
            if entry is None:
                continue
            delta = entry['delta_s']
            if (entry['request_count'] >= self.min_requests and
                    len(set(entry['sources'])) >= self.min_sources and
                    math.isfinite(delta) and delta >= 0):
                return key, delta
        raise ValueError('residual_profile_missing')

    def to_dict(self):
        return {'entries': [dict(e, key=list(e['key']), sources=list(e['sources']))
                            for e in self.entries],
                'min_sources': self.min_sources, 'min_requests': self.min_requests}

    @classmethod
    def from_dict(cls, data):
        if isinstance(data, cls):
            return data
        return cls(tuple(dict(e, key=tuple(e['key']), sources=tuple(e['sources']))
                         for e in data.get('entries', ())),
                   data.get('min_sources', 5), data.get('min_requests', 20))


def fit_residuals(records, min_sources=5, min_requests=20):
    """Pool each request's worst committed-decision residual at every key level."""
    groups = {}
    for row in records:
        if row.get('success', True) is False:
            raise ValueError('failed requests cannot fit residual margins')
        group = (row['execution_mode'], row['route'], row['total_context_tokens'], row['cache_class'])
        error = (row['first_token_ns'] - row['F_base_ns']) / NS
        if not math.isfinite(error):
            raise ValueError('nonfinite residual')
        for key in group_keys(group):
            requests = groups.setdefault(key, {})
            rid = str(row['request_id'])
            old = requests.get(rid)
            source = str(row['source_id'])
            if old and old[1] != source:
                raise ValueError('request_id maps to multiple sources')
            requests[rid] = (max(error, old[0]) if old else error, source)
    entries = []
    for key, requests in groups.items():
        sources = sorted({v[1] for v in requests.values()})
        if len(sources) >= min_sources and len(requests) >= min_requests:
            entries.append({'key': key, 'delta_s': max(0., nearest_rank([v[0] for v in requests.values()])),
                            'sources': tuple(sources), 'request_count': len(requests)})
    return ResidualTable(tuple(entries), min_sources, min_requests)


def verify_residuals(records, table, min_sources=5, min_requests=20, minimum_coverage=.90,
                     required_groups=None, required_bandwidth_classes=None):
    """Independent request-level coverage, quality and unexpected-failure gate."""
    table = ResidualTable.from_dict(table) if isinstance(table, dict) else table
    groups = {}
    unexpected_failures = 0
    quality_ok = True
    rows = tuple(records)
    errors = []
    observed_groups = set()
    for row in rows:
        if row.get('success', True) is False:
            unexpected_failures += 1
            continue
        if 'nll_delta' not in row or 'em_retention' not in row:
            quality_ok = False
        elif (not math.isfinite(row['nll_delta']) or not math.isfinite(row['em_retention']) or
              row['nll_delta'] > .02 or row['em_retention'] < .90):
            quality_ok = False
        group = (row['execution_mode'], row['route'], row['total_context_tokens'], row['cache_class'])
        observed_groups.add(group)
        try:
            key, delta = table.lookup(group)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        safe = row.get('F_safe_ns', row['F_base_ns'] + seconds_ns(delta))
        covered = row['first_token_ns'] <= safe
        requests = groups.setdefault(key, {})
        rid = str(row['request_id'])
        entry = requests.setdefault(rid, {'covered': True, 'source': str(row['source_id']),
                                         'slo': True, 'positive_error_s': 0.})
        if entry['source'] != str(row['source_id']):
            raise ValueError('request_id maps to multiple sources')
        entry['covered'] &= covered
        entry['slo'] &= row['first_token_ns'] <= row.get('deadline_ns', row['first_token_ns'])
        entry['positive_error_s'] = max(entry['positive_error_s'], (row['first_token_ns'] - safe) / NS)
    summaries = []
    for key, requests in groups.items():
        sources = sorted({r['source'] for r in requests.values()})
        n = len(requests)
        coverage = sum(r['covered'] for r in requests.values()) / n
        enough = len(sources) >= min_sources and n >= min_requests
        summaries.append({'key': list(key), 'source_count': len(sources), 'sources': sources,
                          'request_count': n, 'coverage': coverage,
                          'slo_satisfaction': sum(r['slo'] for r in requests.values()) / n,
                          'positive_error_s': max(r['positive_error_s'] for r in requests.values()),
                          'passed': enough and coverage >= minimum_coverage})
    missing_groups = [list(g) for g in (required_groups or ()) if tuple(g) not in observed_groups]
    observed_bandwidth_classes = {r.get('bandwidth_class') for r in rows}
    missing_bandwidth_classes = [c for c in (required_bandwidth_classes or ())
                                 if c not in observed_bandwidth_classes]
    ready = (bool(summaries) and quality_ok and not unexpected_failures and not errors and
             not missing_groups and not missing_bandwidth_classes and all(g['passed'] for g in summaries))
    return {'ready': ready, 'groups': summaries, 'quality_ok': quality_ok,
            'unexpected_failures': unexpected_failures, 'errors': errors,
            'minimum_coverage': minimum_coverage, 'missing_groups': missing_groups,
            'missing_bandwidth_classes': missing_bandwidth_classes}


def validate_source_split(split):
    seen = set()
    for name in ('cost', 'residual', 'verify', 'test'):
        values = split.get('verification') if name == 'verify' and 'verify' not in split else split[name]
        ids = [str(x) for x in values]
        if len(set(ids)) < 5:
            raise ValueError('source partition needs at least five distinct sources: ' + name)
        if len(ids) != len(set(ids)) or seen.intersection(ids):
            raise ValueError('source partition overlap')
        seen.update(ids)
    return True


class BandwidthHistory:
    """Causal completed-message EWMA, independent of a trace/rate limiter."""
    def __init__(self, link_id='default', initial_bps=1e9, alpha=.25,
                 min_sample_bytes=262144, ttl_ns=5 * NS):
        if not math.isfinite(initial_bps) or initial_bps <= 0 or not 0 < alpha <= 1:
            raise ValueError('invalid bandwidth prior or alpha')
        self.link_id, self.initial_bps, self.alpha = link_id, initial_bps, alpha
        self.min_sample_bytes, self.ttl_ns = min_sample_bytes, ttl_ns
        self.reset()

    def reset(self):
        self.beta50_bps = self.beta90_bps = self.initial_bps
        self.valid = False
        self.last_completed_ns = None
        self.samples = []

    def observe(self, size, start_ns, end_ns, *, link_id=None, now_ns=None):
        if link_id is not None and link_id != self.link_id:
            return False
        if size <= 0 or end_ns <= start_ns or (now_ns is not None and end_ns > now_ns):
            self.reset()
            return False
        if size < self.min_sample_bytes:
            return False
        sample = 8 * size * NS / (end_ns - start_ns)
        if not math.isfinite(sample) or sample <= 0:
            self.reset()
            return False
        # The prior seeds the EWMA; completed measurements alone update it.
        self.beta50_bps = (1 - self.alpha) * self.beta50_bps + self.alpha * sample
        self.beta90_bps = min(self.beta50_bps, sample)
        self.valid = True
        self.last_completed_ns = end_ns
        self.samples.append({'size': size, 'start_ns': start_ns, 'end_ns': end_ns,
                             'link_id': self.link_id, 'bps': sample})
        return True

    def import_completed(self, samples, request_start_ns):
        self.reset()
        for row in sorted(samples, key=lambda x: x['end_ns']):
            age = request_start_ns - row['end_ns']
            if row['link_id'] == self.link_id and 0 <= age <= self.ttl_ns:
                self.observe(row['size'], row['start_ns'], row['end_ns'], now_ns=request_start_ns)

    @property
    def estimate_source(self):
        return 'completed_history' if self.valid else 'fixed_prior'


@dataclass
class RuntimeProfile:
    residual_table: ResidualTable = field(default_factory=ResidualTable)
    h50_ns: int = 0
    h90_ns: int = 0
    select50_ns: int = 0
    select90_ns: int = 0
    commit50_ns: int = 0
    commit90_ns: int = 0
    fingerprint: str = ''
    ready: bool = False
    synthetic: bool = False
    admitted_qualities: tuple = (1, 2, 3)
    text_admitted: bool = False
    costs: list = field(default_factory=list)
    first_rows: list = field(default_factory=list)
    quality_by_T: dict = field(default_factory=lambda: {'1024': [1, 2, 3], '2048': [1, 2, 3], '4096': [2, 3]})
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        # This read-only index is built once on load, not once per candidate or
        # forecast. Persisted seconds are converted at this boundary only.
        index = {}
        for row in self.costs:
            key = row.get('key', row)
            values = row.get('values', row)
            ns = {name[:-2] + '_ns': seconds_ns(value)
                  for name, value in values.items()
                  if name in ('g50_s', 'g90_s', 'c50_s', 'c90_s', 'r50_s', 'r90_s',
                              'access50_s', 'access90_s')}
            lookup = (key.get('q'), key.get('encoding'), key.get('N'))
            index.setdefault(lookup, []).append((MappingProxyType(dict(key)), MappingProxyType(ns)))
        self._cost_index = MappingProxyType({key: tuple(rows) for key, rows in index.items()})
        self._first_index = MappingProxyType({(row['T'], row['suffix_tokens']):
                              (seconds_ns(row['h50_s']), seconds_ns(row['h90_s']))
                              for row in self.first_rows})

    def require_formal(self):
        if self.synthetic:
            raise ValueError('synthetic profile is diagnostic_only')
        if not self.ready:
            raise ValueError('runtime_profile_not_ready')
        if not self.fingerprint:
            raise ValueError('profile_fingerprint_missing')

    def to_dict(self):
        return {**{k: v for k, v in self.__dict__.items() if not k.startswith('_')}, 'schema': 'cachegen-joint-v2',
                'residual_table': self.residual_table.to_dict(),
                'admitted_qualities': list(self.admitted_qualities)}

    @classmethod
    def from_dict(cls, data):
        values = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        values['residual_table'] = ResidualTable.from_dict(values.get('residual_table', {}))
        if 'admitted_qualities' in values:
            values['admitted_qualities'] = tuple(values['admitted_qualities'])
        return cls(**values)

    def h_ns(self, total, suffix, percentile):
        if (total, suffix) in self._first_index:
            return self._first_index[total, suffix][0 if percentile == .5 else 1]
        return self.h50_ns if percentile == .5 else self.h90_ns

    def transition_rows(self, candidate):
        return self._cost_index.get((3 if candidate.kind == 'text' else candidate.q,
                                     candidate.encoding, candidate.end - candidate.start), ())


def save_runtime_profile(path, profile):
    path = Path(path)
    if path.suffix != '.json':
        path = path / 'runtime_profile.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile.to_dict(), indent=2, ensure_ascii=False), encoding='utf-8')


def load_runtime_profile(path, fingerprint=None, require_ready=False):
    path = Path(path)
    if path.suffix != '.json':
        path = path / 'runtime_profile.json'
    profile = RuntimeProfile.from_dict(json.loads(path.read_text(encoding='utf-8')))
    if fingerprint is not None and profile.fingerprint != fingerprint:
        raise ValueError('profile_fingerprint_mismatch')
    if require_ready:
        profile.require_formal()
    return profile
