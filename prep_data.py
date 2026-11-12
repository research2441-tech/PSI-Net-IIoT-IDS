"""Data preparation for PSI-Net experiments.

Builds two feature views for Edge-IIoTset (ML version) and one view for
WUSTL-IIoT-2021, assigns every field to a protocol segment, and stores
connection/session group keys used for group-disjoint splitting.
Usage: python prep_data.py RAW_DIR OUT_DIR
"""
import sys, json, numpy as np, pandas as pd

RAW, OUT = sys.argv[1], sys.argv[2]

FERRAG_DROP = ['frame.time', 'ip.src_host', 'ip.dst_host', 'arp.src.proto_ipv4',
               'arp.dst.proto_ipv4', 'http.file_data', 'http.request.full_uri',
               'icmp.transmit_timestamp', 'http.request.uri.query', 'tcp.options',
               'tcp.payload', 'tcp.srcport', 'tcp.dstport', 'udp.port', 'mqtt.msg']
# identifier-like fields additionally removed in the audited view
ID_EXTRA = ['tcp.ack_raw', 'tcp.checksum', 'icmp.checksum', 'udp.stream',
            'mbtcp.trans_id', 'dns.qry.name']
CONST = ['mqtt.msg_decoded_as', 'http.tls_port']

EDGE_SEG = {'arp': 'ARP', 'icmp': 'ICMP', 'http': 'HTTP', 'tcp': 'TCP', 'udp': 'UDP',
            'dns': 'DNS', 'mqtt': 'MQTT', 'mbtcp': 'MBTCP'}
CATEGORICAL = {'http.request.method', 'http.referer', 'http.request.version',
               'mqtt.protoname', 'mqtt.topic'}


def _clean_str(s):
    s = s.astype(str).str.strip()
    return s.where(~s.isin(['0', '0.0', 'nan', '']), 'NONE')


def edge_frame(e, view):
    drop = FERRAG_DROP + CONST
    if view == 'audited':
        drop = [c for c in FERRAG_DROP if c != 'tcp.dstport'] + ID_EXTRA + CONST
    X = e.drop(columns=drop + ['Attack_label', 'Attack_type']).copy()
    num, cat = {}, {}
    for c in X.columns:
        if c in CATEGORICAL:
            s = _clean_str(X[c])
            if c == 'http.referer':
                s = s.where(s.isin(['NONE', '127.0.0.1']), 'OTHER')
            if c == 'http.request.version':
                s = s.where(s.isin(['NONE', 'HTTP/1.1', 'HTTP/1.0']), 'MALFORMED')
            cat[c] = s
        elif c == 'dns.qry.name.len':
            v = pd.to_numeric(X[c], errors='coerce')
            num[c] = v.fillna(X[c].astype(str).str.len()).astype(float)
        elif c == 'mqtt.conack.flags':
            num[c] = X[c].astype(str).map(lambda t: int(t, 16) if t.startswith('0x') else float(t)).astype(float)
        else:
            num[c] = pd.to_numeric(X[c], errors='coerce').fillna(0.0).astype(float)
    return pd.DataFrame(num), pd.DataFrame(cat)


def edge_groups(e):
    a = e['ip.src_host'].astype(str); b = e['ip.dst_host'].astype(str)
    pa = e['tcp.srcport'].astype(str); pb = e['tcp.dstport'].astype(str)
    left = a + ':' + pa; right = b + ':' + pb
    lo = np.where(left < right, left, right); hi = np.where(left < right, right, left)
    key = pd.Series(lo) + '|' + pd.Series(hi) + '|' + e['udp.stream'].astype(str).values
    return pd.factorize(key)[0]


def pack(name, num, cat, seg_of, y_names, groups, extra=None):
    feats = list(num.columns) + list(cat.columns)
    cat_codes, cat_vocab = [], {}
    for c in cat.columns:
        codes, uniq = pd.factorize(cat[c], sort=True)
        cat_codes.append(codes); cat_vocab[c] = list(map(str, uniq))
    classes = sorted(pd.unique(y_names))
    if 'Normal' in classes: classes.remove('Normal'); classes = ['Normal'] + classes
    if 'normal' in classes: classes.remove('normal'); classes = ['normal'] + classes
    y = pd.Categorical(y_names, categories=classes).codes.astype(np.int16)
    np.savez_compressed(f'{OUT}/{name}.npz', Xnum=num.values.astype(np.float64),
                        Xcat=(np.stack(cat_codes, 1) if cat_codes else np.zeros((len(y), 0))).astype(np.int32),
                        y=y, groups=np.asarray(groups, np.int64))
    meta = dict(name=name, num_features=list(num.columns), cat_features=list(cat.columns),
                cat_vocab=cat_vocab, segments=[seg_of(f) for f in feats], classes=classes,
                n=int(len(y)), class_counts={c: int((y == i).sum()) for i, c in enumerate(classes)})
    if extra: meta.update(extra)
    json.dump(meta, open(f'{OUT}/{name}.json', 'w'), indent=1)
    print(name, num.shape, cat.shape, meta['class_counts'])


e = pd.read_csv(f'{RAW}/ML-EdgeIIoT-dataset.csv', low_memory=False)
g = edge_groups(e)
for view in ['audited', 'ferrag']:
    num, cat = edge_frame(e, view)
    pack(f'edge_{view}', num, cat, lambda f: EDGE_SEG[f.split('.')[0]],
         e['Attack_type'].values, g, dict(view=view, n_groups=int(g.max() + 1)))

# ---------------- WUSTL-IIoT-2021 ----------------
w = pd.read_csv(f'{RAW}/wustl_iiot_2021.csv')
rng = np.random.default_rng(2021)
t = pd.to_datetime(w['StartTime'])
win = (t - t.min()).dt.total_seconds() // 120
gkey = w['SrcAddr'].astype(str) + '|' + w['DstAddr'].astype(str) + '|' + win.astype(int).astype(str)
gid = pd.factorize(gkey)[0]
normal_groups = np.unique(gid[w['Traffic'].values == 'normal'])
# keep all attack flows, and normal flows from a random 12% of normal session windows
keep_g = set(rng.choice(normal_groups, size=int(0.12 * len(normal_groups)), replace=False))
mask = (w['Traffic'].values != 'normal') | np.isin(gid, list(keep_g))
w = w[mask].reset_index(drop=True); gid = pd.factorize(gid[mask])[0]
WDROP = ['StartTime', 'LastTime', 'SrcAddr', 'DstAddr', 'sIpId', 'dIpId', 'Sport', 'IdleTime',
         'Traffic', 'Target']
WSEG = {}
for f in ['SrcPkts', 'DstPkts', 'TotPkts', 'SrcBytes', 'DstBytes', 'TotBytes', 'SAppBytes', 'DAppBytes', 'TotAppByte']: WSEG[f] = 'VOLUME'
for f in ['SrcLoad', 'DstLoad', 'Load', 'SrcRate', 'DstRate', 'Rate']: WSEG[f] = 'RATE'
for f in ['SrcLoss', 'DstLoss', 'Loss', 'pLoss']: WSEG[f] = 'LOSS'
for f in ['Dur', 'RunTime', 'Mean', 'Sum', 'Min', 'Max', 'SIntPkt', 'DIntPkt', 'SrcJitter', 'DstJitter',
          'SrcJitAct', 'DstJitAct', 'TcpRtt', 'SynAck']: WSEG[f] = 'TIMING'
for f in ['Proto', 'Dport', 'sTtl', 'dTtl', 'sTos', 'sDSb']: WSEG[f] = 'HEADER'
num = w.drop(columns=WDROP).astype(float)
num = num.loc[:, num.nunique() > 1]
pack('wustl', num, pd.DataFrame(index=num.index), lambda f: WSEG[f], w['Traffic'].values, gid,
     dict(n_groups=int(gid.max() + 1), normal_window_fraction=0.12))
