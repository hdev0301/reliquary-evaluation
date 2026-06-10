import json, re, os
from collections import Counter, defaultdict

TMP = os.environ.get('TEMP', os.environ.get('TMP', '.'))
d = json.load(open(os.path.join(TMP, 'all_miners_live.json'), encoding='utf-8-sig'))
wins = d.get('windows', []) or []
HEX16 = re.compile(r'^[0-9a-f]{16}$')

cnt = defaultdict(lambda: {'math': 0, 'code': 0})
for w in wins:
    for s in (w.get('samples') or []):
        hk = (s.get('hotkey') or '?')[:8]
        if HEX16.match(str(s.get('ground_truth', '')).strip()):
            cnt[hk]['code'] += 1
        else:
            cnt[hk]['math'] += 1

def env(hk):
    c = cnt.get(hk, {'math':0,'code':0})
    if c['math'] == 0 and c['code'] == 0: return '?'
    return 'MATH' if c['math'] >= c['code'] else 'CODE'

KNOWN={'5F6VZ2ro':'uid39','5HEAK6g3':'uid181','5F7YBWD1':'uid243','5HQbAQ4U':'uid226','5Hp6EPJd':'uid15','5G3wUjwf':'uid116'}

print('=== ALL TOP MINERS by total accepted in-zone groups (72 windows), env-labeled ===')
ranked = sorted(cnt.items(), key=lambda x: -(x[1]['math']+x[1]['code']))
mathers = 0
for hk, c in ranked[:16]:
    tot = c['math'] + c['code']
    e = env(hk)
    if e == 'MATH': mathers += 1
    print('  %-9s %-8s total=%3d  [MATH=%3d CODE=%3d]  -> %s' % (hk, KNOWN.get(hk,''), tot, c['math'], c['code'], e))
print('  ... %d of the top 16 are MATH-environment miners' % mathers)

print('\n=== WINDOW WINNERS (top_hotkey across %d windows), env-labeled ===' % len(wins))
tops = Counter((w.get('top_hotkey') or '?')[:8] for w in wins)
mw = 0
for hk, n in tops.most_common():
    e = env(hk)
    if e == 'MATH': mw += n
    print('  %-9s %-8s won %2d windows  -> %s' % (hk, KNOWN.get(hk,''), n, e))
print('  -> %d/%d window-wins went to MATH-environment miners' % (mw, len(wins)))
