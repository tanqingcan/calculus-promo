"""Concatenate the parts into calculus.html; with --embed, inline music.mp3 as base64 -> 微积分宣传片.html"""
import base64, json, os, sys
here = os.path.dirname(os.path.abspath(__file__))
parts = ['0_core.html', '1_scenes_a.js', '2_scenes_b.js', '3_pipeline.html']
html = ''.join(open(os.path.join(here, 'parts', p), encoding='utf-8').read() for p in parts)
open(os.path.join(here, 'calculus.html'), 'w', encoding='utf-8').write(html)
if '--embed' in sys.argv:
    audio = base64.b64encode(open(os.path.join(here, 'music.mp3'), 'rb').read()).decode('ascii')
    marker = "<script>\n'use strict';"
    assert marker in html
    html = html.replace(marker, f'<script>window.EMBED = {json.dumps({"audio": audio})};</script>\n' + marker, 1)
    out = os.path.join(here, '微积分宣传片.html')
    open(out, 'w', encoding='utf-8').write(html)
    print(f'wrote {out} ({os.path.getsize(out)/1e6:.1f} MB)')
print('ok')
