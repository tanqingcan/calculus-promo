"""Score + sound design for the calculus promo: A minor, 90 BPM, locked to calculus.html's bar timeline.

    node shoot.mjs cues      (the page's sound-effect cue sheet -> cues.json)
    python music.py          -> music.wav, music.mp3

Everything is synthesized here: organ, formant choir, strings, piano, celesta, plucks, drums, braams and
foley from the cue sheet; separate buses, convolution reverb, kick sidechain, glue compression, limiter,
then ffmpeg two-pass loudnorm to -14 LUFS.
"""
import json, subprocess, wave
import numpy as np
from scipy.signal import butter, sosfilt, sosfiltfilt, fftconvolve, lfilter
from scipy.ndimage import maximum_filter1d

SR = 44100
BPM = 90
BEAT = 60 / BPM
BAR = 4 * BEAT
EIGHTH = BEAT / 2
NBARS = 71
TOTAL = NBARS * BAR + 4.0
N = int((TOTAL + 1) * SR)
F32 = np.float32
TAU = 2 * np.pi
rng = np.random.default_rng(7)
BUS = {k: np.zeros((2, N), F32) for k in ('mus', 'pad', 'drm', 'sfx', 'hall', 'plate', 'sfxwet')}
KICKS = []


def midi(m): return 440.0 * 2 ** ((m - 69) / 12)
def b(x): return x * BAR
def tv(d): return np.arange(max(1, int(d * SR))) / SR
def noise(n): return rng.standard_normal(n).astype(F32)


def put(sig, t0, bus, gain=1.0, pan=0.0, hall=0.0, plate=0.0, wet=0.0):
    sig = np.asarray(sig, F32)
    if sig.ndim == 1:
        a = (np.clip(pan, -1, 1) + 1) * np.pi / 4
        sig = np.stack([sig * np.cos(a), sig * np.sin(a)]) * 1.4142
    i = int(round(t0 * SR))
    if i < 0: sig, i = sig[:, -i:], 0
    n = min(sig.shape[1], N - i)
    if n <= 0: return
    seg = sig[:, :n] * gain
    BUS[bus][:, i:i + n] += seg
    for k, g in (('hall', hall), ('plate', plate), ('sfxwet', wet)):
        if g: BUS[k][:, i:i + n] += seg * g


def adsr(n, a, d, s, r, hold):
    t = np.arange(n) / SR
    e = np.where(t < a, t / max(a, 1e-4), s + (1 - s) * np.exp(-(t - a) / max(d, 1e-4)))
    return (e * (1 - np.clip((t - hold) / max(r, 1e-4), 0, 1))).astype(F32)


def _f(kind, f): return butter(2, f, kind, fs=SR, output='sos')
def lp(x, f): return sosfilt(_f('low', min(f, SR * .45)), x).astype(F32)
def hp(x, f): return sosfilt(_f('high', max(f, 10)), x).astype(F32)
def bp(x, lo, hi): return sosfilt(_f('band', [max(lo, 10), min(hi, SR * .45)]), x).astype(F32)


def saw(f, t, det=0.0, vib=0.0, vr=5.5, ph=0.0):
    inst = f * (1 + det) * (1 + vib * np.sin(TAU * vr * t + ph)) if vib else np.full(t.shape, f * (1 + det))
    phase = (np.cumsum(inst) / SR + ph / TAU) % 1.0
    dt = inst / SR
    y = 2 * phase - 1
    m = phase < dt; x = phase[m] / dt[m]; y[m] -= x + x - x * x - 1
    m = phase > 1 - dt; x = (phase[m] - 1) / dt[m]; y[m] -= x * x + x + x + 1
    return y.astype(F32)


def sweep(f0, f1, dur):
    t = tv(dur); f = f0 * (f1 / f0) ** (t / dur)
    return np.sin(TAU * np.cumsum(f) / SR).astype(F32), t


# ================================================================== instruments
STOPS = {'full': [(1, 1), (2, .6), (3, .3), (4, .32), (6, .14), (8, .1)], 'flute': [(1, 1), (2, .25), (3, .06)], 'pedal': [(1, 1), (2, .5), (3, .22)]}


def organ(t0, notes, dur, gain=.06, att=.4, rel=1.4, stops='full', hall=1.0):
    n = int((dur + rel) * SR); t = np.arange(n) / SR
    e = adsr(n, att, .5, .92, rel, dur)
    for i, m in enumerate(notes):
        f = midi(m); s = np.zeros(n, F32)
        for h, a in STOPS[stops]:
            if f * h > 8000: continue
            for dv in (-.0011, .0012): s += (a * np.sin(TAU * f * h * (1 + dv) * t + rng.random() * TAU)).astype(F32)
        put(s * e / (2 * len(STOPS[stops])), t0, 'pad', gain, ((i / max(1, len(notes) - 1)) - .5) * .6, hall=hall)


FORM = {'a': [(800, 1.0), (1150, .5), (2900, .22), (3900, .1)], 'o': [(450, 1.0), (800, .45), (2830, .12)]}


def choir(t0, notes, dur, gain=.05, att=1.0, rel=1.8, vowel='a', hall=1.3):
    n = int((dur + rel) * SR); t = np.arange(n) / SR
    e = adsr(n, att, 1.0, .9, rel, dur)
    for i, m in enumerate(notes):
        f = midi(m)
        src = sum(saw(f, t, det=(v - 1) * .005, vib=.0045, vr=4.8 + v * .7, ph=rng.random() * 6) for v in range(3)) + noise(n) * .25
        out = sum(a * bp(src, fc * .86, fc * 1.14) for fc, a in FORM[vowel])
        put(out * e, t0, 'pad', gain, ((i / max(1, len(notes) - 1)) - .5) * .8, hall=hall)


def strings(t0, notes, dur, gain=.04, att=.5, rel=1.0, cut=3500, hall=.8):
    n = int((dur + rel) * SR); t = np.arange(n) / SR
    e = adsr(n, att, .8, .9, rel, dur)
    for i, m in enumerate(notes):
        s = sum(saw(midi(m), t, det=(v - 1.5) * .0035, vib=.0035, vr=5.2 + v * .4, ph=rng.random() * 6) for v in range(4))
        put(lp(s, cut) * e / 4, t0, 'pad', gain, ((i / max(1, len(notes) - 1)) - .5) * .9, hall=hall)


def spic(t0, m, gain=.05, dur=.16, pan=0.0):
    t = tv(dur + .25)
    s = sum(saw(midi(m), t, det=d) for d in (-.004, 0, .004)) / 3
    put(lp(s, 2800) * adsr(len(t), .006, .08, .35, .12, dur), t0, 'mus', gain, pan, hall=.35)


def piano(t0, m, gain=.1, vel=.8, pan=0.0):
    f = midi(m); t = tv(4.5); B = .00015 * 2 ** ((m - 48) / 12); s = np.zeros(len(t), F32)
    for h in range(1, 11):
        fh = f * h * np.sqrt(1 + B * h * h)
        if fh > 12000: break
        a = vel ** (1 + .25 * (h - 1)) / h ** .85
        tau = max(.25, 3.2 - (m - 60) * .04) / (1 + .45 * (h - 1))
        dec = np.exp(-t / tau).astype(F32)
        for dc in (-.0006, 0, .0007): s += (a / 3 * np.sin(TAU * fh * (1 + dc) * t + rng.random() * 6)).astype(F32) * dec
    hn = int(.012 * SR)
    s[:hn] += lp(noise(hn), 1500 + 3000 * vel) * np.exp(-np.arange(hn) / SR / .003).astype(F32) * .3 * vel
    put(s * np.minimum(1, t / .002), t0, 'mus', gain, pan, hall=.55)


def bell(t0, m, gain=.05, dec=1.4, pan=0.0, ratio=3.5, index=1.8, bus='mus', hall=.8, plate=0.0, wet=0.0):
    f = midi(m); t = tv(dec * 5)
    s = np.sin(TAU * f * t + index * np.exp(-t / .3) * np.sin(TAU * f * ratio * t)) * np.exp(-t / dec)
    s += .3 * np.sin(TAU * f * 2.01 * t) * np.exp(-t / (dec * .4))
    put(s * np.minimum(1, t / .0015), t0, bus, gain, pan, hall=hall, plate=plate, wet=wet)


def celesta(t0, m, gain=.04, dec=1.0, pan=0.0, bus='mus', wet=0.0):
    bell(t0, m, gain, dec, pan, ratio=4.0, index=.9, bus=bus, hall=.7, plate=.25, wet=wet)


def pluck(t0, m, gain=.04, dur=.2, pan=0.0):
    t = tv(dur + .3)
    s = saw(midi(m), t, det=.003) + saw(midi(m), t, det=-.003)
    k = np.exp(-t / .07).astype(F32)
    s = (lp(s, 6000) * k + lp(s, 900) * (1 - k)) * adsr(len(t), .003, .15, .3, .12, dur)
    put(s, t0, 'mus', gain, pan, hall=.3, plate=.3)


def sub(t0, m, dur, gain=.16):
    f = midi(m); t = tv(dur + .3)
    s = np.tanh(1.5 * (np.sin(TAU * f * t) + .2 * np.sin(TAU * 2 * f * t))) * adsr(len(t), .02, .3, .9, .25, dur)
    put(s, t0, 'pad', gain)


def bass8(t0, m, gain=.1, dur=EIGHTH * .9):
    f = midi(m); t = tv(dur + .15)
    s = (lp(saw(f, t), 700) + .6 * np.sin(TAU * f * t).astype(F32)) * adsr(len(t), .004, .12, .6, .08, dur)
    put(s, t0, 'mus', gain)


def braam(t0, root, dur, gain):
    t = tv(dur + .5)
    for m in (root, root + 12, root + 19, root + 24):
        s = saw(midi(m), t, det=.004) + saw(midi(m), t, det=-.005)
        k = np.exp(-t / 1.0).astype(F32)
        s = np.tanh(2.2 * (lp(s, 2600) * k + lp(s, 380) * (1 - k))) * adsr(len(t), .04, 1.5, .5, .5, dur)
        put(s, t0, 'mus', gain / 4, hall=.7)


def growl(t0, dur, gain=.07):
    t = tv(dur)
    s = saw(midi(21), t, det=.003) + saw(midi(21), t, det=-.004) + .5 * saw(midi(28), t)
    o = (t / dur) ** 1.5
    s = np.tanh(3 * (lp(s, 180) * (1 - o) + lp(s, 900) * o)) * np.minimum(1, t / 2) * np.minimum(1, (dur - t) / .5)
    put(s.astype(F32), t0, 'mus', gain, hall=.4)


def kick(t0, gain=.45, duck=.6):
    t = tv(.55); f = 44 + 130 * np.exp(-t / .03)
    s = np.sin(TAU * np.cumsum(f) / SR) * np.exp(-t / .26) + .35 * hp(noise(len(t)), 2000) * np.exp(-t / .004)
    put(np.tanh(1.6 * s), t0, 'drm', gain); KICKS.append((t0, duck))


def taiko(t0, gain=.35, pitch=1.0, pan=0.0):
    t = tv(1.6); f = (58 + 70 * np.exp(-t / .045)) * pitch
    body = np.sin(TAU * np.cumsum(f) / SR) * np.exp(-t / .55) + .35 * np.sin(TAU * np.cumsum(f * 1.58) / SR) * np.exp(-t / .25)
    s = body + bp(noise(len(t)), 150, 900) * np.exp(-t / .07) * .6 + hp(noise(len(t)), 2500) * np.exp(-t / .004) * .4
    put(np.tanh(1.3 * s), t0, 'drm', gain, pan, hall=.35); KICKS.append((t0, .4))


def snare(t0, gain=.18, pan=0.0, big=False):
    t = tv(.7)
    s = bp(noise(len(t)), 900, 8000) * np.exp(-t / (.22 if big else .14))
    s += .6 * np.sin(TAU * 185 * t) * np.exp(-t / .06) + .3 * np.sin(TAU * 330 * t) * np.exp(-t / .04)
    put(s, t0, 'drm', gain, pan, hall=.4 if big else .15)


def clap(t0, gain=.13):
    n = int(.5 * SR); s = np.zeros(n, F32)
    for k, d in enumerate((0, .011, .023)):
        i = int(d * SR); s[i:] += bp(noise(n - i), 1000, 5000) * np.exp(-np.arange(n - i) / SR / (.012 if k < 2 else .12)).astype(F32)
    put(s, t0, 'drm', gain, plate=.3)


def hat(t0, gain=.035, pan=.25):
    t = tv(.07); put(hp(noise(len(t)), 7000) * np.exp(-t / .018), t0, 'drm', gain, pan)


def shaker(t0, gain=.025, pan=-.2):
    t = tv(.12); put(bp(noise(len(t)), 4000, 11000) * np.sin(np.pi * t / .12) ** 2, t0, 'drm', gain, pan)


def crash(t0, gain=.1, dur=2.6, rev=False):
    t = tv(dur); s = hp(noise(len(t)), 4500) * np.exp(-t / (dur * .35))
    for r in (3.1, 4.7, 6.9, 9.4): s += .08 * np.sin(TAU * 410 * r * t) * np.exp(-t / (dur * .25))
    if rev: put(s[::-1].copy(), t0 - dur, 'drm', gain, hall=.5)
    else: put(s, t0, 'drm', gain, hall=.5)


def tom(t0, m, gain=.22, pan=0.0):
    t = tv(.8); f = midi(m) * (1 + .5 * np.exp(-t / .05))
    put(np.sin(TAU * np.cumsum(f) / SR) * np.exp(-t / .25), t0, 'drm', gain, pan, hall=.3)


def impact(t0, lvl=1.0, root=21):
    t = tv(6); f = 28 + 60 * np.exp(-t / .3)
    put(np.tanh(2 * np.sin(TAU * np.cumsum(f) / SR)) * np.exp(-t / 1.6), t0, 'drm', .5 * lvl)
    taiko(t0, .45 * lvl, .8); taiko(t0 + .012, .3 * lvl, 1.3, -.3)
    put(lp(noise(len(t)), 2500) * np.exp(-t / .45), t0, 'drm', .1 * lvl, hall=1.2)
    crash(t0, .09 * lvl); braam(t0, root, 4.5, .11 * lvl)
    KICKS.append((t0, 1.0))


def riser(t_end, dur, gain=.14):
    t = tv(dur); x = t / dur
    s = hp(noise(len(t)), 1200) * x ** 2.2 * .7
    for o in range(3):
        f = 180 * 2 ** o * 2 ** (x * 2)
        s += (np.sin(TAU * np.cumsum(f) / SR) * x ** 2 * .12).astype(F32)
    put(s.astype(F32), t_end - dur, 'mus', gain, hall=.6)


# ================================================================== harmony helpers (A minor)
CH = {'Am': [57, 60, 64, 69], 'F': [53, 57, 60, 65], 'C': [55, 60, 64, 67], 'G': [55, 59, 62, 67], 'Dm': [53, 57, 62, 65],
      'E': [56, 59, 64, 68], 'Esus': [57, 59, 64, 69], 'A': [57, 61, 64, 69], 'Am9': [57, 60, 64, 71], 'Fmaj7': [53, 57, 60, 64]}
ROOT = {'Am': 33, 'F': 29, 'C': 36, 'G': 31, 'Dm': 38, 'E': 28, 'Esus': 28, 'A': 33, 'Am9': 33, 'Fmaj7': 29}
ARP = {k: sorted(set(v + [x + 12 for x in v])) for k, v in CH.items()}


def pads(start, chords, kind='organ', **kw):
    for i, c in enumerate(chords):
        if not c: continue
        if kind == 'organ': organ(b(start + i), CH[c], BAR, **kw)
        elif kind == 'strings': strings(b(start + i), CH[c], BAR, **kw)
        else: choir(b(start + i), [x + 12 for x in CH[c][1:]], BAR, **kw)


def bass_hold(start, chords, gain=.14):
    for i, c in enumerate(chords): sub(b(start + i), ROOT[c] + 12, BAR, gain)


def bass_8(start, chords, gain=.1):
    for i, c in enumerate(chords):
        for k in range(8): bass8(b(start + i) + k * EIGHTH, ROOT[c] + 12 + (12 if k % 4 == 3 else 0), gain)
        sub(b(start + i), ROOT[c] + 12, BAR, .09)


def arps(start, chords, inst='celesta', div=2, oct_=12, gain=.03, pat=(0, 2, 1, 3, 2, 4, 3, 5)):
    step = BEAT / div
    for i, c in enumerate(chords):
        notes = ARP[c]
        for k in range(4 * div):
            m = notes[pat[k % len(pat)] % len(notes)] + oct_
            t, g, pan = b(start + i) + k * step, gain * (1 if k % div == 0 else .75), np.sin(k * 1.3) * .5
            if inst == 'celesta': celesta(t, m, g, dec=.7, pan=pan)
            else: pluck(t, m, g, dur=step * .9, pan=pan)


def drums(start, bars, kick_=(0, 2), snare_=(), clap_=(), hats=0, taikos=(), kg=.45, sg=.18, fill=False, shak=0):
    for i in range(bars):
        t = b(start + i)
        for k in kick_: kick(t + k * BEAT, kg)
        for s in snare_: snare(t + s * BEAT, sg, big=True)
        for c in clap_: clap(t + c * BEAT)
        for k in taikos: taiko(t + k * BEAT, .28, 1.0, pan=.2 * np.sin(k))
        for h in range(hats): hat(t + h * BAR / hats, .035 if h % 2 == 0 else .024)
        for h in range(shak): shaker(t + h * BAR / shak, .022 if h % 2 else .03)
        if fill and i == bars - 1:
            for k, m in enumerate([52, 50, 47, 45, 43, 40, 38, 36]): tom(t + 2 * BEAT + k * BEAT / 4, m, .2, pan=.6 - k * .17)


def sparkle_notes(t0, dur, count, gain=.025, seed=0):
    r = np.random.default_rng(seed); pent = [69, 72, 74, 76, 79]
    for _ in range(count):
        celesta(t0 + r.random() * dur, int(pent[r.integers(5)] + 12 * r.integers(1, 3)), gain * (.5 + .5 * r.random()), dec=1.2, pan=r.random() * 1.6 - .8)


# ================================================================== the score
# ---- 0-6  cold open: drone under the teaser, silence, the question, the curve
organ(0.0, [33, 40], 6.55, gain=.07, att=1.5, rel=.06, stops='pedal')
strings(1.0, [57, 64, 69, 76], 5.55, gain=.03, att=4.5, rel=.06, cut=1600)
riser(6.62, 4.5, .07)
organ(7.35, [69, 76, 81], 3.2, gain=.02, att=1.6, rel=1.6, stops='flute')
organ(b(4), CH['Am'], b(2) - .25, gain=.05, att=2.6, rel=.5)
choir(b(4) + 2.4, [69, 72, 76], b(2) - 2.7, gain=.03, att=2.2, rel=.5)
sparkle_notes(b(4), b(2) - .5, 22, seed=1)
for k, m in enumerate([69, 72, 76, 79, 81, 84]):                     # the rider sets off
    celesta(12.9 + k * BEAT / 2, m, .025, dec=1.0, pan=(k - 2.5) * .2)
riser(b(6), b(1), .16)
crash(b(6), .08, 2.2, rev=True)
# ---- 6-9  title
impact(b(6), 1.0)
organ(b(6), CH['Am'] + [33, 40], b(3) - .3, gain=.075, att=.08, rel=1.6)
choir(b(6), [69, 72, 76, 81], b(3) - .3, gain=.055, att=.5)
sub(b(6), 45, b(3) - .5, .16)
for k, m in enumerate([81, 84, 88, 93, 88, 84, 81, 76, 81, 84, 88, 93]):
    celesta(b(6.75) + k * EIGHTH, m, .035, dec=1.4, pan=(k % 4 - 1.5) * .3)
# ---- 9-17  limits
LM = ['Am', 'F', 'C', 'G', 'Am', 'F', 'Dm', 'E']
pads(9, LM, 'organ', gain=.038, stops='flute', att=.8)
pads(9, LM, 'strings', gain=.026, att=.9)
bass_hold(9, LM, .11)
arps(10, LM[1:5], 'pluck', div=2, gain=.022)
arps(14, LM[5:], 'celesta', div=4, oct_=12, gain=.02)
for i in range(13, 15): kick(b(i), .22, duck=.3)
strings(b(14.75), [45, 52, 57, 64], b(2.25), gain=.03, att=2.0, rel=.3, cut=1800)   # the Mandelbrot dive
riser(b(17), b(2), .16)
choir(b(16), [68, 71, 76], BAR, gain=.045, att=.3)
braam(b(16) + .02, 28, 2.4, .05)
# ---- 17-26  the derivative
impact(b(17), .35)
DR = ['Am', 'F', 'C', 'G', 'Am', 'F', 'G', 'Esus', 'E']
pads(17, DR, 'strings', gain=.03)
organ(b(17), [45, 52], b(2), gain=.03, att=.6, stops='flute')
bass_hold(17, DR[:2], .1)
bass_8(19, DR[2:], .09)
arps(19, DR[2:], 'pluck', div=4, gain=.024, pat=(0, 2, 4, 2, 1, 3, 5, 3))
drums(19, 7, kick_=(0, 2), kg=.34)
drums(20, 6, kick_=(), hats=8)
drums(22, 4, kick_=(), clap_=(1, 3))
riser(b(26), b(1), .13)
# ---- 26-34  the integral
impact(b(26), .4)
IN = ['F', 'G', 'Am', 'Am', 'F', 'G', 'C', 'E']
pads(26, IN, 'strings', gain=.028, cut=2600)
for i, c in enumerate(IN):
    r = ROOT[c] + 36
    for k in range(8): spic(b(26 + i) + k * EIGHTH, [r, r + 7, r + 12, r + 7][k % 4], .042 if i < 3 else .05, pan=(k % 2 - .5) * .4)
bass_8(26, IN, .1)
choir(b(29), [69, 72, 76], b(1.5), gain=.04, att=.6)                 # the long S
drums(26, 3, kick_=(0, 2), hats=16, kg=.3)
drums(29, 2, kick_=(0, 2), kg=.24)
drums(31, 3, kick_=(0, 1.75, 2, 3.5), snare_=(1, 3), hats=16, kg=.4, sg=.17, fill=True)
crash(b(34), .08, 2.0, rev=True)
# ---- 34-41  the fundamental theorem
impact(b(34), .5)
FT = ['Am', 'F', 'C', 'G']
pads(34, FT, 'organ', gain=.04)
pads(34, FT, 'strings', gain=.028)
bass_hold(34, FT, .11)
arps(34, FT, 'celesta', div=2, oct_=24, gain=.018)
for i in range(34, 38): taiko(b(i), .2, .9); taiko(b(i) + 2 * BEAT, .14, 1.0)
riser(b(38), b(1), .15)
impact(b(38), .6)
FT2 = ['F', 'G', 'Am']
pads(38, FT2, 'organ', gain=.06)
pads(38, FT2, 'choir', gain=.05)
pads(38, FT2, 'strings', gain=.03)
for i, c in enumerate(FT2): braam(b(38 + i), ROOT[c], BAR * .9, .04)
bass_hold(38, FT2, .12)
drums(38, 3, kick_=(0, 2), taikos=(0,), hats=8, kg=.36)
# ---- 41-47  Taylor
TY = ['F', 'C', 'G', 'Am', 'F', 'E']
pads(41, TY, 'strings', gain=.026, cut=2600)
pads(41, TY, 'organ', gain=.025, stops='flute')
bass_hold(41, TY, .1)
arps(41, TY, 'celesta', div=2, oct_=12, gain=.02)
MEL = [(0, 76), (1.5, 77), (2, 76), (4, 72), (5.5, 74), (6, 76), (8, 79), (9.5, 77), (10, 76), (12, 74), (14, 72), (15, 71),
       (16, 72), (17.5, 74), (18, 76), (20, 77), (22, 76), (23, 74)]
for bt, m in MEL: piano(b(41) + bt * BEAT, m, .08, .65, .1); piano(b(41) + bt * BEAT, m - 12, .035, .55, -.1)
drums(42, 5, kick_=(0, 2), kg=.24, shak=16)
# ---- 47-54  e and Euler
EU = ['Am', 'Am', 'Dm', 'Dm', 'E']
pads(47, EU, 'strings', gain=.026, att=1.2, cut=2000)
bass_hold(47, EU, .09)
for i in range(2):
    for k, m in enumerate([69, 72, 76, 81, 76, 72, 69, 64]): piano(b(47 + i) + k * EIGHTH, m + 12, .04, .55, np.sin(k) * .3)
for i, c in enumerate(['Dm', 'Dm', 'E']):
    for k in range(8): pluck(b(49 + i) + k * EIGHTH, ARP[c][[0, 2, 4, 2, 1, 3, 5, 3][k]] + 12, .022, dur=.18, pan=np.sin(k) * .4)
drums(50, 2, kick_=(0, 2), hats=8, kg=.28)
for k in range(16): snare(b(51) + k * BEAT / 4, .015 + .008 * k)
riser(b(52), b(1), .17)
impact(b(52), .9, root=21)
organ(b(52), CH['A'] + [33, 45], b(2) - .2, gain=.07, att=.08, rel=1.6)          # the Picardy third: A major for e^{iπ}+1=0
choir(b(52), [69, 73, 76, 81], b(2) - .2, gain=.055, att=.4)
strings(b(52), [57, 64, 69, 73], b(2) - .2, gain=.03, att=.3)
sub(b(52), 45, b(2) - .4, .14)
for k, m in enumerate([81, 85, 88, 93, 97]): celesta(b(52) + .4 + k * .2, m, .03, dec=2.0, pan=(k - 2) * .3)
# ---- 54-63  a universe of change
impact(b(54), .5)
UN = ['Am', 'F', 'C', 'G']
pads(54, UN, 'strings', gain=.03)
pads(54, UN, 'organ', gain=.026, stops='flute')
bass_8(54, UN, .09)
arps(54, UN, 'pluck', div=4, gain=.02)
drums(55, 2, kick_=(0, 2), hats=8, kg=.34)
impact(b(57), .45)
growl(b(57), b(3), .06)
LZ = ['Am', 'Dm', 'E']
pads(57, LZ, 'strings', gain=.03, cut=1300)
for i, c in enumerate(LZ):
    for k in range(16): bass8(b(57 + i) + k * EIGHTH / 2, ROOT[c] + 12 + (12 if k % 8 in (3, 6) else 0), .075, dur=EIGHTH / 2 * .8)
drums(57, 3, kick_=(0, 2.5), clap_=(1, 3), hats=16, kg=.4)
impact(b(60), .5)
WV = ['F', 'G', 'Am']
pads(60, WV, 'organ', gain=.05)
pads(60, WV, 'choir', gain=.045)
pads(60, WV, 'strings', gain=.03)
for i, c in enumerate(WV): braam(b(60 + i), ROOT[c], BAR * .9, .04)
arps(60, WV, 'celesta', div=4, oct_=24, gain=.02)
bass_8(60, WV, .1)
drums(60, 2, kick_=(0, 1.5, 2, 3.5), snare_=(1, 3), hats=16, taikos=(0, 2), kg=.44, sg=.19)
for k in range(32): snare(b(62) + k * BEAT / 8, .01 + .0035 * k)
riser(b(63), b(2), .18)
# ---- 63-71  finale
impact(b(63), .9)
FN = ['F', 'G', 'Am', 'C']
pads(63, FN, 'organ', gain=.06)
pads(63, FN, 'choir', gain=.05)
pads(63, FN, 'strings', gain=.032)
for i, c in enumerate(FN): braam(b(63 + i), ROOT[c], BAR * .9, .045)
arps(63, FN, 'celesta', div=4, oct_=24, gain=.022)
bass_8(63, FN, .11)
drums(63, 4, kick_=(0, 1.5, 2, 3.5), snare_=(1, 3), hats=16, taikos=(0, 2), kg=.46, sg=.2, fill=True)
impact(b(67), .7)
organ(b(67), CH['F'] + [29], BAR, gain=.065)
organ(b(68), CH['G'] + [31], BAR, gain=.065)
choir(b(67), [72, 77, 81], BAR, gain=.045)
choir(b(68), [71, 74, 79], BAR, gain=.045)
bass_hold(67, ['F', 'G'], .12)
drums(67, 2, kick_=(0, 2), snare_=(2,), hats=8, taikos=(0,), kg=.42)
impact(b(69), .5, root=21)
END = b(2) + 3.6
organ(b(69), CH['A'] + [33, 40, 73, 76], END, gain=.075, att=.1, rel=2.6)
choir(b(69), [73, 76, 81, 85], END, gain=.05, att=.4, rel=2.6)
strings(b(69), [57, 64, 69, 73], END, gain=.03, att=.5, rel=2.6)
sub(b(69), 45, END, .12)
for k, m in enumerate([81, 85, 88, 93, 97, 100]): celesta(b(69) + 3.2 + k * .22, m, .035, dec=2.4, pan=(k - 2.5) * .25)


# ================================================================== sound effects from the cue sheet
def whoosh_sig(dur, lo=300, hi=3000, rev=False):
    t = tv(dur); x = t / dur; n = noise(len(t)); k = np.sin(np.pi * x).astype(F32)
    s = (bp(n, lo, lo * 3) * (1 - k) + bp(n, hi / 3, hi) * k) * np.sin(np.pi * x) ** 1.5
    return (s[::-1].copy() if rev else s).astype(F32)


def fx_thump(t, **o):
    for d, g in ((0, 1), (.28, .6)):
        s, tt = sweep(62, 38, .35); put(s * np.exp(-tt / .12), t + d, 'sfx', .5 * g, wet=.2)


def fx_heart(t, **o):
    for d, g in ((0, 1), (.26, .65)):
        s, tt = sweep(58, 40, .3); put(s * np.exp(-tt / .1), t + d, 'sfx', .33 * g, wet=.25)


def fx_hit(t, k=.6, **o):
    s, tt = sweep(80, 34, 1.2); put(np.tanh(2 * s * np.exp(-tt / .35)), t, 'sfx', .42 * k, wet=.3)
    put(bp(noise(len(tv(.5))), 1800, 9000) * np.exp(-tv(.5) / .025), t, 'sfx', .22 * k, wet=.4)
    tt = tv(1.2); met = sum(np.sin(TAU * f * tt) * np.exp(-tt / d) for f, d in ((1390, .3), (2310, .2), (3710, .12)))
    put(met, t, 'sfx', .05 * k, wet=.6)
    put(whoosh_sig(.22, 800, 6000, rev=True), t - .22, 'sfx', .1 * k)


def fx_suck(t, dur=.45, **o):
    tt = tv(dur); put(hp(noise(len(tt)), 3000) * (tt / dur) ** 3, t, 'sfx', .22, wet=.2)


def fx_ring(t, dur=2.5, **o):
    tt = tv(dur); e = np.minimum(1, tt / .05) * np.exp(-tt / (dur * .45))
    put((np.sin(TAU * 3150 * tt) + .5 * np.sin(TAU * 4270 * tt)) * e, t, 'sfx', .02, wet=.4)
    put(lp(noise(len(tt)), 90) * np.exp(-tt / (dur * .5)), t, 'sfx', .25)


def fx_sparkle(t, dur=2.0, **o):
    r = np.random.default_rng(int(t * 100))
    for _ in range(int(dur * 26)):
        u = r.random(); f = midi([88, 91, 93, 95, 98, 100, 103, 105][r.integers(8)]); tt = tv(.35)
        put(np.sin(TAU * f * tt) * np.exp(-tt / (.08 + .2 * r.random())), t + u * dur, 'sfx', .012 * np.sin(np.pi * u) ** .5, r.random() * 1.6 - .8, wet=.8)


def fx_swell(t, dur=1.5, **o):
    tt = tv(dur); x = tt / dur
    put(lp(noise(len(tt)), 2500) * x ** 3, t, 'sfx', .12, wet=.7)
    for m in (69, 76, 81): put(np.sin(TAU * midi(m) * tt) * x ** 2.5, t, 'sfx', .02, wet=1.0)


def fx_whoosh(t, dur=1.0, rev=0, **o):
    s = whoosh_sig(dur, 250, 3500, bool(rev)); a = (np.linspace(-.6, .6, len(s)) + 1) * np.pi / 4
    put(np.stack([s * np.cos(a), s * np.sin(a)]) * 1.41, t, 'sfx', .2, wet=.35)


def fx_shimmer(t, **o):
    tt = tv(3.0)
    s = sum(np.sin(TAU * 1760 * r * tt * (1 + .002 * np.sin(TAU * 5 * tt))) * np.exp(-tt / d) for r, d in ((1, 2.2), (2.76, 1.4), (5.4, .9), (8.93, .5)))
    put(s * np.minimum(1, tt / .01), t, 'sfx', .03, wet=.9); fx_sparkle(t, 1.2)


def fx_chime(t, midi_=81, **o): bell(t, midi_, .06, dec=2.4, bus='sfx', hall=0, wet=.8)
def fx_ding(t, midi_=81, **o): bell(t, midi_, .055, dec=1.6, index=2.4, bus='sfx', hall=0, wet=.6)
def fx_note(t, midi_=72, **o): celesta(t, midi_, .07, dec=1.6, bus='sfx', wet=.6)


def fx_tick(t, gain=1.0, pitch=1.0, **o):
    tt = tv(.05)
    s = np.sin(TAU * 3200 * pitch * tt) * np.exp(-tt / .006) + .5 * hp(noise(len(tt)), 4000) * np.exp(-tt / .003)
    put(s, t, 'sfx', .09 * gain, .1, wet=.25)


def fx_pop(t, midi_=76, **o):
    f = midi(midi_); tt = tv(.18); fr = f * (1 + np.exp(-tt / .015))
    put(np.sin(TAU * np.cumsum(fr) / SR) * np.exp(-tt / .06), t, 'sfx', .05, np.sin(midi_ * 1.7) * .5, wet=.4)


def fx_blip(t, midi_=84, **o):
    f = midi(midi_); tt = tv(.25); fr = f * (1 + .5 * np.exp(-tt / .02))
    put(np.tanh(2.5 * np.sin(TAU * np.cumsum(fr) / SR)) * np.exp(-tt / .1), t, 'sfx', .04, np.sin(midi_) * .4, wet=.3)


def fx_boom(t, **o):
    s, tt = sweep(70, 30, 2.5); put(np.tanh(1.8 * s) * np.exp(-tt / .9), t, 'sfx', .38, wet=.4)
    put(lp(noise(len(tt)), 400) * np.exp(-tt / .4), t, 'sfx', .15, wet=.6)


def fx_glide(t, dur=8.0, **o):
    tt = tv(dur); x = tt / dur; s = np.zeros(len(tt), F32)
    for f0 in (440, 660, 880):
        s += np.sin(TAU * np.cumsum(f0 * 2 ** x * (1 + .004 * np.sin(TAU * 5.5 * tt))) / SR).astype(F32)
    put(s * np.sin(np.pi * x) ** 2, t, 'sfx', .01, wet=1.0)


def fx_quill(t, dur=1.0, **o):
    tt = tv(dur); r = np.random.default_rng(int(t * 10))
    rate = 10 + 3 * np.sin(TAU * .7 * tt + r.random() * 6)
    strokes = np.abs(np.sin(np.pi * np.cumsum(rate) / SR)) ** 3
    s = bp(noise(len(tt)), 2500, 7500) * strokes * (.6 + .4 * r.random(len(tt)).astype(F32))
    put(s * np.minimum(1, tt / .05) * np.minimum(1, (dur - tt) / .08), t, 'sfx', .05, .25, wet=.2)


def fx_stamp(t, **o):
    s, tt = sweep(95, 55, .5)
    put(s * np.exp(-tt / .12) + bp(noise(len(tt)), 200, 900) * np.exp(-tt / .05) + .5 * hp(noise(len(tt)), 2000) * np.exp(-tt / .02), t, 'sfx', .32, wet=.35)


FX = {k[3:]: v for k, v in globals().items() if k.startswith('fx_')}
cues = json.load(open('cues.json', encoding='utf-8'))
missing = sorted({c['name'] for c in cues} - FX.keys())
if missing: raise SystemExit(f'no sound for cues: {missing}')
for c in cues:
    FX[c['name']](c['t'], **{('midi_' if k == 'midi' else k): v for k, v in c.items() if k not in ('t', 'name')})
print(f'placed {len(cues)} sound effects')


# ================================================================== mix + master
def ir(bands, dur, pre=.02, seed=0):
    r = np.random.default_rng(seed); n = int(dur * SR); t = np.arange(n) / SR; out = []
    for ch in range(2):
        x = r.standard_normal(n).astype(F32); y = np.zeros(n, F32)
        for (lo, hi), tau in bands: y += (bp(x, lo, hi) if lo else lp(x, hi)) * np.exp(-t / tau)
        y *= np.minimum(1, t / .008)
        y = np.concatenate([np.zeros(int(pre * SR), F32), y]); out.append(y / np.sqrt(np.sum(y ** 2)))
    return out


HALL = ir([((0, 300), 1.2), ((300, 1500), 1.0), ((1500, 5000), .7), ((5000, 16000), .35)], 3.6, .025, 1)
PLATE = ir([((0, 500), .55), ((500, 4000), .6), ((4000, 16000), .45)], 1.8, .008, 2)


def conv(bus, h):
    out = np.stack([fftconvolve(BUS[bus][c], h[c])[:N].astype(F32) for c in range(2)]); BUS[bus] = None; return out


print('reverbs…')
wet_music = conv('hall', HALL) * .55 + conv('plate', PLATE) * .5
wet_sfx = conv('sfxwet', PLATE) * .6
duck = np.zeros(N, F32)
for tk, depth in KICKS:
    i = int(tk * SR); L = min(int(.45 * SR), N - i)
    if L > 0: duck[i:i + L] = np.maximum(duck[i:i + L], depth * np.exp(-np.arange(L) / SR / .16))
music = BUS['mus'] + BUS['pad'] * (1 - .5 * duck) + BUS['drm'] * .9 + wet_music
sfx = BUS['sfx'] + wet_sfx
tt = np.arange(N, dtype=F32) / SR
gate = np.clip((6.66 - tt) / .05, 0, 1) + np.clip((tt - 7.2) / .6, 0, 1)       # hard cut to silence after the teaser
music *= np.minimum(gate, 1)
mix = music * .92 + sfx
mix = np.stack([hp(mix[c], 30) for c in range(2)])
mix -= .6 * sosfiltfilt(butter(2, 120, 'low', fs=SR, output='sos'), mix, axis=1).astype(F32)
mix -= .3 * sosfiltfilt(butter(2, 7000, 'high', fs=SR, output='sos'), mix, axis=1).astype(F32)
mono = np.sqrt((mix[0] ** 2 + mix[1] ** 2) / 2)
a = np.exp(-1 / (.12 * SR))
env = np.sqrt(lfilter([1 - a], [1, -a], mono ** 2) + 1e-12)
lvl = 20 * np.log10(env / (np.max(env) + 1e-9) + 1e-9)
mix *= (10 ** (-np.maximum(0, lvl + 12) * (1 - 1 / 2.2) / 20)).astype(F32)
mix /= np.max(np.abs(mix)) + 1e-9
L = int(.004 * SR)
pk = maximum_filter1d(np.max(np.abs(mix), axis=0), size=2 * L + 1)
g = np.minimum(1, .5 / np.maximum(pk, 1e-9)); a = np.exp(-1 / (.09 * SR))
mix *= np.minimum(g, lfilter([1 - a], [1, -a], g)).astype(F32)
end = int(TOTAL * SR); mix = mix[:, :end]
mix *= np.minimum(1, (end - np.arange(end)) / (2.5 * SR)).astype(F32)
mix /= np.max(np.abs(mix)) + 1e-9
with wave.open('music.wav', 'wb') as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes((mix.T * 32000).astype(np.int16).tobytes())
LN = 'loudnorm=I=-14:TP=-1.5:LRA=11'
m = subprocess.run(['ffmpeg', '-hide_banner', '-i', 'music.wav', '-af', LN + ':print_format=json', '-f', 'null', '-'], capture_output=True, text=True).stderr
m = json.loads(m[m.rindex('{'):m.rindex('}') + 1])
LN += f":measured_I={m['input_i']}:measured_TP={m['input_tp']}:measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}:offset={m['target_offset']}:linear=true"
subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', 'music.wav', '-af', LN + ',alimiter=limit=-1.8dB:level=false', '-ar', '44100', '-b:a', '160k', 'music.mp3'], check=True)
print(f'music: {TOTAL:.2f}s')
