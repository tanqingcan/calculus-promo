// node shoot.mjs cues            -> cues.json
// node shoot.mjs 5,20,60         -> shots/tXXX.jpg
import { chromium } from 'playwright';
import { writeFileSync, mkdirSync } from 'node:fs';
const arg = process.argv[2] || 'cues';
const proxy = process.env.HTTPS_PROXY ? { server: process.env.HTTPS_PROXY } : undefined;
const b = await chromium.launch({ args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-certificate-errors'], proxy });
const p = await b.newPage({ viewport: { width: 1920, height: 1080 }, ignoreHTTPSErrors: true });
const errs = [];
p.on('pageerror', e => errs.push('pageerror: ' + e.message));
p.on('console', m => { if (m.type() === 'error') errs.push('console: ' + m.text()); });
await p.goto('file://' + process.cwd() + '/calculus.html?export=1');
const t0 = Date.now();
await p.waitForFunction(() => window.__ready && window.__ready(), null, { timeout: 30000 });
console.log('ready in', Date.now() - t0, 'ms; fonts:', await p.evaluate(() => document.fonts.check('900 40px "Noto Serif SC"', '微积分')));
if (arg === 'cues') { writeFileSync('cues.json', JSON.stringify(await p.evaluate(() => window.__cues()), null, 1)); console.log('cues written'); }
else {
  mkdirSync('shots', { recursive: true });
  for (const t of arg.split(',').map(Number)) {
    const url = await p.evaluate(t => window.__frame(t), t);
    writeFileSync(`shots/t${t.toFixed(2).padStart(7, '0')}.jpg`, Buffer.from(url.split(',')[1], 'base64'));
  }
  console.log('shots done');
}
console.log(errs.slice(0, 20).join('\n') || 'no errors');
await b.close();
