# 微积分：变化之美

一部约 3 分 13 秒的微积分宣传片。画面由 HTML Canvas 和 WebGL 逐帧绘制，配乐与音效由 Python 合成，带中英双语字幕，画面为 1920 × 1080。

**直接观看**：用浏览器打开 `微积分宣传片.html`（配乐已内嵌；字体从 Google Fonts 加载，需要联网）。

## 章节

| 小节 | 章节 | 内容 |
| --- | --- | --- |
| 0–6 | 冷开场 | 心跳光点 → 闪切预告 → 粒子聚成曲线与切线 |
| 6–9 | 标题 | 粒子聚成「微积分」，体积光 + 冲击波 |
| 9–17 | Ⅰ 极限 | 阿基米德割圆 → 科赫雪花 → 曼德博集合深度放大 |
| 17–26 | Ⅱ 导数 | 割线收缩成切线 → f′(x) → 3D 曲面上的切平面与梯度 |
| 26–34 | Ⅲ 积分 | 黎曼和逐拍翻倍 → ∫ → 圆环展开得 πr² → 旋转体 |
| 34–41 | Ⅳ 基本定理 | 面积积累与斜率 → 微积分基本定理 |
| 41–47 | Ⅴ 泰勒级数 | 多项式逼近 sin x → 复平面域着色 |
| 47–54 | Ⅵ e 与欧拉 | 复利 → (1+iπ/n)ⁿ → 3D 螺旋 e^{it} → 欧拉恒等式 |
| 54–63 | Ⅶ 变化的宇宙 | 开普勒轨道 → 1200 粒子洛伦兹混沌 → 弯曲时空与六大方程 |
| 63–71 | 终章 | 每拍一镜的蒙太奇 → 标题 → 收回心跳光点 |

## 工程结构

| 文件 | 用途 |
| --- | --- |
| `parts/0_core.html` | 页面骨架、工具函数、WebGL 背景着色器与电影感后期 |
| `parts/1_scenes_a.js` | 开场、标题、极限、导数、积分 |
| `parts/2_scenes_b.js` | 基本定理、泰勒、欧拉、宇宙、终章 |
| `parts/3_pipeline.html` | 时间轴、字幕、重音冲击、播放器与导出接口 |
| `build.py` | 拼接 parts → `calculus.html`；加 `--embed` 内嵌配乐 → `微积分宣传片.html` |
| `music.py` | 合成配乐与音效（读取 `cues.json`）→ `music.wav`、`music.mp3` |
| `shoot.mjs` | 用无头浏览器导出音效清单或截取检查帧 |
| `play.mjs` | 播放模式冒烟测试 |
| `cues.json` | 由页面导出的音效时间表 |

整部片子以 90 BPM 的小节为时间单位，画面（`scene({ start, len })`）和配乐（`b(小节)`）共用同一套常量，所以二者天然对齐。

## 重新生成

需要 Python 3、Node.js 18+、FFmpeg。

```bash
pip install -r requirements.txt
npm install
npx playwright install chromium

python build.py               # 拼出 calculus.html
node shoot.mjs cues           # 导出音效清单 cues.json
python music.py               # 合成 music.mp3
python build.py --embed       # 生成单文件播放器 微积分宣传片.html
node shoot.mjs 20,60,140      # 截检查帧到 shots/
```
