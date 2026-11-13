# -*- coding: utf-8 -*-
import os
import json
import math
import urllib.parse
import html as html_lib
import functools
import asyncio

import gradio as gr
from PIL import Image as PILImage, ImageDraw
from leaf_inference import predict_image

# ----------------------------------------------------------------------
# 常量与预设文案
# ----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(BASE_DIR, "class_names.json"), encoding="utf-8") as class_file:
    CLASSES = json.load(class_file)

# 各类别的人工预设特征说明（识别完成后展示，与具体图片无关）
CATEGORY_FEATURES = {
    "卵形": "叶片基部凹入，整体呈卵形轮廓。",
    "披针形": "叶片狭长，形似披针，向两端逐渐收窄，先端尖。",
    "椭圆形": "叶片中部最宽，两端渐窄，整体呈椭圆轮廓。",
    "线形": "叶片细长如线，长宽比明显偏大。",
}

# 允许的本地图片格式
ALLOWED_EXTS = {".jpg", ".jpeg", ".png"}

SAMPLE_DIR = os.path.join(BASE_DIR, "assets", "samples")

# ----------------------------------------------------------------------
# 统一风格的线性 SVG 图标（stroke 风格，避免使用 emoji）
# ----------------------------------------------------------------------
ICONS = {
    "upload": '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/>',
    "cloud_upload": '<path d="M16 16l-4-4-4 4"/><line x1="12" y1="12" x2="12" y2="21"/><path d="M20.39 18.39A5 5 0 0 0 18 9h-1.26A8 8 0 1 0 3 16.3"/>',
    "image_search": '<rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/><circle cx="17.5" cy="17.5" r="2.5"/><line x1="19.3" y1="19.3" x2="22" y2="22"/>',
    "layers": '<polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/>',
    "rotate": '<polyline points="23 4 23 10 17 10"/><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/>',
    "trash": '<polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>',
    "folder": '<path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/>',
    "check_circle": '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/>',
    "alert": '<circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>',
    "info": '<circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/>',
    "chart": '<line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/>',
    "file": '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/>',
    "search_file": '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><circle cx="11.5" cy="14.5" r="2.5"/><line x1="13.3" y1="16.3" x2="16" y2="19"/>',
    "arrow_left": '<line x1="19" y1="12" x2="5" y2="12"/><polyline points="12 19 5 12 12 5"/>',
    "image": '<rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/>',
}


def _icon(name, color="#a9d4ff", size=20, sw=2.0):
    """返回统一 stroke 风格的内联 SVG 图标字符串。"""
    return (
        f'<svg class="ic" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
        f'stroke="{color}" stroke-width="{sw}" stroke-linecap="round" '
        f'stroke-linejoin="round">{ICONS[name]}</svg>'
    )


def _data_icon(name, color, sw=2.0):
    """生成可放入 CSS background-image 的 data-URI SVG（用于按钮图标）。"""
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
        f'stroke="{color}" stroke-width="{sw}" stroke-linecap="round" '
        f'stroke-linejoin="round">{ICONS[name]}</svg>'
    )
    return 'url("data:image/svg+xml;utf8,' + urllib.parse.quote(svg) + '")'


def _esc(text):
    """HTML 转义，避免文件名/评估内容注入标签。"""
    return html_lib.escape(str(text))


# ----------------------------------------------------------------------
# 参考样例示意图自动生成（首次运行时生成到 assets/samples/）
# 说明：这些是程序绘制的“叶形示意图”，不是真实叶片照片，页面已明确标注。
# ----------------------------------------------------------------------
def _canvas(size=720):
    img = PILImage.new("RGB", (size, size), "#eef3f9")
    return img, ImageDraw.Draw(img)


def _gen_heart(size=720):
    """卵形叶示意图。"""
    img, d = _canvas(size)
    raw = []
    for deg in range(0, 360, 3):
        t = math.radians(deg)
        x = 16 * math.sin(t) ** 3
        y = 13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t)
        raw.append((x, -y))  # y 取负，使心尖朝下
    xs = [p[0] for p in raw]
    ys = [p[1] for p in raw]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    scale = (size * 0.62) / max(x1 - x0, y1 - y0)
    pts = [((size / 2) + (px - (x0 + x1) / 2) * scale,
            (size / 2) + (py - (y0 + y1) / 2) * scale) for px, py in raw]
    d.polygon(pts, fill="#3f9e4d", outline="#2c6e35")
    d.line([(size / 2, size * 0.28), (size / 2, size * 0.70)], fill="#2c6e35", width=3)
    return img


def _gen_lanceolate(size=720):
    """披针形叶示意图：狭长、下部略宽，两端渐尖。"""
    img, d = _canvas(size)
    cx = size / 2
    top, bottom = size * 0.10, size * 0.86
    left, right = [], []
    for i in range(101):
        t = i / 100
        y = top + (bottom - top) * t
        width = size * 0.15 * math.sin(math.pi * t) * (0.65 + 0.5 * t)
        left.append((cx - width, y))
        right.append((cx + width, y))
    d.polygon(left + right[::-1], fill="#58b368", outline="#2c6e35")
    d.line([(cx, top), (cx, size * 0.93)], fill="#2c6e35", width=3)
    for t in (0.30, 0.43, 0.56, 0.69, 0.80):
        y = top + (bottom - top) * t
        vein_t = t - 0.09
        width = size * 0.15 * math.sin(math.pi * vein_t) * (0.65 + 0.5 * vein_t) * 0.85
        vein_y = top + (bottom - top) * vein_t
        for side in (-1, 1):
            d.line([(cx, y), (cx + side * width, vein_y)], fill="#2c6e35", width=2)
    return img


def _gen_ellipse(size=720):
    """椭圆形叶示意图。"""
    img, d = _canvas(size)
    cx, cy = size / 2, size * 0.5
    rx, ry = size * 0.33, size * 0.185
    d.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill="#3f9e4d", outline="#2c6e35")
    d.line([(cx - rx * 0.92, cy), (cx + rx * 0.92, cy)], fill="#2c6e35", width=3)
    for i in range(1, 6):
        x = cx - rx + 2 * rx * i / 6
        d.line([(x, cy), (x + rx * 0.15, cy - ry * 0.55)], fill="#2c6e35", width=1)
        d.line([(x, cy), (x + rx * 0.15, cy + ry * 0.55)], fill="#2c6e35", width=1)
    return img


def _gen_linear(size=720):
    """线形叶示意图。"""
    img, d = _canvas(size)
    x0, x1 = size * 0.10, size * 0.90
    cy = size * 0.5
    w_max = size * 0.022
    upper, lower = [], []
    n = 80
    for i in range(n + 1):
        t = i / n
        x = x0 + (x1 - x0) * t
        w = w_max * (math.sin(math.pi * t) ** 0.75)
        upper.append((x, cy - w))
        lower.append((x, cy + w))
    d.polygon(upper + lower[::-1], fill="#58b368", outline="#2c6e35")
    d.line([(x0, cy), (x1, cy)], fill="#2c6e35", width=2)
    return img


def make_sample_images():
    """确保 4 张样例示意图存在，返回 {类别: 绝对路径}。"""
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    gens = {"披针形": ("lanceolate.png", _gen_lanceolate),
            "椭圆形": ("ellipse.png", _gen_ellipse),
            "线形": ("linear.png", _gen_linear),
            "卵形": ("heart.png", _gen_heart)}
    paths = {}
    for cls, (fname, gen) in gens.items():
        p = os.path.join(SAMPLE_DIR, fname)
        if not os.path.exists(p):
            gen().save(p)
        paths[cls] = os.path.abspath(p)
    return paths


# ----------------------------------------------------------------------
# 真实模型推理
# ----------------------------------------------------------------------
def model_predict(image_path):
    result = predict_image(image_path, expected_classes=CLASSES)
    return [result['scores'][name] * 100 for name in CLASSES]


# ----------------------------------------------------------------------
# HTML 片段构建
# ----------------------------------------------------------------------
def left_head_html(state):
    """左卡（图片上传）头部：标题 + 状态点。"""
    if state == "empty":
        status = '<span class="dot dot-gray"></span><span>未上传图片</span>'
    elif state == "uploaded":
        status = '<span class="dot dot-green"></span><span>已加载图片</span>'
    elif state == "recognizing":
        status = '<span class="mini-spin"></span><span>正在识别…</span>'
    else:
        status = '<span class="dot dot-red"></span><span>识别失败</span>'
    return (f'<div class="panel-head"><div class="ph-title">{_icon("upload", "#8fd0ff", 21)}'
            f'<span>图片上传</span></div><div class="card-status">{status}</div></div>')


def right_head_html(state):
    """右卡（识别结果）头部与真实推理状态。"""
    if state == "empty":
        status = '<span class="dot dot-gray"></span><span>等待上传</span>'
    elif state == "uploaded":
        status = '<span class="dot dot-blue"></span><span>等待识别</span>'
    elif state == "recognizing":
        status = '<span class="mini-spin"></span><span>正在识别…</span>'
    elif state == "done":
        status = '<span class="dot dot-green"></span><span>识别完成</span>'
    else:
        status = '<span class="dot dot-red"></span><span>识别失败</span>'
    return (f'<div class="panel-head"><div class="ph-title">{_icon("image_search", "#8fd0ff", 21)}'
            f'<span>识别结果</span></div><div class="card-status">{status}</div></div>')


def zone_html(state, filename=""):
    """上传区域覆盖层：空状态显示提示与“选择图片”，已上传/识别中显示预览说明与扫描线。"""
    if state == "empty":
        return (
            '<div class="zone-prompt"><div class="zone-inner">'
            f'<div class="zone-icon">{_icon("cloud_upload", "#8fd0ff", 54, 1.6)}</div>'
            '<div class="zone-title">请上传叶片图片</div>'
            '<div class="zone-sub">拖拽图片至此，或点击选择</div>'
            '<button class="leaf-btn btn-light zone-select" onclick="leafPickFile()">选择图片</button>'
            '<div class="zone-hint">支持 JPG、JPEG、PNG</div>'
            '</div></div>'
        )
    scan = '<div class="scan-line"></div>' if state == "recognizing" else ""
    caption = "上传预览" if state == "uploaded" else "上传预览 · 识别中"
    return (
        '<div class="zone-prompt zone-uploaded">'
        f'<div class="zone-caption">{caption}</div>'
        f'{scan}'
        f'<div class="zone-filename">{_esc(filename)}</div>'
        '<div class="zone-hint zone-hint-bottom">拖拽图片至此，或点击选择 · 支持 JPG、JPEG、PNG</div>'
        '</div>'
    )


def scores_empty_html():
    """四类分数的空状态：空条 + “—”，不显示任何伪造数值。"""
    rows = "".join(
        f'<div class="score-row"><span class="score-name">{c}</span>'
        f'<div class="score-track"></div><span class="score-val">—</span></div>'
        for c in CLASSES
    )
    return f'<div class="score-list">{rows}</div>'


def result_empty_html():
    return (f'<div class="result-empty"><div class="big-ic">{_icon("image_search", "#8fb6e8", 64, 1.5)}</div>'
            '<h3>上传图片后查看识别结果</h3>'
            '<p>请选择叶片图片，或点击下方参考样例</p></div>'
            + scores_empty_html())


def result_uploaded_html():
    return (f'<div class="result-wait"><div class="rw-icon">{_icon("check_circle", "#8fd0ff", 44, 1.6)}</div>'
            '<h3>图片已上传，请点击开始识别</h3>'
            '<p>识别前可更换图片，或清除后重新选择</p></div>'
            + scores_empty_html())


def result_recognizing_html():
    return ('<div class="result-loading"><div class="spinner"></div>'
            '<h3>正在识别…</h3><p>模型正在处理图片，请稍候</p></div>'
            + scores_empty_html())


def result_error_html(message):
    return (f'<div class="result-error"><div class="err-ic">{_icon("alert", "#ff9a9a", 46, 1.8)}</div>'
            f'<h3>识别失败</h3><p>{_esc(message)}</p>'
            '<p class="err-sub">可更换图片后重试</p></div>'
            + scores_empty_html())


def result_done_html(percents):
    """识别完成：类别徽章 + 置信度 + 四条分数条（带动画）+ 参考特征。"""
    top_i = max(range(len(CLASSES)), key=lambda i: percents[i])
    top_name = CLASSES[top_i]
    top_conf = percents[top_i]
    rows = []
    for i, cname in enumerate(CLASSES):
        w = percents[i]
        top_cls = " top" if i == top_i else ""
        rows.append(
            f'<div class="score-row"><span class="score-name">{cname}</span>'
            f'<div class="score-track"><div class="score-fill{top_cls}" style="--w:{w}%"></div></div>'
            f'<span class="score-val">{w:.1f}%</span></div>'
        )
    return (
        '<div class="result-done">'
        '<div class="result-top">'
        f'<div class="rt-item"><span class="rt-label">识别类别：</span><span class="rt-badge">{top_name}</span></div>'
        '<div class="rt-sep"></div>'
        f'<div class="rt-item"><span class="rt-label">模型分数：</span><span class="rt-conf">{top_conf:.1f}%</span></div>'
        '</div>'
        '<div class="score-title">四类分数</div>'
        f'<div class="score-list">{"".join(rows)}</div>'
        '<div class="fb-note">分数为四类之间的相对得分，不代表识别正确概率；范围外图片也可能被分到其中一类。</div>'
        '<div class="feature-box">'
        f'<div class="fb-title">{_icon("info", "#8fd0ff", 16)}<span>该类别的参考特征</span></div>'
        f'<div class="fb-text">{CATEGORY_FEATURES[top_name]}</div>'
        '<div class="fb-note">基于类别的预设说明，非本次图像的逐项分析。</div>'
        '</div></div>'
    )


# ---------------- 模型评估页 ----------------
def stat_acc_html(acc=None):
    if acc is None:
        body = '<div class="stat-value v-wait">待评估</div><div class="stat-sub">完成测试后展示</div>'
    else:
        body = f'<div class="stat-value">{acc * 100:.1f}%</div><div class="stat-sub">基于独立测试集的整体表现</div>'
    return (f'<div class="stat-card"><div class="stat-head">{_icon("chart", "#8fd0ff", 20)}'
            f'<span>测试集准确率</span></div><div class="stat-body">{body}</div></div>')


def stat_num_html(n=None):
    if n is None:
        body = '<div class="stat-value v-num">—</div><div class="stat-sub">暂无测试记录</div>'
    else:
        body = f'<div class="stat-value v-num">{n}</div><div class="stat-sub">个测试样本</div>'
    return (f'<div class="stat-card"><div class="stat-head">{_icon("file", "#8fd0ff", 20)}'
            f'<span>测试样本数</span></div><div class="stat-body">{body}</div></div>')


def stat_status_html(done=False):
    if done:
        badge = '<span class="status-badge ok">已完成测试</span>'
        sub = "评估结果已导入"
    else:
        badge = '<span class="status-badge">未完成测试</span>'
        sub = "等待导入评估结果"
    return (f'<div class="stat-card"><div class="stat-head">{_icon("check_circle", "#8fd0ff", 20)}'
            f'<span>评估状态</span></div><div class="stat-body"><div class="stat-badge-wrap">{badge}</div>'
            f'<div class="stat-sub">{sub}</div></div></div>')


MIS_TABLE_HEAD = ('<table class="mis-table"><thead><tr>'
                  '<th style="width:17%">样例图片</th><th>真实类别</th><th>预测类别</th>'
                  '<th>分类置信度</th><th>原因记录</th></tr></thead>')


def mis_empty_html(title="暂无误判记录", sub="完成测试后展示真实误判案例"):
    return (f'<div class="mis-empty">'
            f'<div class="mis-empty-icon">{_icon("search_file", "#8fb6e8", 54, 1.4)}</div>'
            f'<h3>{_esc(title)}</h3><p>{_esc(sub)}</p>'
            f'{MIS_TABLE_HEAD}<tbody><tr><td colspan="5" class="mis-novalid">暂无数据</td></tr></tbody></table>'
            '</div>')


def mis_table_html(items):
    rows = []
    for it in items:
        conf = it.get("confidence", 0)
        conf_str = f"{conf * 100:.1f}%" if isinstance(conf, (int, float)) else "—"
        note = (it.get("image_note") or "").strip()
        thumb = (f'<div class="mis-thumb">{_icon("image", "#8fb6e8", 22, 1.6)}</div>'
                 + (f'<div class="mis-note-name" title="{_esc(note)}">{_esc(note)}</div>' if note else ""))
        reason = str(it.get("reason", "") or "—")
        rows.append(
            f'<tr><td>{thumb}</td>'
            f'<td>{_esc(it["true_label"])}</td>'
            f'<td>{_esc(it["pred_label"])}</td>'
            f'<td>{conf_str}</td>'
            f'<td class="mis-reason">{_esc(reason)}</td></tr>'
        )
    return f'{MIS_TABLE_HEAD}<tbody>{"".join(rows)}</tbody></table>'


def validate_eval_data(data):
    """校验评估 JSON 的必要字段，返回错误信息（None 表示通过）。"""
    if not isinstance(data, dict):
        return "评估文件内容必须是 JSON 对象（{...}）。"
    acc = data.get("test_accuracy")
    if (not isinstance(acc, (int, float))) or isinstance(acc, bool) or not (0 <= acc <= 1):
        return '字段 test_accuracy 缺失或无效：必须是 0~1 之间的小数（如 0.932）。'
    n = data.get("test_samples")
    if (not isinstance(n, int)) or isinstance(n, bool) or n < 0:
        return "字段 test_samples 缺失或无效：必须是不小于 0 的整数。"
    items = data.get("misclassified", [])
    if not isinstance(items, list):
        return "字段 misclassified 无效：必须是数组。"
    for i, it in enumerate(items):
        if not isinstance(it, dict):
            return f"第 {i + 1} 条误判记录必须是对象。"
        for key in ("true_label", "pred_label"):
            v = it.get(key)
            if not isinstance(v, str) or not v.strip():
                return f"第 {i + 1} 条误判记录缺少字段 {key}（类别文字）。"
        conf = it.get("confidence")
        if (not isinstance(conf, (int, float))) or isinstance(conf, bool) or not (0 <= conf <= 1):
            return f"第 {i + 1} 条误判记录的 confidence 无效：必须是 0~1 之间的小数。"
    return None


# ----------------------------------------------------------------------
# 自定义 CSS（深蓝主题还原）
# ----------------------------------------------------------------------
CSS = r'''
/* ============ 全局 ============ */
html, body { margin:0; padding:0; background:#071226; }
.gradio-container{
  max-width:1520px !important; margin:0 auto !important; padding:0 30px 26px !important;
  background:radial-gradient(1100px 620px at 50% -8%, rgba(38,76,140,.55), rgba(0,0,0,0) 62%), #0a1930 !important;
  font-family:"PingFang SC","Hiragino Sans GB","Microsoft YaHei","Noto Sans SC",sans-serif !important;
  color:#eaf2ff;
}
footer, .gradio-container footer{ display:none !important; }
.ic{ display:inline-block; vertical-align:middle; flex:0 0 auto; }
.page{ width:100%; margin-top:6px; }
#page-eval{ display:none; }

/* ============ 顶栏 ============ */
.app-header{ display:flex; align-items:center; justify-content:space-between; padding:24px 4px 14px; }
.brand{ display:flex; align-items:center; gap:14px; }
.logo{ width:40px; height:40px; filter:drop-shadow(0 0 8px rgba(143,208,255,.4)); }
.brand h1{ font-size:26px; font-weight:700; margin:0; color:#fff; letter-spacing:1.5px; }
.brand p{ margin:3px 0 0; font-size:14px; color:#93aed4; letter-spacing:.5px; }
.top-nav{ display:flex; align-items:center; gap:20px; font-size:16px; }
.top-nav a{ color:#b9cdea; text-decoration:none; cursor:pointer; padding:6px 2px; position:relative; letter-spacing:1.5px; }
.top-nav a:hover{ color:#e4f0ff; }
.top-nav a.active{ color:#fff; font-weight:600; }
.top-nav a.active::after{ content:""; position:absolute; left:12%; right:12%; bottom:-4px; height:3px; border-radius:2px;
  background:linear-gradient(90deg,#8fd0ff,#5aa9f0); }
.nav-sep{ color:#54688c; }

/* ============ 卡片通用 ============ */
.panel, .panel-upload, .panel-result, .samples-panel, .mis-card{
  background:linear-gradient(180deg, rgba(46,84,148,.32), rgba(22,45,88,.38));
  border:1px solid rgba(130,170,240,.22); border-radius:14px; padding:18px 20px;
  box-shadow:0 8px 24px rgba(2,8,20,.35); }
.panel-upload, .panel-result{ min-height:470px; }
.panel-head{ display:flex; align-items:center; justify-content:space-between; margin-bottom:6px; }
.ph-title{ display:flex; align-items:center; gap:10px; font-size:20px; font-weight:600; color:#fff; letter-spacing:1px; }
.card-status{ display:flex; align-items:center; gap:8px; font-size:14px; color:#9db4d8; }
.dot{ width:10px; height:10px; border-radius:50%; display:inline-block; }
.dot-gray{ background:#8ea6c9; }
.dot-green{ background:#41d98c; box-shadow:0 0 8px rgba(65,217,140,.8); }
.dot-blue{ background:#8fd0ff; box-shadow:0 0 8px rgba(143,208,255,.7); }
.dot-red{ background:#ff7a7a; box-shadow:0 0 8px rgba(255,122,122,.7); }

/* ============ 按钮 ============ */
.leaf-btn{ display:inline-flex !important; align-items:center; justify-content:center; gap:8px;
  border-radius:10px !important; padding:10px 18px !important; font-size:15px !important; font-weight:600 !important;
  letter-spacing:1px; cursor:pointer; border:1px solid transparent;
  transition:transform .15s ease, filter .15s ease, box-shadow .15s ease, background .15s ease; }
.leaf-btn:hover{ transform:translateY(-1px); filter:brightness(1.1); }
.leaf-btn:disabled{ opacity:.4 !important; cursor:not-allowed !important; transform:none !important; filter:none !important; }
.btn-primary{ background:linear-gradient(135deg,#9bdcff,#5aa9f0) !important; color:#0a2c52 !important; border:none !important;
  box-shadow:0 4px 14px rgba(90,169,240,.35); }
.btn-ghost{ background:rgba(120,160,220,.10) !important; color:#cfe2ff !important;
  border:1px solid rgba(140,180,240,.35) !important; }
.btn-ghost:hover:not(:disabled){ background:rgba(120,160,220,.18) !important; }
.btn-light{ background:linear-gradient(135deg,#a8dcff,#6cb6f5) !important; color:#0a2c52 !important; border:none !important;
  box-shadow:0 4px 12px rgba(90,169,240,.4); }
.btn-refresh::before{ content:""; width:15px; height:15px; flex:0 0 15px; background:__IC_ROTATE__ no-repeat center/contain; }
.btn-trash::before{ content:""; width:15px; height:15px; flex:0 0 15px; background:__IC_TRASH__ no-repeat center/contain; }
.zone-select::before{ content:""; width:15px; height:15px; flex:0 0 15px; background:__IC_FOLDER_DARK__ no-repeat center/contain; }
#btn-import, #btn-import button{ background:linear-gradient(135deg,#9bdcff,#5aa9f0) !important; color:#0a2c52 !important;
  border:none !important; border-radius:10px; font-weight:600; letter-spacing:1px;
  box-shadow:0 4px 14px rgba(90,169,240,.35); cursor:pointer; }
#btn-import::before, #btn-import button::before{ content:""; width:15px; height:15px; display:inline-block; margin-right:7px;
  vertical-align:-2px; background:__IC_UPLOAD_DARK__ no-repeat center/contain; }
.btn-row{ margin-top:14px; }
.btn-row > *{ flex:1; min-width:0 !important; }
.btn-row button{ width:100%; }

/* ============ 上传区 ============ */
#upload-zone{ position:relative; height:300px; margin-top:12px; }
#zone-overlay{ position:absolute !important; inset:0; height:100%; width:100%;
  overflow:visible !important; pointer-events:none; z-index:5; padding:0; }
#zone-overlay .html-container, #zone-overlay .prose{ height:100%; padding:0; }
#leaf-upload{ height:300px !important; margin:0 !important; }
#leaf-upload .image-container{ background:transparent !important; border:none !important; height:232px !important;
  display:flex; align-items:center; justify-content:center; }
#leaf-upload img{ object-fit:contain !important; max-height:226px; border-radius:8px; }
#leaf-upload .upload-container{ background:transparent !important; border:none !important; }
.zone-prompt{ position:absolute; inset:0; z-index:5; display:flex; flex-direction:column; align-items:center;
  justify-content:center; background:#102344; border:1.5px dashed rgba(150,190,255,.5);
  border-radius:12px; pointer-events:none; overflow:hidden; }
.zone-inner{ display:flex; flex-direction:column; align-items:center; gap:8px; }
.zone-title{ font-size:22px; font-weight:600; color:#fff; margin-top:6px; letter-spacing:1px; }
.zone-sub{ font-size:15px; color:#a9c2e6; }
.zone-select{ pointer-events:auto; margin-top:10px; }
.zone-hint{ font-size:13px; color:#7f9cc4; }
.zone-uploaded{ background:transparent; border-color:rgba(150,190,255,.45); }
.zone-caption{ position:absolute; top:8px; left:0; right:0; text-align:center; font-size:14px; color:#bcd3f2; }
.zone-filename{ position:absolute; bottom:32px; left:0; right:0; text-align:center; font-size:14px; color:#cfe2ff; }
.zone-hint-bottom{ position:absolute; bottom:10px; left:0; right:0; text-align:center; font-size:12.5px; }
.upload-tip{ margin-top:12px; text-align:center; font-size:13px; color:#7f9cc4; }
.upload-note{ margin-top:10px; text-align:center; font-size:13px; color:#7f9cc4; }

/* 扫描线动画（识别中，往返移动） */
.scan-line{ position:absolute; left:5%; right:5%; top:8%; height:3px; border-radius:2px;
  background:linear-gradient(90deg, rgba(140,220,255,0), #aee6ff 30%, #ffffff 50%, #aee6ff 70%, rgba(140,220,255,0));
  box-shadow:0 0 14px rgba(140,220,255,.9); pointer-events:none;
  animation:scanMove 1.5s ease-in-out infinite alternate; }
@keyframes scanMove{ from{ top:8%; } to{ top:90%; } }

/* ============ 加载动画 ============ */
.spinner{ width:46px; height:46px; margin:0 auto 16px; border-radius:50%;
  border:4px solid rgba(143,208,255,.18); border-top-color:#8fd0ff; animation:spin .9s linear infinite; }
.mini-spin{ width:14px; height:14px; border-radius:50%; display:inline-block;
  border:2px solid rgba(143,208,255,.25); border-top-color:#8fd0ff; animation:spin .8s linear infinite; }
@keyframes spin{ to{ transform:rotate(360deg); } }

/* ============ 结果区 ============ */
.result-empty, .result-wait, .result-loading, .result-error{ text-align:center; padding:24px 10px 8px; }
.result-empty h3, .result-wait h3, .result-loading h3, .result-error h3{
  font-size:20px; color:#fff; margin:14px 0 6px; font-weight:600; letter-spacing:1px; }
.result-empty p, .result-wait p, .result-loading p, .result-error p{ color:#8fa9cf; font-size:14px; margin:0; }
.result-error .err-sub{ margin-top:6px; color:#c3d4ee; }
.score-title{ font-size:15px; font-weight:600; color:#cfe2ff; margin:18px 0 8px; letter-spacing:1px; }
.score-list{ display:flex; flex-direction:column; gap:10px; }
.score-row{ display:flex; align-items:center; gap:12px; }
.score-name{ width:52px; font-size:14.5px; color:#dcebff; text-align:right; flex:0 0 52px; }
.score-track{ flex:1; height:14px; background:rgba(150,180,230,.10); border-radius:8px; overflow:hidden; }
.score-fill{ height:100%; width:var(--w); border-radius:8px; background:linear-gradient(90deg,#3f7fd0,#8fd0ff);
  animation:fillBar .8s cubic-bezier(.2,.7,.3,1) both; }
.score-fill.top{ background:linear-gradient(90deg,#6cb6f5,#a8dcff); box-shadow:0 0 10px rgba(140,210,255,.55); }
@keyframes fillBar{ from{ width:0; } to{ width:var(--w); } }
.score-val{ width:56px; flex:0 0 56px; font-size:14.5px; color:#cfe2ff; text-align:right; font-variant-numeric:tabular-nums; }

.result-done{ animation:fadeUp .45s ease both; }
@keyframes fadeUp{ from{ opacity:0; transform:translateY(8px); } to{ opacity:1; transform:translateY(0); } }
.result-top{ display:flex; align-items:center; gap:16px; padding:4px 0 2px; }
.rt-item{ display:flex; align-items:center; gap:8px; }
.rt-label{ font-size:15px; color:#bcd3f2; }
.rt-badge{ background:linear-gradient(135deg,#9bdcff,#5aa9f0); color:#0a2c52; font-weight:700; font-size:17px;
  padding:4px 18px; border-radius:8px; letter-spacing:2px; }
.rt-conf{ font-size:34px; font-weight:800; color:#8fd0ff; letter-spacing:1px; font-variant-numeric:tabular-nums; }
.rt-sep{ width:1px; height:34px; background:rgba(140,180,240,.25); }
.feature-box{ margin-top:18px; background:rgba(80,130,200,.12); border:1px solid rgba(140,180,240,.22);
  border-radius:10px; padding:14px 16px; }
.fb-title{ display:flex; align-items:center; gap:8px; font-size:15px; font-weight:600; color:#fff; }
.fb-text{ margin-top:8px; font-size:14.5px; color:#dcebff; }
.fb-note{ margin-top:8px; font-size:12.5px; color:#8fa9cf; }

/* ============ 参考样例 ============ */
.samples-panel{ margin-top:18px; }
.samples-head{ display:flex; justify-content:space-between; align-items:center; }
.samples-hint{ font-size:14px; color:#9db4d8; }
.samples-grid{ display:grid !important; grid-template-columns:repeat(4, 1fr); gap:16px; margin-top:14px; }
.sample-card{ background:rgba(20,42,82,.45); border:1px solid rgba(130,170,240,.20); border-radius:12px;
  padding:10px 10px 12px; cursor:pointer; position:relative; min-width:0 !important;
  transition:transform .18s ease, border-color .18s ease, box-shadow .18s ease; }
.sample-card:hover{ transform:translateY(-3px); border-color:rgba(143,208,255,.55); box-shadow:0 10px 22px rgba(3,10,25,.45); }
.sample-tag{ display:inline-block; background:rgba(10,24,48,.78);
  border:1px solid rgba(143,208,255,.4); color:#a8d4ff; font-size:11px; padding:2px 8px;
  border-radius:6px; letter-spacing:1px; }
.sample-img{ background:#f2f6fb; border-radius:8px; }
.sample-img img{ object-fit:contain !important; }
.sample-foot{ display:flex; align-items:center; justify-content:space-between; margin-top:10px; }
.sample-foot > div:first-child{ flex:1; }
.sample-label{ font-size:16.5px; font-weight:600; color:#fff; letter-spacing:2px; }
.sample-use{ padding:6px 12px !important; font-size:13px !important; border-radius:8px !important; }
.sample-use::after{ content:"›"; font-size:15px; line-height:1; }

/* ============ 模型评估页 ============ */
.eval-head{ padding:6px 4px 0; }
.back-link{ display:inline-flex; align-items:center; gap:6px; color:#8fb6e8; font-size:14.5px; cursor:pointer; }
.back-link:hover{ color:#e4f0ff; }
.eval-title{ display:flex; align-items:center; gap:12px; margin-top:12px; }
.eval-title .bar{ width:5px; height:30px; border-radius:3px; background:linear-gradient(180deg,#8fd0ff,#5aa9f0); }
.eval-title h2{ font-size:26px; font-weight:700; color:#fff; margin:0; letter-spacing:2px; }
.eval-sub{ color:#93aed4; font-size:14.5px; margin:6px 0 0 17px; }
.eval-stats{ display:grid !important; grid-template-columns:repeat(3, 1fr); gap:16px; margin-top:16px; }
.stat-card{ background:linear-gradient(180deg, rgba(46,84,148,.32), rgba(22,45,88,.38));
  border:1px solid rgba(130,170,240,.22); border-radius:12px; overflow:hidden; }
.stat-head{ display:flex; align-items:center; gap:10px; padding:12px 16px; font-size:15.5px; font-weight:600;
  color:#fff; letter-spacing:1px; border-bottom:1px solid rgba(130,170,240,.16); background:rgba(30,60,110,.25); }
.stat-body{ padding:20px 16px 22px; text-align:center; }
.stat-value{ font-size:34px; font-weight:800; color:#8fd0ff; letter-spacing:1px; }
.stat-value.v-wait{ color:#8fb6e8; font-size:30px; letter-spacing:8px; }
.stat-value.v-num{ font-size:30px; }
.stat-sub{ margin-top:6px; font-size:13px; color:#8fa9cf; }
.stat-badge-wrap{ margin-bottom:4px; }
.status-badge{ display:inline-block; padding:6px 18px; border-radius:8px; font-size:15px; font-weight:600;
  border:1px solid rgba(140,180,240,.4); color:#cfe2ff; background:rgba(120,160,220,.10); letter-spacing:1px; }
.status-badge.ok{ color:#7fe3ae; border-color:rgba(90,220,150,.5); background:rgba(60,200,120,.10); }

.mis-card{ margin-top:16px; }
.mis-head{ display:flex; align-items:center; gap:10px; font-size:20px; font-weight:600; color:#fff;
  letter-spacing:1px; padding-bottom:12px; border-bottom:1px solid rgba(130,170,240,.16); }
.mis-empty{ text-align:center; padding:28px 0 16px; }
.mis-empty h3{ font-size:19px; color:#fff; margin:12px 0 4px; font-weight:600; letter-spacing:1px; }
.mis-empty p{ color:#8fa9cf; font-size:14px; margin:0 0 20px; }
.mis-table{ width:100%; border-collapse:separate; border-spacing:0; border:1px solid rgba(130,170,240,.22);
  border-radius:10px; overflow:hidden; font-size:14px; }
.mis-table th{ background:rgba(60,100,170,.25); color:#dceaff; font-weight:600; padding:10px 12px;
  text-align:center; border-bottom:1px solid rgba(130,170,240,.22); letter-spacing:1px; }
.mis-table td{ padding:12px; text-align:center; color:#d5e5fb; border-bottom:1px solid rgba(130,170,240,.10);
  vertical-align:middle; }
.mis-table tr:last-child td{ border-bottom:none; }
.mis-novalid{ color:#8fa9cf !important; }
.mis-thumb{ width:46px; height:46px; margin:0 auto; border-radius:8px; background:rgba(120,160,220,.12);
  border:1px solid rgba(130,170,240,.25); display:flex; align-items:center; justify-content:center; }
.mis-note-name{ font-size:12px; color:#8fa9cf; margin-top:4px; max-width:120px; margin-left:auto; margin-right:auto;
  overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.mis-reason{ max-width:340px; }
.mis-foot{ display:flex; align-items:center; justify-content:space-between; gap:16px; margin-top:16px;
  padding-top:14px; border-top:1px solid rgba(130,170,240,.16); }
.note-inline{ display:flex; align-items:center; gap:10px; font-size:13.5px; color:#9db4d8; }
.note-inline .divider{ width:1px; height:16px; background:rgba(140,180,240,.3); }
.eval-alert{ margin-top:14px; padding:10px 14px; border-radius:8px; font-size:14px; }
.eval-alert.error{ background:rgba(255,110,110,.12); border:1px solid rgba(255,120,120,.4); color:#ffc9c9; }
.eval-alert.ok{ background:rgba(80,220,140,.10); border:1px solid rgba(90,220,150,.35); color:#b9f4d2; }

/* ============ 页脚 ============ */
.app-footer{ margin-top:26px; padding:16px 0 6px; border-top:1px solid rgba(130,170,240,.14);
  text-align:center; color:#7f9cc4; font-size:14px; letter-spacing:2px; }

/* ============ 响应式 ============ */
@media (max-width:1100px){ .samples-grid{ grid-template-columns:repeat(2, 1fr) !important; } }
@media (max-width:980px){
  .two-col{ flex-direction:column !important; }
  .eval-stats{ grid-template-columns:1fr !important; }
  .panel-upload, .panel-result{ min-height:0; }
}
@media (max-width:760px){
  .app-header{ flex-direction:column; gap:12px; align-items:flex-start; }
  .brand h1{ font-size:20px; }
  .samples-grid{ grid-template-columns:1fr !important; }
}
@media (max-width:640px){ .samples-grid{ grid-template-columns:1fr !important; } }

/* 尊重系统“减少动态效果”设置 */
@media (prefers-reduced-motion: reduce){
  .scan-line, .spinner, .mini-spin, .score-fill, .result-done{ animation:none !important; }
  .score-fill{ width:var(--w) !important; }
  .leaf-btn, .sample-card{ transition:none !important; }
  .leaf-btn:hover, .sample-card:hover{ transform:none !important; }
}
'''

# 用统一的 data-URI 图标替换 CSS 占位符
CSS = (CSS
       .replace("__IC_ROTATE__", _data_icon("rotate", "#cfe2ff"))
       .replace("__IC_TRASH__", _data_icon("trash", "#cfe2ff"))
       .replace("__IC_FOLDER_DARK__", _data_icon("folder", "#0a2c52"))
       .replace("__IC_UPLOAD_DARK__", _data_icon("upload", "#0a2c52")))

# ----------------------------------------------------------------------
# 自定义 JS：顶部导航切换、“选择图片”触发、样例整卡可点击
# ----------------------------------------------------------------------
JS = r'''
() => {
function leafShow(el){ el.style.display = 'flex'; el.style.flexDirection = 'column'; }

window.leafGo = function(target){
  var rec = document.getElementById('page-recognize');
  var ev  = document.getElementById('page-eval');
  var links = document.querySelectorAll('.top-nav a');
  for (var i = 0; i < links.length; i++){ links[i].classList.remove('active'); }
  function mark(name){
    var l = document.querySelector('.top-nav a[data-nav="' + name + '"]');
    if (l){ l.classList.add('active'); }
  }
  if (target === 'eval'){
    if (rec){ rec.style.display = 'none'; }
    if (ev){ leafShow(ev); }
    mark('eval');
    window.scrollTo({top: 0, behavior: 'smooth'});
  } else {
    if (ev){ ev.style.display = 'none'; }
    if (rec){ leafShow(rec); }
    mark('recognize');
    window.scrollTo({top: 0, behavior: 'smooth'});
  }
};

window.leafPickFile = function(){
  var zone = document.getElementById('leaf-upload');
  var input = zone ? zone.querySelector('input[type="file"]') : null;
  if (!input){ input = document.querySelector('#page-recognize input[type="file"]'); }
  if (input){ input.click(); }
};

document.addEventListener('click', function(e){
  var t = e.target;
  var a = t && t.closest ? t.closest('.top-nav a[data-nav]') : null;
  if (a){ e.preventDefault(); window.leafGo(a.getAttribute('data-nav')); return; }
  var back = t && t.closest ? t.closest('.back-link') : null;
  if (back){ e.preventDefault(); window.leafGo('recognize'); }
});

document.querySelectorAll('.sample-card').forEach(function(card){
  card.addEventListener('click', function(e){
    if (e.target.closest && e.target.closest('.sample-use')){ return; }
    var btn = card.querySelector('.sample-use');
    if (btn){ btn.click(); }
  });
});
}
'''

# ----------------------------------------------------------------------
# 事件处理函数
# ----------------------------------------------------------------------
def bump_state(state):
    s = dict(state) if isinstance(state, dict) else {}
    s["image_id"] = int(s.get("image_id", 0)) + 1
    return s


def ready_updates(fname, state):
    """图片就绪（已上传、未识别）时，左右两卡的更新项。"""
    return [left_head_html("uploaded"),
            zone_html("uploaded", fname),
            gr.update(value="▶ 开始识别", interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=True),
            right_head_html("uploaded"),
            result_uploaded_html(),
            bump_state(state)]


def clear_updates(state):
    """恢复到初始（未上传）状态。"""
    return [left_head_html("empty"),
            zone_html("empty"),
            gr.update(value="▶ 开始识别", interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            right_head_html("empty"),
            result_empty_html(),
            bump_state(state)]


def error_updates(fname, state, message):
    """识别失败：恢复按钮、显示中文错误提示。"""
    return [left_head_html("uploaded"),
            zone_html("uploaded", fname),
            gr.update(value="▶ 开始识别", interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=True),
            right_head_html("error"),
            result_error_html(message),
            state]


def on_upload_changed(path, state):
    """上传组件值变化（含拖拽、点击选择）。"""
    if path is None:
        return clear_updates(state)
    fname = os.path.basename(path) or "已上传图片"
    ext = os.path.splitext(fname)[1].lower()
    if ext and ext not in ALLOWED_EXTS:
        return [left_head_html("empty"),
                zone_html("empty"),
                gr.update(value="▶ 开始识别", interactive=False),
                gr.update(interactive=True),
                gr.update(interactive=True),
                right_head_html("error"),
                result_error_html("仅支持 JPG、JPEG、PNG 格式的图片，请重新选择。"),
                bump_state(state)]
    return ready_updates(fname, state)


def on_sample_click(state, *, path):
    """点击样例卡片 / “使用样例”：填入上传区并清空旧结果。"""
    return ready_updates(os.path.basename(path), state) + [gr.update(value=path)]


def on_clear(state):
    """清除：恢复初始状态，同时清空图片、文件名、结果和分数。"""
    return clear_updates(state) + [gr.update(value=None)]


def on_start(image, state):
    """开始识别（生成器）：识别中 → 真实推理 → 完成 / 失败。"""
    fname = os.path.basename(image) if image else ""
    # 前置校验：未上传 / 格式不支持 / 文件损坏
    if image is None or not os.path.exists(image):
        yield error_updates(fname, state, "未检测到有效图片，请重新上传后再试。")
        return
    ext = os.path.splitext(image)[1].lower()
    if ext and ext not in ALLOWED_EXTS:
        yield error_updates(fname, state, "仅支持 JPG、JPEG、PNG 格式的图片。")
        return
    try:
        with PILImage.open(image) as im:
            im.verify()  # 校验图片可读取，损坏文件会抛出异常
    except Exception:
        yield error_updates(fname, state, "图片文件读取失败，文件可能已损坏，请更换图片。")
        return

    # 状态 1：正在识别（扫描线、旋转加载、按钮防重复点击）
    yield [left_head_html("recognizing"),
           zone_html("recognizing", fname),
           gr.update(value="◌ 正在识别…", interactive=False),
           gr.update(interactive=True),
           gr.update(interactive=True),
           right_head_html("recognizing"),
           result_recognizing_html(),
           state]

    try:
        percents = model_predict(image)
    except Exception as error:
        yield error_updates(fname, state, f"模型识别失败：{error}")
        return

    # 展示模型实际输出，按类别名称映射到界面。
    yield [left_head_html("uploaded"),
           zone_html("uploaded", fname),
           gr.update(value="▶ 开始识别", interactive=True),
           gr.update(interactive=True),
           gr.update(interactive=True),
           right_head_html("done"),
           result_done_html(percents),
           state]


def _eval_error(msg):
    """评估导入失败：保留原有卡片内容，仅显示中文错误提示。"""
    return [gr.update(), gr.update(), gr.update(), gr.update(),
            f'<div class="eval-alert error">{_esc(msg)}</div>']


def on_eval_import(file):
    """导入评估 JSON：解析、校验必要字段、刷新统计卡片与误判表格。"""
    if file is None:
        return [gr.update(), gr.update(), gr.update(), gr.update(), "<div></div>"]

    path = file[0] if isinstance(file, (list, tuple)) else getattr(file, "name", file)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except UnicodeDecodeError:
        return _eval_error("评估文件必须是 UTF-8 编码的文本文件。")
    except json.JSONDecodeError as e:
        return _eval_error(f"JSON 解析失败：{e}")
    except Exception as e:
        return _eval_error(f"读取评估文件失败：{e}")

    msg = validate_eval_data(data)
    if msg:
        return _eval_error(msg)

    acc = data["test_accuracy"]
    n = data["test_samples"]
    items = data.get("misclassified", [])
    body = mis_table_html(items) if items else mis_empty_html("暂无误判记录", "本次评估未记录误判案例")
    ok_msg = (f'<div class="eval-alert ok">导入成功：测试样本数 {n}，'
              f'测试集准确率 {acc * 100:.1f}%。</div>')
    return [stat_acc_html(acc), stat_num_html(n), stat_status_html(True), body, ok_msg]


# ----------------------------------------------------------------------
# 构建界面
# ----------------------------------------------------------------------
HEADER_HTML = (
    '<header class="app-header">'
    '<div class="brand">'
    '<svg class="logo" viewBox="0 0 24 24" fill="none" stroke="#8fd0ff" stroke-width="1.7" '
    'stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M4 20C4 10 10 4 20 4c0 10-6 16-16 16z" fill="rgba(143,208,255,.14)"/>'
    '<path d="M4 20C4 10 10 4 20 4c0 10-6 16-16 16z"/>'
    '<path d="M4 20C9 15 13 11 20 4"/></svg>'
    '<div class="brand-text"><h1>基于深度学习的叶形分类系统</h1>'
    f'<p>识别{"、".join(CLASSES)}{len(CLASSES)}类叶片</p></div></div>'
    '<nav class="top-nav">'
    '<a data-nav="recognize" class="active">叶片识别</a><span class="nav-sep">/</span>'
    '<a data-nav="eval">模型评估</a>'
    '</nav></header>'
)

TIP_HTML = '<div class="upload-tip">建议上传背景简单、轮廓完整的单片叶片。</div>'
NOTE_HTML = '<div class="upload-note">更换图片将清空上次结果</div>'

EVAL_HEAD_HTML = (
    '<div class="eval-head">'
    f'<a class="back-link">{_icon("arrow_left", "#8fb6e8", 16)} 返回叶片识别</a>'
    '<div class="eval-title"><span class="bar"></span><h2>模型评估</h2></div>'
    '<p class="eval-sub">基于独立测试集的整体性能评估</p></div>'
)

MIS_HEAD_HTML = f'<div class="mis-head">{_icon("file", "#8fd0ff", 20)}<span>误判案例与原因记录</span></div>'
MIS_NOTE_HTML = (f'<div class="note-inline">{_icon("info", "#8fd0ff", 18)}'
                 '<span class="divider"></span>'
                 '<span>误判原因由人工复核后记录，实验结论以实际测试结果为准。</span></div>')

FOOTER_HTML = '<div class="app-footer">ResNet 迁移学习 · 植物形态识别</div>'


def build_app():
    # Gradio 5.12 在构建队列时需要当前线程存在事件循环。
    # Python 3.14 不再隐式创建该循环，需在首次构建界面前初始化。
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())
    # 首次运行时自动生成 4 张样例示意图
    sample_paths = make_sample_images()

    with gr.Blocks(css=CSS, js=JS, title="基于深度学习的叶形分类系统",
                   theme=gr.themes.Base()) as demo:
        gr.HTML(HEADER_HTML)
        state = gr.State({"image_id": 0})

        # ================= 页面一：叶片识别 =================
        with gr.Column(elem_id="page-recognize", elem_classes=["page"]):
            with gr.Row(elem_classes=["two-col"]):
                # ---- 左卡：图片上传 ----
                with gr.Column(elem_classes=["panel", "panel-upload"]):
                    left_head = gr.HTML(left_head_html("empty"))
                    with gr.Column(elem_id="upload-zone"):
                        upload_img = gr.Image(
                            label=None, show_label=False, type="filepath",
                            sources=["upload"], container=False, interactive=True,
                            height=300, show_download_button=False, show_fullscreen_button=False,
                            elem_id="leaf-upload")
                        zone_overlay = gr.HTML(zone_html("empty"), elem_id="zone-overlay")
                    gr.HTML(TIP_HTML)
                    with gr.Row(elem_classes=["btn-row"]):
                        btn_start = gr.Button("▶ 开始识别", elem_id="btn-start",
                                              elem_classes=["leaf-btn", "btn-primary"],
                                              interactive=False)
                        btn_replace = gr.Button("更换图片", elem_id="btn-replace",
                                                elem_classes=["leaf-btn", "btn-ghost", "btn-refresh"],
                                                interactive=False)
                        btn_clear = gr.Button("清除", elem_id="btn-clear",
                                              elem_classes=["leaf-btn", "btn-ghost", "btn-trash"],
                                              interactive=False)
                    gr.HTML(NOTE_HTML)

                # ---- 右卡：识别结果 ----
                with gr.Column(elem_classes=["panel", "panel-result"]):
                    right_head = gr.HTML(right_head_html("empty"))
                    result_html = gr.HTML(result_empty_html())

            # ---- 参考样例 ----
            with gr.Column(elem_id="samples-anchor", elem_classes=["panel", "samples-panel"]):
                gr.HTML(f'<div class="samples-head"><div class="ph-title">{_icon("layers", "#8fd0ff", 20)}'
                        f'<span>四类叶片参考样例</span></div>'
                        f'<div class="samples-hint">点击样例可载入上传区</div></div>')
                sample_btns = []
                with gr.Row(elem_classes=["samples-grid"]):
                    for cls in CLASSES:
                        with gr.Column(elem_classes=["sample-card"]):
                            gr.HTML('<div class="sample-tag">示意图</div>')
                            gr.Image(value=sample_paths[cls], label=None, show_label=False,
                                     container=False, interactive=False, height=170,
                                     show_download_button=False, show_fullscreen_button=False,
                                     elem_classes=["sample-img"])
                            with gr.Row(elem_classes=["sample-foot"]):
                                gr.HTML(f'<div class="sample-label">{cls}</div>')
                                b = gr.Button("使用样例", elem_classes=["sample-use", "leaf-btn", "btn-ghost"])
                                sample_btns.append(b)

        # ================= 页面二：模型评估 =================
        with gr.Column(elem_id="page-eval", elem_classes=["page"]):
            gr.HTML(EVAL_HEAD_HTML)
            with gr.Row(elem_classes=["eval-stats"]):
                stat_acc = gr.HTML(stat_acc_html())
                stat_num = gr.HTML(stat_num_html())
                stat_status = gr.HTML(stat_status_html())
            with gr.Column(elem_classes=["panel", "mis-card"]):
                gr.HTML(MIS_HEAD_HTML)
                mis_body = gr.HTML(mis_empty_html())
                eval_msg = gr.HTML("<div></div>")
                with gr.Row(elem_classes=["mis-foot"]):
                    gr.HTML(MIS_NOTE_HTML)
                    btn_import = gr.UploadButton("导入评估结果", file_types=[".json"],
                                                 elem_id="btn-import",
                                                 elem_classes=["leaf-btn", "btn-primary", "btn-import"])

        gr.HTML(FOOTER_HTML)

        # ================= 事件绑定 =================
        recog_outputs = [left_head, zone_overlay, btn_start, btn_replace, btn_clear,
                         right_head, result_html, state]

        # 上传（拖拽 / 点击选择）后：显示预览与文件名，启用按钮，清空旧结果
        upload_img.change(fn=on_upload_changed, inputs=[upload_img, state],
                          outputs=recog_outputs, api_name=False)

        # 点击样例：填入上传区（含 Image 值更新），清空旧结果
        for cls, b in zip(CLASSES, sample_btns):
            b.click(fn=functools.partial(on_sample_click, path=sample_paths[cls]),
                    inputs=[state],
                    outputs=recog_outputs + [upload_img], api_name=False)

        # 更换图片：打开文件选择框（与“选择图片”共用同一入口）
        btn_replace.click(fn=None, js="() => leafPickFile()", api_name=False)

        # 清除：恢复初始状态
        btn_clear.click(fn=on_clear, inputs=[state],
                        outputs=recog_outputs + [upload_img], api_name=False)

        # 开始识别：生成器实现 识别中 → 完成 / 失败 的状态切换
        btn_start.click(fn=on_start, inputs=[upload_img, state],
                        outputs=recog_outputs, api_name=False)

        # 模型评估：导入评估 JSON
        btn_import.upload(fn=on_eval_import, inputs=[btn_import],
                          outputs=[stat_acc, stat_num, stat_status, mis_body, eval_msg],
                          api_name=False)

    return demo


if __name__ == "__main__":
    app = build_app()
    app.queue()
    # 手动运行 python app.py 后自动打开浏览器
    app.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True, show_error=False)
