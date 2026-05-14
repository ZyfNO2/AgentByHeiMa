# LangChain 学习笔记 — 设计系统规范（Light Theme）

> 基于 Web Design Engineer Skill | 亮色主题 | 2026-05-13

---

## Design Decisions

### Color Palette (Light Theme)

```css
--bg-page:       #f8f9fc;      /* 页面底色 — 冷白 */
--bg-surface:    #ffffff;      /* 卡片/容器 — 纯白 */
--bg-elevated:   #ffffff;      /* 悬浮卡片 */
--bg-subtle:     #f1f3f8;      /* 次级背景/代码块 */
--bg-accent:     #eef2ff;      /* 强调色浅底 */

--text-primary:   #1a1d26;     /* 主文字 — 深墨黑 */
--text-secondary: #5a6072;     /* 次要文字 — 中灰蓝 */
--text-muted:     #949aad;      /* 弱化文字 — 浅灰 */
--text-on-accent: #ffffff;      /* 强调色上的文字 */

--primary:        oklch(0.55 0.22 250);    /* 主色 — 靛蓝 #4F6EF7 */
--primary-light:  oklch(0.72 0.15 250);    /* 主色浅 */
--primary-bg:     oklch(0.95 0.05 250);    /* 主色极浅底 */
--accent:         oklch(0.70 0.18 85);     /* 强调色 — 琥珀橙 #F59E0B */
--accent-light:   oklch(0.85 0.12 85);
--accent-bg:      oklch(0.97 0.04 85);
--success:        oklch(0.55 0.16 155);     /* 成功 — 翠绿 */
--danger:         oklch(0.55 0.20 25);      /* 危险 — 珊瑚红 */
--info:           oklch(0.55 0.15 230);     /* 信息 — 天蓝 */

--border:         rgba(26,29,38,0.08);      /* 边框 */
--border-strong:  rgba(26,29,38,0.14);
--shadow-sm:      0 1px 3px rgba(0,0,0,0.04), 0 1px 2px rgba(0,0,0,0.02);
--shadow-md:      0 4px 12px rgba(0,0,0,0.06), 0 1px 3px rgba(0,0,0,0.04);
--shadow-lg:      0 12px 32px rgba(0,0,0,0.08), 0 4px 12px rgba(0,0,0,0.04);
--shadow-glow:    0 0 30px rgba(79,110,247,0.10);
```

### Typography

| 用途 | 字体 | 字重 | 大小 |
|------|------|------|------|
| **页面标题 H1** | Space Grotesk | 800 | clamp(28px, 4vw, 44px) |
| **章节标题 H2** | Space Grotesk | 700 | clamp(20px, 2.5vw, 30px) |
| **小节标题 H3** | Space Grotesk | 600 | 18px |
| **正文** | Outfit | 400 | 16px |
| **代码/公式** | JetBrains Mono | 400 | 14px |
| **标注/标签** | Space Grotesk | 500 | 13px |

### Spacing System

- 基础单位: **4px**
- 常用间距: 8 / 12 / 16 / 24 / 32 / 48 / 64px
- 容器内边距: 40px (desktop) / 24px (mobile)
- 卡片内边距: 24px
- 段落间距: 16px

### Border Radius

| 元素 | 圆角 |
|------|------|
| 按钮/标签 | 8px |
| 卡片 | 16px |
| 输入框/代码块 | 12px |
| 徽章/Pill | 100px (全圆) |
| 弹窗 | 20px |

### Motion

- 入场动画: `fadeUp` 0.5s ease-out
- Hover 过渡: `transform 0.25s ease`, `box-shadow 0.25s ease`
- Easing: `cubic-bezier(0.25, 0.46, 0.45, 0.94)` (ease-out-cubic)

---

## Page Layout (每集笔记页)

```
┌──────────────────────────────────────────────┐
│  Header: Logo + 导航面包屑 + 集数标签        │
├──────────────────────────────────────────────┤
│                                              │
│  Hero: 标题 + 副标题 + 视频信息条            │
│                                              │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐       │
│  │ 章节 1   │ │ 章节 2   │ │ 章节 3   │       │
│  │ Card    │ │ Card    │ │ Card    │       │
│  └─────────┘ └─────────┘ └─────────┘       │
│                                              │
│  代码示例区域 (带语法高亮)                    │
│                                              │
│  关键知识点高亮框                             │
│                                              │
│  Footer: 上下集导航 + 参考来源               │
│                                              │
└──────────────────────────────────────────────┘
```

---

## 组件规范

### 1. 章节卡片 (Section Card)
- 白底 + 微阴影 + 16px圆角
- 左侧 4px 色带区分类型（primary/accent/success/info）
- hover 时阴影加深 + translateY(-2px)

### 2. 代码块 (Code Block)
- 背景 `#f1f3f8`
- JetBrains Mono 14px
- 顶部有语言标签 + 复制按钮
- 关键字着色：primary色

### 3. 对比表格 (Comparison Table)
- 斑马纹行（偶数行浅灰底）
- 表头：primary色背景白字
- hover 行高亮

### 4. 知识点高亮 (Key Point)
- primary浅底 + 左侧竖线 + 图标
- 三种变体：💡 Tips / ⚠️ Warning / ✅ Key Concept

### 5. 导航栏 (Navigation)
- 固定顶部，半透明毛玻璃效果
- 当前章节高亮
- 上一集/下一集按钮

### 6. Tweaks Panel
- 右下角浮动面板
- 可调：字体大小、行距、主题切换(预留暗色)

---

## 文件命名规则

```
LangChain_Notes/
├── index.html              ← 总览导航页
├── 13-认识LangChain.html   ← 每集独立HTML
├── 14-快速入门和Agent原理.html
├── 15-初始化模型.html
├── ... (共15个文件)
└── design-system.md        ← 本设计规范
```

---

## Narrative Role

- **角色**: 技术学习笔记设计师
- **受众**: 正在学 LangChain 的开发者/学生
- **视觉温度**: 清爽、专业、易读、现代感
- **参考风格**: Linear.app 文档风 + Notion 学习笔记风
