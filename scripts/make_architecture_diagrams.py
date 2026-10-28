"""
Generates publication-quality architecture diagrams and flowcharts for Adaptive-RAG:
1. diag1_end_to_end_framework.png: 4-Stage System Architecture Map
2. diag2_runtime_decision_flowchart.png: Detailed Runtime Decision Engine Flowchart
3. diag3_evidence_state_features.png: 7-Dimensional Evidence State Mathematical Framework
4. diag4_distractor_penalty_mechanism.png: Context Bloat & Distractor Shielding Mechanism
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

os.makedirs("figures", exist_ok=True)

# Styling
C_NAVY = "#1e3a8a"
C_BLUE = "#2563eb"
C_BLUE_LT = "#eff6ff"
C_GREEN = "#16a34a"
C_GREEN_LT = "#f0fdf4"
C_RED = "#dc2626"
C_RED_LT = "#fef2f2"
C_AMBER = "#d97706"
C_AMBER_LT = "#fffbeb"
C_SLATE = "#0f172a"
C_GRAY_LT = "#f8fafc"
C_BORDER = "#cbd5e1"


def draw_box(ax, x, y, w, h, title, subtitle="", bg_color=C_BLUE_LT, border_color=C_BLUE, title_color=C_NAVY, title_size=10, sub_size=8, corner=0.03):
    box = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.01,rounding_size={corner}",
        facecolor=bg_color,
        edgecolor=border_color,
        linewidth=1.2,
        zorder=2
    )
    ax.add_patch(box)
    if subtitle:
        ax.text(x + w / 2, y + h * 0.65, title, ha="center", va="center", fontsize=title_size, fontweight="bold", color=title_color, zorder=3)
        ax.text(x + w / 2, y + h * 0.30, subtitle, ha="center", va="center", fontsize=sub_size, color="#334155", zorder=3)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=title_size, fontweight="bold", color=title_color, zorder=3)


def draw_arrow(ax, x1, y1, x2, y2, color="#475569", lw=1.5, label=""):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=14, shrinkA=0, shrinkB=0),
        zorder=4
    )
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.02, label, ha="center", va="bottom", fontsize=8, fontweight="bold", color=color, zorder=5)


# ══════════════════════════════════════════════════════════════════════════════
# Diagram 1: 4-Stage End-to-End System Architecture Framework
# ══════════════════════════════════════════════════════════════════════════════
def make_diag1():
    fig, ax = plt.subplots(figsize=(11, 4.5), dpi=300)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4.5)
    ax.axis("off")

    ax.text(5.5, 4.2, "ADAPTIVE-RAG : 4-STAGE END-TO-END SYSTEM ARCHITECTURE", ha="center", va="center", fontsize=13, fontweight="bold", color=C_NAVY)
    ax.text(5.5, 3.85, "Seamless transition from offline index preparation & supervision to real-time inference & evaluation", ha="center", va="center", fontsize=9, color="#64748b")

    # 4 Columns / Stages
    stages = [
        (0.3, 0.5, 2.3, 3.0, "STAGE 1", "Ingestion & Indexing", C_BLUE_LT, C_BLUE, [
            "HotpotQA Multi-Hop JSON",
            "SHA-256 Document Hashing",
            "MiniLM Vector Embeddings",
            "FAISS IndexFlatIP Store"
        ]),
        (2.95, 0.5, 2.3, 3.0, "STAGE 2", "Offline Supervision", C_AMBER_LT, C_AMBER, [
            "Run B2 (k=5) & B3 (k=10)",
            "Compute ΔF1 = F1@10 - F1@5",
            "Threshold delta=0.05",
            "Train GBDT Classifier"
        ]),
        (5.6, 0.5, 2.5, 3.0, "STAGE 3", "Real-Time Adaptive QA", C_GREEN_LT, C_GREEN, [
            "Top-5 Dense Retrieval (D5)",
            "Extract 7 Evidence Feats X5",
            "GBDT Gate: P(MORE) >= 0.30",
            "STOP (k=5) vs MORE (k=10)",
            "Frozen Ollama Qwen2.5:7B"
        ]),
        (8.4, 0.5, 2.3, 3.0, "STAGE 4", "Evaluation & Audit", C_GRAY_LT, C_SLATE, [
            "Exact Match & Token F1",
            "Pareto Frontiers (Cost/Latency)",
            "Essential Ablations (A1-A6)",
            "Dual PDF Publishing"
        ])
    ]

    for x, y, w, h, s_id, s_title, bg, border, items in stages:
        # Container
        container = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", facecolor=bg, edgecolor=border, lw=1.5, zorder=1)
        ax.add_patch(container)
        
        # Header Box
        header = patches.FancyBboxPatch((x + 0.1, y + h - 0.75), w - 0.2, 0.65, boxstyle="round,pad=0.01,rounding_size=0.04", facecolor=border, edgecolor="none", zorder=2)
        ax.add_patch(header)
        ax.text(x + w / 2, y + h - 0.35, s_id, ha="center", va="center", fontsize=8, fontweight="bold", color="#ffffff", zorder=3)
        ax.text(x + w / 2, y + h - 0.58, s_title, ha="center", va="center", fontsize=9, fontweight="bold", color="#ffffff", zorder=3)

        # Bullets
        for idx, item in enumerate(items):
            iy = y + h - 1.05 - idx * 0.38
            bullet = patches.Circle((x + 0.25, iy), 0.05, facecolor=border, edgecolor="none", zorder=3)
            ax.add_patch(bullet)
            ax.text(x + 0.38, iy, item, ha="left", va="center", fontsize=7.8, color="#1e293b", zorder=3)

    # Connecting Arrows between stages
    draw_arrow(ax, 2.65, 2.0, 2.90, 2.0, color=C_NAVY, lw=2.0)
    draw_arrow(ax, 5.30, 2.0, 5.55, 2.0, color=C_NAVY, lw=2.0)
    draw_arrow(ax, 8.15, 2.0, 8.35, 2.0, color=C_NAVY, lw=2.0)

    plt.tight_layout()
    out_path = "figures/diag1_end_to_end_framework.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"✔ Generated {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Diagram 2: Runtime Decision Engine Flowchart
# ══════════════════════════════════════════════════════════════════════════════
def make_diag2():
    fig, ax = plt.subplots(figsize=(11, 5.2), dpi=300)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5.2)
    ax.axis("off")

    ax.text(5.5, 4.9, "ADAPTIVE-RAG : RUNTIME CONTROLLER DECISION FLOWCHART", ha="center", va="center", fontsize=13, fontweight="bold", color=C_NAVY)
    ax.text(5.5, 4.55, "Dynamically routing multi-hop queries to prevent context bloat and shield against distractor passages", ha="center", va="center", fontsize=9, color="#64748b")

    # Step 1: Input Query
    draw_box(ax, 0.4, 2.8, 1.8, 1.0, "Input Question q", "Multi-Hop Query", bg_color="#f1f5f9", border_color="#64748b")
    draw_arrow(ax, 2.2, 3.3, 2.6, 3.3, color=C_NAVY)

    # Step 2: Stage 1 Retrieval
    draw_box(ax, 2.6, 2.8, 2.0, 1.0, "FAISS Dense Search", "Retrieve Top-5 Docs (D5)", bg_color=C_BLUE_LT, border_color=C_BLUE)
    draw_arrow(ax, 4.6, 3.3, 5.0, 3.3, color=C_NAVY)

    # Step 3: Feature Extraction
    draw_box(ax, 5.0, 2.8, 2.2, 1.0, "Evidence Extraction", "Compute X5 in R^7", bg_color=C_BLUE_LT, border_color=C_BLUE)
    draw_arrow(ax, 7.2, 3.3, 7.6, 3.3, color=C_NAVY)

    # Step 4: GBDT Decision Gate (Diamond / Octagon style)
    gate = patches.FancyBboxPatch((7.6, 2.6), 2.8, 1.4, boxstyle="round,pad=0.03,rounding_size=0.15", facecolor=C_AMBER_LT, edgecolor=C_AMBER, lw=2.0, zorder=2)
    ax.add_patch(gate)
    ax.text(9.0, 3.5, "GBDT META-GATE", ha="center", va="center", fontsize=10, fontweight="bold", color=C_AMBER, zorder=3)
    ax.text(9.0, 3.2, "P(MORE) >= tau (0.30)?", ha="center", va="center", fontsize=8.5, fontweight="bold", color="#78350f", zorder=3)
    ax.text(9.0, 2.9, "Score decay & redundancy check", ha="center", va="center", fontsize=7.5, color="#92400e", zorder=3)

    # Branch 1: STOP (Early Stopping) -> Downward / Straight Left
    draw_arrow(ax, 9.0, 2.6, 9.0, 1.8, color=C_GREEN, lw=2.0, label="NO [STOP] (95.3%)")
    draw_box(ax, 7.6, 0.7, 2.8, 1.1, "Assemble Context: D5", "Shields 6-10 Distractors\nSaves 48% Tokens", bg_color=C_GREEN_LT, border_color=C_GREEN, title_color=C_GREEN)

    # Branch 2: MORE (Deep Retrieval) -> Right/Down
    # Upwards or Right
    draw_arrow(ax, 10.4, 3.3, 10.7, 3.3, color=C_RED, lw=2.0)
    # Loop over top
    ax.plot([10.7, 10.7], [3.3, 1.25], color=C_RED, lw=2.0, zorder=4)
    ax.annotate("", xy=(6.8, 1.25), xytext=(10.7, 1.25), arrowprops=dict(arrowstyle="-|>", color=C_RED, lw=2.0, mutation_scale=14), zorder=4)
    ax.text(10.75, 2.3, "YES [MORE]\n(4.7%)", ha="left", va="center", fontsize=8, fontweight="bold", color=C_RED)

    # Deep Retrieval Box
    draw_box(ax, 4.2, 0.7, 2.6, 1.1, "Expand Retrieval: D10", "Fetch Ranks 6-10 (D6:10)\nContext = D5 + D6:10", bg_color=C_RED_LT, border_color=C_RED, title_color=C_RED)

    # Converge into Generator
    draw_arrow(ax, 7.6, 1.25, 6.8, 1.25, color=C_NAVY, lw=1.5)
    draw_arrow(ax, 4.2, 1.25, 3.5, 1.25, color=C_NAVY, lw=2.0)

    # Generator Box
    draw_box(ax, 1.2, 0.7, 2.3, 1.1, "Frozen Generator", "Qwen2.5:7B (temp=0.0)\nDeterministic Extractive", bg_color=C_BLUE_LT, border_color=C_BLUE)

    # Final Answer
    draw_arrow(ax, 1.2, 1.25, 0.3, 1.25, color=C_NAVY, lw=2.0)
    ax.text(0.15, 1.25, "Final\nAnswer\na*", ha="right", va="center", fontsize=9, fontweight="bold", color=C_NAVY)

    plt.tight_layout()
    out_path = "figures/diag2_runtime_decision_flowchart.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"✔ Generated {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Diagram 3: 7-Dimensional Evidence State Mathematical Framework
# ══════════════════════════════════════════════════════════════════════════════
def make_diag3():
    fig, ax = plt.subplots(figsize=(11, 4.8), dpi=300)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4.8)
    ax.axis("off")

    ax.text(5.5, 4.55, "MATHEMATICAL FORMULATION : 7 EVIDENCE-STATE FEATURES (X_5 in R^7)", ha="center", va="center", fontsize=13, fontweight="bold", color=C_NAVY)
    ax.text(5.5, 4.20, "Extracting vector distribution geometry and semantic density directly from top-5 retrieved passages", ha="center", va="center", fontsize=9, color="#64748b")

    feats = [
        ("f1: Top-1 Score", "s1 = cos(q, d1)", "Primary anchor of semantic alignment. Dictates minimum confidence.", C_BLUE_LT, C_BLUE, "Importance: 24%"),
        ("f2: Score Mean", "mean(s1...s5)", "Measures average relevance across the entire retrieved candidate pool.", C_BLUE_LT, C_BLUE, "Importance: 14%"),
        ("f3: Score Std Dev", "sigma(s1...s5)", "Variance in similarity scores; quantifies evidence dispersion.", C_BLUE_LT, C_BLUE, "Importance: 9%"),
        ("f4: Score Margin", "s1 - s2", "Decay from rank 1 to 2; high margin indicates a clear, single standout passage.", C_BLUE_LT, C_BLUE, "Importance: 19%"),
        ("f5: Score Range", "s1 - s5", "Decay across all 5 ranks; steep drop indicates weak trailing evidence.", C_BLUE_LT, C_BLUE, "Importance: 21%"),
        ("f6: Max Pairwise", "max cos(di, dj)", "Maximum semantic overlap between any two retrieved passages.", C_GREEN_LT, C_GREEN, "Importance: 8%"),
        ("f7: Redundancy", "mean cos(di, dj)", "Average pairwise overlap; flags redundant vs diverse context passages.", C_GREEN_LT, C_GREEN, "Importance: 5%"),
    ]

    col_w = 4.8
    row_h = 0.82
    positions = [
        (0.6, 3.1), (0.6, 2.15), (0.6, 1.2), (0.6, 0.25),
        (5.8, 3.1), (5.8, 2.15), (5.8, 1.2)
    ]

    for idx, (title, formula, desc, bg, border, imp) in enumerate(feats):
        px, py = positions[idx]
        b = patches.FancyBboxPatch((px, py), col_w, row_h, boxstyle="round,pad=0.01,rounding_size=0.04", facecolor=bg, edgecolor=border, lw=1.2, zorder=1)
        ax.add_patch(b)

        # Title & Formula
        ax.text(px + 0.15, py + 0.55, title, fontsize=9.5, fontweight="bold", color=C_NAVY, zorder=2)
        ax.text(px + 2.0, py + 0.55, f"[{formula}]", fontsize=8.5, fontstyle="italic", color="#0284c7", zorder=2)
        ax.text(px + col_w - 0.15, py + 0.55, imp, ha="right", fontsize=8, fontweight="bold", color=C_AMBER if "2" in imp else "#64748b", zorder=2)
        # Description
        ax.text(px + 0.15, py + 0.22, desc, fontsize=7.8, color="#334155", zorder=2)

    # Explanatory Callout on bottom right
    callout = patches.FancyBboxPatch((5.8, 0.25), col_w, row_h, boxstyle="round,pad=0.01,rounding_size=0.04", facecolor=C_AMBER_LT, edgecolor=C_AMBER, lw=1.2, zorder=1)
    ax.add_patch(callout)
    ax.text(5.8 + col_w / 2, 0.25 + 0.55, "CORE EMPIRICAL FINDING (Ablation A2 & A3)", ha="center", fontsize=9, fontweight="bold", color=C_AMBER, zorder=2)
    ax.text(5.8 + col_w / 2, 0.25 + 0.22, "Top-1 Score + Margin + Range account for 64% of decision power.\nVector score decay governs evidence sufficiency.", ha="center", fontsize=7.5, color="#78350f", zorder=2)

    plt.tight_layout()
    out_path = "figures/diag3_evidence_state_features.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"✔ Generated {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Diagram 4: Distractor Penalty & Context Bloat Mechanism
# ══════════════════════════════════════════════════════════════════════════════
def make_diag4():
    fig, ax = plt.subplots(figsize=(11, 4.5), dpi=300)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4.5)
    ax.axis("off")

    ax.text(5.5, 4.2, "DISTRACTOR PENALTY & CONTEXT BLOAT MECHANISM", ha="center", va="center", fontsize=13, fontweight="bold", color=C_NAVY)
    ax.text(5.5, 3.85, "Visualizing why blind retrieval expansion (k=10) degrades performance on 15.9% of multi-hop queries", ha="center", va="center", fontsize=9, color="#64748b")

    # Panel A: Fixed RAG-5 (Optimal Shielding)
    panel_a = patches.FancyBboxPatch((0.5, 0.5), 4.6, 3.0, boxstyle="round,pad=0.02,rounding_size=0.06", facecolor=C_GREEN_LT, edgecolor=C_GREEN, lw=1.5, zorder=1)
    ax.add_patch(panel_a)
    ax.text(2.8, 3.25, "POLICY A: STOP AT k=5 (PROPOSED B6)", ha="center", fontsize=10, fontweight="bold", color=C_GREEN, zorder=2)
    ax.text(2.8, 2.95, "Shielded Context (High Information Density)", ha="center", fontsize=8, color="#15803d", zorder=2)

    draw_box(ax, 0.8, 2.0, 4.0, 0.65, "Top-5 Retrieved Passages (D5)", "High Relevance (Gold Bridge Facts)", bg_color="#ffffff", border_color=C_GREEN, title_color=C_GREEN, title_size=9)
    draw_box(ax, 0.8, 1.15, 4.0, 0.65, "LLM Attention Focus (Qwen2.5:7B)", "Clean signal -> Exact Answer Generated", bg_color="#ffffff", border_color=C_GREEN, title_color=C_GREEN, title_size=9)
    ax.text(2.8, 0.75, "Optimal: Peak Token F1 (0.5560) | 796 Tokens", ha="center", fontsize=9, fontweight="bold", color=C_GREEN)

    # Panel B: Fixed RAG-10 (Distractor Contamination)
    panel_b = patches.FancyBboxPatch((5.9, 0.5), 4.6, 3.0, boxstyle="round,pad=0.02,rounding_size=0.06", facecolor=C_RED_LT, edgecolor=C_RED, lw=1.5, zorder=1)
    ax.add_patch(panel_b)
    ax.text(8.2, 3.25, "POLICY B: BLIND EXPANSION TO k=10 (B3)", ha="center", fontsize=10, fontweight="bold", color=C_RED, zorder=2)
    ax.text(8.2, 2.95, "Contaminated Context (Context Bloat)", ha="center", fontsize=8, color="#b91c1c", zorder=2)

    draw_box(ax, 6.2, 2.0, 4.0, 0.65, "Top-10 Passages (D5 + D6:10)", "D6:10 contains noisy distractor paragraphs", bg_color="#ffffff", border_color=C_RED, title_color=C_RED, title_size=9)
    draw_box(ax, 6.2, 1.15, 4.0, 0.65, "Diluted Attention Focus", "Distractors trigger hallucination / confusion", bg_color="#ffffff", border_color=C_RED, title_color=C_RED, title_size=9)
    ax.text(8.2, 0.75, "Degradation: F1 Collapses (-0.048 on comparison) | 1,514 Tokens", ha="center", fontsize=8.5, fontweight="bold", color=C_RED)

    plt.tight_layout()
    out_path = "figures/diag4_distractor_penalty_mechanism.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"✔ Generated {out_path}")


if __name__ == "__main__":
    print("Generating Architectural Frameworks and Flowcharts in 'figures/'...")
    make_diag1()
    make_diag2()
    make_diag3()
    make_diag4()
    print("🎉 All 4 architectural diagrams successfully generated!")
