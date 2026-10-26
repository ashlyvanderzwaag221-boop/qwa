"""Build evidence tables and static research figures from actual experiment outputs."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '00_公共支持'))
from project_paths import DATA_RESULTS, TRAIN_RESULTS, EVAL_RESULTS, REPORTS_DIR, TRAIN_STAGE
sys.path.insert(0, str(TRAIN_STAGE))
from pipeline import LABELS, save_json
from experiment import write_csv

OUT = EVAL_RESULTS
OUT.mkdir(parents=True, exist_ok=True)
FIG = REPORTS_DIR / 'figures'
FIG.mkdir(parents=True, exist_ok=True)
data = json.loads((DATA_RESULTS / 'data_audit.json').read_text())
result = json.loads((TRAIN_RESULTS / 'metrics.json').read_text())
models = result['models']
names = {'majority': 'Majority predictor', 'rbf_svm': 'RBF-SVM', 'random_forest': 'Random forest'}
palette = {'majority': '#969EA9', 'rbf_svm': '#186B78', 'random_forest': '#CE8741'}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.labelcolor': '#24313C', 'text.color': '#24313C', 'figure.facecolor': 'white'})

class_rows = [{'class': name, 'clips': data['class_counts'][name]} for name in LABELS]
write_csv(DATA_RESULTS / 'class_distribution.csv', class_rows)
fig, ax = plt.subplots(figsize=(9, 5.4), constrained_layout=True)
labels = [name.replace('_', ' ') for name in LABELS]
bars = ax.barh(labels, [r['clips'] for r in class_rows], color='#186B78')
ax.invert_yaxis()
ax.bar_label(bars, padding=5)
ax.set(xlim=(0, 1150), xlabel='Number of audio clips', title='UrbanSound8K class distribution (8,732 clips)')
fig.savefig(FIG / 'class_distribution.png', dpi=200)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
for ax, metric, title in zip(axes, ['accuracy', 'macro_f1'], ['Accuracy', 'Macro-F1']):
    bars = ax.bar([names[r['model']] for r in models], [r[metric] for r in models], color=[palette[r['model']] for r in models], width=.62)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.grid(axis='y', color='#E3E7EA', linewidth=.7)
    ax.set_axisbelow(True)
    if metric == 'accuracy':
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.bar_label(bars, labels=[f'{r[metric]:.1%}' for r in models], padding=5)
    else:
        ax.bar_label(bars, labels=[f'{r[metric]:.3f}' for r in models], padding=5)
fig.suptitle(f"Development validation on fold 9 (n={result['protocol']['n_validation']})\nScores use the same source-purged split; fold 10 remains unused", fontsize=12)
fig.savefig(FIG / 'model_comparison.png', dpi=200)
plt.close(fig)

per_class, errors = [], []
for record in models:
    name = record['model']
    cm = np.asarray(record['confusion_matrix'])
    for label in LABELS:
        per_class.append({'model': name, 'class': label, **record['classification_report'][label]})
    if name == 'majority':
        continue
    fig, ax = plt.subplots(figsize=(9.3, 8), constrained_layout=True)
    fraction = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    image = ax.imshow(fraction, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(10), labels=labels, rotation=45, ha='right')
    ax.set_yticks(range(10), labels=labels)
    ax.set(xlabel='Predicted class', ylabel='Actual class', title=f'{names[name]}: fold 9 validation confusion matrix')
    for i in range(10):
        for j in range(10):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=9,
                    color='white' if fraction[i, j] > .55 else '#24313C')
            if i != j and cm[i, j]:
                errors.append({'model': name, 'actual': LABELS[i], 'predicted': LABELS[j],
                               'count': int(cm[i, j]), 'class_support': int(cm[i].sum()),
                               'within_class_rate': float(fraction[i, j])})
    fig.colorbar(image, ax=ax, label='Fraction within actual class', fraction=.04, pad=.02)
    fig.savefig(FIG / f'confusion_{name}.png', dpi=200)
    plt.close(fig)
write_csv(OUT / 'per_class_metrics.csv', per_class)
errors.sort(key=lambda r: r['count'], reverse=True)
write_csv(OUT / 'confusion_pairs.csv', errors)
save_json(OUT / 'error_summary.json', {
    name: sorted([r for r in errors if r['model'] == name], key=lambda r: r['count'], reverse=True)[:5]
    for name in ['rbf_svm', 'random_forest']})
print('Saved four figures, per-class metrics and ranked confusion pairs.')
