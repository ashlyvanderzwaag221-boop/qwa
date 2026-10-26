"""Build the one-page English milestone report directly from verified result files."""
import json
from pathlib import Path
import sys
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from pypdf import PdfReader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '00_公共支持'))
from project_paths import REPORTS_DIR, TRAIN_RESULTS, FEATURES_DIR, EVAL_RESULTS
OUT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else REPORTS_DIR
OUT.mkdir(parents=True, exist_ok=True)
TEAM_NAMES = ('孙煜皓', '茅人方', '程志阳')
name_font_path = Path('C:/Windows/Fonts/msyh.ttc')
if name_font_path.exists():
    pdfmetrics.registerFont(TTFont('TeamNamesCJK', str(name_font_path), subfontIndex=0))
    NAME_FONT = 'TeamNamesCJK'
else:
    pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'))
    NAME_FONT = 'STSong-Light'
read = lambda path: json.loads(Path(path).read_text(encoding='utf-8'))
result = read(TRAIN_RESULTS / 'metrics.json')
audit = read(FEATURES_DIR / 'audio_audit.json')
verified = read(EVAL_RESULTS / 'verification.json')
assert verified['passed'] and audit['succeeded'] == 8732 and not audit['failed']
models = {r['model']: r for r in result['models']}
protocol = result['protocol']
air_conditioner_error = next(
    row for row in read(EVAL_RESULTS / 'error_summary.json')['rbf_svm']
    if row['actual'] == 'air_conditioner' and row['predicted'] == 'drilling'
)
display = {'majority': 'Majority predictor', 'rbf_svm': 'RBF-SVM', 'random_forest': 'Random forest'}
svm, forest = models['rbf_svm'], models['random_forest']
rates = sorted(int(k) for k in audit['sample_rate_counts'])
channels = ', '.join(str(k) for k in sorted(int(k) for k in audit['channel_counts']))
paragraphs = [
    ('Data', f'We use UrbanSound8K: 8,732 labeled WAV clips, ten urban sound classes and ten predefined folds. '
     'Clips are nominally up to four seconds long (decoded durations: 0.05-4.04 s). Seven classes contain 1,000 clips each; siren has 929, car horn 429 and gun shot 374. '
     f'All audio files pass the official per-file MD5 checks, and metadata agree with the Soundata index. '
     f'Observed sampling rates range from {rates[0]:,} to {rates[-1]:,} Hz, with mono or stereo channels. '
     'We convert audio to mono at 22,050 Hz and summarize 20 MFCCs by their temporal means and standard deviations (40 features).'),
    ('Models and evaluation', 'Following the proposal feedback, we explicitly use an RBF kernel for the SVM and compare it with a random forest and a majority-class reference. '
     f'Training uses folds 1-8 ({protocol["n_train"]:,} clips after excluding {protocol["n_excluded_source_overlap"]} clips sharing an original recording source with validation data). '
     f'Fold 9 provides {protocol["n_validation"]} validation clips; fold 10 ({protocol["n_held_out_unused"]} clips) remains unused for fitting and scoring. '
     'We fit SVM standardization on training features only. We select settings by validation macro-F1 from nine SVM and four forest configurations. '
     f'Selected SVM: C={svm["parameters"]["C"]}, gamma={svm["parameters"]["gamma"]}. '
     f'Selected forest: 200 trees, max_depth={forest["parameters"]["max_depth"]}, min_samples_leaf={forest["parameters"]["min_samples_leaf"]}.'),
    ('Progress since the proposal', 'We completed full data acquisition, checksum and metadata checks, extraction of the 8,732 x 40 feature matrix, and the first model comparison. '
     'The submission package includes executable scripts, pinned dependencies, split manifests, tuning logs, per-clip predictions, confusion matrices and saved models. '
     'Nine unit tests pass, and reloading all three models reproduces the saved predictions and metrics. '
     'These scores come from the validation set used for model selection. They are preliminary and are not independent test scores or directly comparable to published ten-fold benchmarks.'),
    ('Problems and limitations', '1. Source leakage. Clips from the same original recording can share background acoustics, inflating validation scores without demonstrating generalization to new recording environments. '
     f'We checked original recording-source IDs, not only duplicate filenames, and excluded {protocol["n_excluded_source_overlap"]} clips from training to reduce this risk. We retained the original audio files.\n\n'
     '2. Class imbalance. Macro-F1 gives each class equal weight in evaluation but does not change training sample counts. '
     'We use it with per-class analysis to assess performance more fully. We have not yet tested whether class weighting or oversampling improves results.\n\n'
     f'3. Error interpretation. On validation data, RBF-SVM predicted drilling for {air_conditioner_error["count"]} of {air_conditioner_error["class_support"]} air-conditioner clips. '
     'Similar spectra, background interference, and loss of temporal order through MFCC mean/standard-deviation aggregation are possible explanations, not established causes. '
     'We need to listen to misclassified clips and compare alternative feature representations.'),
    ('Next steps', 'Freeze the experimental protocol and perform outer ten-fold evaluation with tuning confined to the training portion of each round. '
     'Analyze per-class errors and evaluate added white noise at 20, 10 and 0 dB SNR. '
     f'Responsibilities: {TEAM_NAMES[0]} handles data and features; {TEAM_NAMES[1]} handles models and tuning; {TEAM_NAMES[2]} handles evaluation, reproducibility and noise experiments.'),
]

styles = getSampleStyleSheet()
body = ParagraphStyle('BodyCustom', fontName='Helvetica', fontSize=9.6, leading=12.2, textColor=colors.HexColor('#28313A'), spaceAfter=5)
heading = ParagraphStyle('HeadingCustom', parent=body, fontName='Helvetica-Bold', fontSize=10.5, leading=13, textColor=colors.HexColor('#186B78'), spaceBefore=5, spaceAfter=3)
title = ParagraphStyle('TitleCustom', parent=body, fontName='Helvetica-Bold', fontSize=19.5, leading=23, textColor=colors.HexColor('#172430'), spaceAfter=3)
small = ParagraphStyle('SmallCustom', parent=body, fontSize=8, leading=10)
story = [Paragraph('Urban Sound Classification', title),
         Paragraph('Second Project Report / Milestone', small),
         Spacer(1, 5)]
plain = ['URBAN SOUND CLASSIFICATION', 'Second Project Report / Milestone', '']
for name, content in paragraphs:
    story.append(Paragraph(name, heading))
    for block in content.split('\n\n'):
        marked_content = block
        for member_name in TEAM_NAMES:
            marked_content = marked_content.replace(member_name, f'<font name="{NAME_FONT}">{member_name}</font>')
        if name == 'Problems and limitations':
            for lead in ['1. Source leakage.', '2. Class imbalance.', '3. Error interpretation.']:
                marked_content = marked_content.replace(lead, f'<b>{lead}</b>')
        story.append(Paragraph(marked_content, body))
    plain += [name.upper(), content, '']
    if name == 'Progress since the proposal':
        rows = [['Model', 'Validation accuracy', 'Validation macro-F1']]
        for key in ['majority', 'rbf_svm', 'random_forest']:
            record = models[key]
            rows.append([display[key], f'{record["accuracy"]:.2%}', f'{record["macro_f1"]:.4f}'])
        table = Table(rows, colWidths=[175, 158, 158], hAlign='LEFT')
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#186B78')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#F1F5F5'), colors.white]),
            ('LINEBELOW', (0, -1), (-1, -1), .5, colors.HexColor('#CED7DA')),
        ]))
        story += [table, Spacer(1, 4)]
        plain += [' | '.join(row) for row in rows] + ['']
source = 'Source: Salamon, Jacoby & Bello, ACM Multimedia (2014). <link href="https://urbansounddataset.weebly.com/urbansound8k.html" color="#186B78">UrbanSound8K dataset</link>.<br/>Code and experiment evidence: <link href="https://github.com/ashlyvanderzwaag221-boop/qwa" color="#186B78">https://github.com/ashlyvanderzwaag221-boop/qwa</link>.'
story += [Spacer(1, 3), Paragraph(source, small)]
path = OUT / 'Milestone_Report.pdf'
document = SimpleDocTemplate(str(path), pagesize=A4, rightMargin=45, leftMargin=45, topMargin=34, bottomMargin=30,
                            title='Urban Sound Classification - Second Project Report', author='Urban Sound Project Team')
document.build(story)
reader = PdfReader(path)
if len(reader.pages) != 1:
    raise ValueError(f'Report must be one page, got {len(reader.pages)}')
(OUT / 'Milestone_Report.txt').write_text('\n'.join(plain) + '\nSource: https://urbansounddataset.weebly.com/urbansound8k.html\nCode and experiment evidence: https://github.com/ashlyvanderzwaag221-boop/qwa\n', encoding='utf-8')
print('One-page report created:', path)
