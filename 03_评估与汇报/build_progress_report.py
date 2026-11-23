"""Render the third course report from the outer-fold results and the saved milestone predictions."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from urllib.parse import quote

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph


REPO = "https://github.com/ashlyvanderzwaag221-boop/qwa"
COMMIT = "036ab948fbc1a6a39f4d51302703b9d4e6be6124"
SCRIPT = REPO + "/blob/" + COMMIT + "/" + quote("00_公共支持/pipeline.py") + "#L83-L139"
INK = colors.HexColor("#162C38")
BODY = colors.HexColor("#223944")
TEAL = colors.HexColor("#17665D")
SUBTLE = colors.HexColor("#49616A")
LINE = colors.HexColor("#C6D7D5")
FILL = colors.HexColor("#E9F2EF")
PAPER = colors.white
WIDTH, HEIGHT = A4
LEFT = 43
RIGHT = WIDTH - LEFT


def setup_fonts():
    fonts = Path("C:/Windows/Fonts")
    for alias, name in [("Body", "segoeui.ttf"), ("Bold", "segoeuib.ttf"),
                        ("Serif", "georgiab.ttf"), ("Names", "msyh.ttc")]:
        pdfmetrics.registerFont(TTFont(alias, str(fonts / name), subfontIndex=0))
    pdfmetrics.registerFontFamily("Body", normal="Body", bold="Bold")


def summary(results_file, predictions_file):
    payload = json.loads(results_file.read_text(encoding="utf-8"))
    assert payload["mode"] == "final"
    results = {}
    for model in ["majority", "svm", "random_forest"]:
        rows = sorted((r for r in payload["results"] if r["model"] == model), key=lambda r: r["test_fold"])
        assert [r["test_fold"] for r in rows] == list(range(1, 11)) and sum(r["n"] for r in rows) == 8732
        results[model] = rows
    with predictions_file.open(encoding="utf-8-sig", newline="") as stream:
        air = [r for r in csv.DictReader(stream)
               if r["model"] == "rbf_svm" and r["fold"] == "9" and r["actual"] == "air_conditioner"]
    counts = Counter(r["fsID"] for r in air if r["predicted"] == "drilling")
    assert (len(air), sum(counts.values()), sum(v for _, v in counts.most_common(2))) == (100, 43, 38)
    return results, Counter(r["predicted"] for r in air)


class Page:
    def __init__(self, output):
        self.pdf = canvas.Canvas(str(output), pagesize=A4, pageCompression=1)
        self.pdf.setAuthor("孙煜皓, 茅人方, 程志阳")
        self.pdf.setTitle("Urban Sound Classification | Third Project Report")
        self.y = HEIGHT - 35

    def line(self, y):
        self.pdf.setStrokeColor(LINE)
        self.pdf.setLineWidth(0.65)
        self.pdf.line(LEFT, y, RIGHT, y)

    def text(self, x, y, content, font="Body", size=10, color=BODY):
        self.pdf.setFont(font, size)
        self.pdf.setFillColor(color)
        self.pdf.drawString(x, y, content)

    def paragraph(self, content, size=10.8, leading=14.1, gap=4.5):
        style = ParagraphStyle("body", fontName="Body", fontSize=size, leading=leading, textColor=BODY)
        para = Paragraph(content, style)
        _, height = para.wrap(RIGHT - LEFT, HEIGHT)
        para.drawOn(self.pdf, LEFT, self.y - height)
        self.y -= height + gap

    def heading(self, title):
        self.text(LEFT, self.y - 14, title, "Bold", 13.4, TEAL)
        self.y -= 21

    def results_panel(self, results):
        x1, x2, x3 = LEFT + 7, LEFT + 117, LEFT + 173
        table_right = LEFT + 224
        self.pdf.setFillColor(TEAL)
        self.pdf.rect(LEFT, self.y - 20, table_right - LEFT, 20, fill=1, stroke=0)
        for x, label in [(x1, "Model"), (x2, "Accuracy"), (x3, "Macro-F1")]:
            self.text(x, self.y - 14, label, "Bold", 9.2, PAPER)
        for i, (key, name) in enumerate([("svm", "RBF-SVM"), ("random_forest", "Random forest"), ("majority", "Majority baseline")]):
            y = self.y - 20 - (i + 1) * 19
            if i % 2 == 0:
                self.pdf.setFillColor(FILL)
                self.pdf.rect(LEFT, y, table_right - LEFT, 19, fill=1, stroke=0)
            rows = results[key]
            self.text(x1, y + 5.7, name, "Bold" if i == 0 else "Body", 9.7)
            self.text(x2, y + 5.7, f"{mean(r['accuracy'] for r in rows):.2%}", "Body", 9.7)
            self.text(x3, y + 5.7, f"{mean(r['macro_f1'] for r in rows):.4f}", "Body", 9.7)

        plot_left, plot_right = table_right + 41, RIGHT - 7
        self.text(table_right + 16, self.y - 9, "Macro-F1 by test fold", "Bold", 9, INK)
        gold = colors.HexColor("#8C641B")

        def marker(x, y, svm):
            self.pdf.setLineWidth(1.0)
            self.pdf.setStrokeColor(TEAL if svm else gold)
            self.pdf.setFillColor(TEAL if svm else PAPER)
            if svm:
                self.pdf.circle(x, y, 2.15, stroke=1, fill=1)
            else:
                self.pdf.rect(x - 2.15, y - 2.15, 4.3, 4.3, stroke=1, fill=1)
        for dx, name, svm in [(104, "SVM", True), (60, "Forest", False)]:
            marker(RIGHT - dx, self.y - 6.5, svm)
            self.text(RIGHT - dx + 5, self.y - 9, name, "Body", 8.1)
        def y_for(score):
            return self.y - 64 + (score - .54) / .21 * 43
        for tick in [.55, .65, .75]:
            y = y_for(tick)
            self.text(table_right + 16, y - 2.4, f"{tick:.2f}", "Body", 7.6, SUBTLE)
            self.pdf.setLineWidth(.4)
            self.pdf.setStrokeColor(LINE)
            self.pdf.line(plot_left, y, plot_right + 3, y)
        for i, (s, f) in enumerate(zip(results["svm"], results["random_forest"])):
            x = plot_left + 4 + i * (plot_right - plot_left - 4) / 9
            a, b = y_for(s["macro_f1"]), y_for(f["macro_f1"])
            self.pdf.setStrokeColor(SUBTLE)
            self.pdf.setLineWidth(.7)
            self.pdf.line(x, a, x, b)
            marker(x, a, True)
            marker(x, b, False)
            self.text(x - (2 if i < 9 else 4), self.y - 75, str(i + 1), "Body", 8.0, SUBTLE)
        self.y -= 84


def render(output, results, milestone_air_labels, diagnostics, fold_summary):
    setup_fonts()
    page = Page(output)
    page.text(LEFT, page.y, "MACHINE LEARNING   /   THIRD PROJECT REPORT", "Bold", 9, TEAL)
    names = "孙煜皓     茅人方     程志阳"
    page.text(RIGHT - pdfmetrics.stringWidth(names, "Names", 10), page.y, names, "Names", 10, SUBTLE)
    page.y -= 31
    page.text(LEFT, page.y, "Urban Sound Classification", "Serif", 24.5, INK)
    page.y -= 10
    page.line(page.y)
    page.y -= 7
    page.heading("Progress since the second report")
    comparison = diagnostics["class_weight_comparison"]
    n_validation = comparison["validation_clips"]
    n_clips = fold_summary["n_clips"]
    n_folds = fold_summary["n_folds"]
    n_models = fold_summary["n_models"]
    page.paragraph(
        f"At the milestone, we evaluated our ten-class UrbanSound8K classifiers on {n_validation} clips in one validation fold. "
        f"We have now tested all <b>{n_clips:,} clips</b> once per model across ten held-out folds and checked the clean-audio scores "
        f"against all {n_models * n_folds} confusion matrices.", size=11, leading=14.6, gap=7)
    page.results_panel(results)
    paired = list(zip(results["svm"], results["random_forest"]))
    forest_wins = [s["test_fold"] for s, f in paired if f["macro_f1"] > s["macro_f1"]]
    svm_wins = sum(s["macro_f1"] > f["macro_f1"] for s, f in paired)
    assert forest_wins == [1, 2, 8] and svm_wins == 7
    page.paragraph(
        f"The table gives each test fold equal weight. Macro-F1 averages F1 over the ten classes. "
        f"The SVM led on macro-F1 in {svm_wins} of {n_folds} folds. The forest led on folds 1, 2 and 8.", gap=6)
    weights = comparison["scores"]
    plain, balanced = weights["unweighted"], weights["balanced"]
    page.paragraph(
        f"<b>Controlled class-weight test.</b> On the original fold-9 split, we fixed the {comparison['train_clips']:,} training clips, "
        f"{n_validation} validation clips, features and SVM settings (C = {comparison['parameters']['C']}, gamma = scale). "
        f"Switching from equal to inverse-frequency weights lowered macro-F1 from {plain['macro_f1']:.4f} to {balanced['macro_f1']:.4f}. "
        f"Correct predictions fell from {plain['correct']} to {balanced['correct']} on this split.")
    audit = diagnostics["source_audit"]
    sources = audit["by_source"][:2]
    n_air = sum(milestone_air_labels.values())
    old_count = milestone_air_labels["drilling"]
    assert n_air == audit["air_conditioner_clips"] and old_count == audit["predicted_drilling"]
    page.paragraph(
        f"<b>Tracing errors to source recordings.</b> The milestone SVM labeled {old_count} of {n_air} air-conditioner clips as drilling. "
        f"Grouping those errors by recording ID showed that <b>{audit['errors_from_two_largest_contributors']} of the {old_count}</b> "
        f"came from two recordings. Recording {sources[0]['fsID']} contributed {sources[0]['as_drilling']} clips and recording "
        f"{sources[1]['fsID']} contributed {sources[1]['as_drilling']}.", gap=6)
    page.heading("Models and current pipeline")
    page.paragraph(
        "<b>From WAV to label.</b> We mix each clip to mono and resample to 22,050 Hz. We extract 20 mel-frequency cepstral coefficients "
        "(MFCCs) in 2,048-sample windows with a 512-sample hop. Their means and standard deviations give 40 features. "
        "For a new clip, the fitted classifier takes these same features and returns one of ten labels.")
    page.paragraph(
        "<b>Three classifiers.</b> The majority rule predicts the most frequent training label. The support vector machine (SVM) uses a radial "
        "basis function (RBF) kernel to learn curved boundaries from standardized features. We tested C values of 0.1, 1 and 10 with gamma set to scale. "
        "The random forest averages class probabilities from 200 trees. We tested a depth limit of 20 and unlimited depth, with at least two clips per leaf.")
    page.paragraph(
        "<b>Training and testing.</b> We held out one official fold, tuned on the next by macro-F1 and trained the candidates on the other eight. "
        "We excluded overlapping recording IDs and fitted the SVM scaler on training clips only. We then refitted on eligible clips from nine "
        "folds and scored the held-out fold once.", gap=6)
    page.heading("Difficulties in this phase")
    ninth = next(r for r in results["svm"] if r["test_fold"] == 9)
    ninth_summary = next(r for r in fold_summary["folds"] if r["test_fold"] == 9)
    new_count = ninth["confusion_matrix"][0][4]
    page.paragraph(
        f"<b>Interpreting fewer drilling errors.</b> We compared the milestone and outer-test SVMs on the same {n_air} fold-9 air-conditioner clips. "
        f"Drilling predictions fell from {old_count} to {new_count}, while correct labels rose only from {milestone_air_labels['air_conditioner']} to {ninth['confusion_matrix'][0][0]}. "
        f"Training size ({comparison['train_clips']:,} to {ninth_summary['n_refit']:,} clips) and C "
        f"({comparison['parameters']['C']} to {ninth['parameters']['C']}) both changed. Their individual effects remain open.")
    top_error = fold_summary["models"]["svm"]["largest_confusion"]
    assert (top_error["actual_id"], top_error["predicted_id"]) == (0, 5)
    page.paragraph(
        f"<b>Locating individual errors.</b> Pooling the ten matrices revealed the SVM's most frequent error. It labeled {top_error['count']} of "
        f"{top_error['actual_support']:,} air-conditioner clips as engine idling. We saved fold settings and counts. "
        "Finding these WAVs requires refitting the models and linking each prediction to its clip ID.", gap=6)
    page.heading("Remaining work")
    page.paragraph(
        "We will replay each fold with its saved settings, log clip IDs and predictions, and verify the clean matrices. Holding the fitted model and "
        "clips fixed, we will measure the change in macro-F1 under white noise at 20, 10 and 0 dB signal-to-noise ratio. We will listen to errors and "
        "correct examples, then compare 40 whole-clip features with 80 from two half-clips using the same folds.", gap=0)
    footer_y = 42
    if page.y < footer_y + 12:
        raise ValueError(f"Body reaches footer: body bottom {page.y:.1f}, footer {footer_y}")
    page.line(footer_y)
    repo_label = REPO
    page.text(LEFT, footer_y - 18, "Code", "Bold", 9.3, INK)
    page.text(LEFT + 30, footer_y - 18, repo_label, "Body", 9.3, TEAL)
    page.pdf.linkURL(REPO, (LEFT + 28, footer_y - 21, LEFT + 32 + pdfmetrics.stringWidth(repo_label, "Body", 9.3), footer_y - 6))
    script_label = "Ten-fold training and evaluation code"
    x = RIGHT - pdfmetrics.stringWidth(script_label, "Body", 9.1)
    page.text(x, footer_y - 18, script_label, "Body", 9.1, TEAL)
    page.pdf.linkURL(SCRIPT, (x, footer_y - 21, RIGHT, footer_y - 6))
    page.pdf.showPage()
    page.pdf.save()
    print(f"Saved {output} with body bottom at {page.y:.1f} pt.")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("results", type=Path)
    cli.add_argument("predictions", type=Path)
    cli.add_argument("diagnostics", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    results, milestone_air_labels = summary(args.results, args.predictions)
    diagnostics = json.loads(args.diagnostics.read_text(encoding="utf-8"))
    fold_summary = json.loads(args.results.with_name("tenfold_summary.json").read_text(encoding="utf-8"))
    render(args.output, results, milestone_air_labels, diagnostics, fold_summary)
