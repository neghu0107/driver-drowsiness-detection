from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
import pandas as pd

def generate_report():

    df = pd.read_csv("data/driver_log.csv")

    avg_score = df["ai_score"].mean()
    max_sleep = df["sleep_time"].max()

    doc = SimpleDocTemplate("Driver_Report.pdf")
    styles = getSampleStyleSheet()

    content = []

    content.append(
        Paragraph("AI Driver Monitoring Report",
        styles['Title'])
    )

    content.append(
        Paragraph(f"Average Risk Score: {avg_score:.2f}",
        styles['Normal'])
    )

    content.append(
        Paragraph(f"Maximum Sleep Duration: {max_sleep:.2f}s",
        styles['Normal'])
    )

    doc.build(content)

generate_report()