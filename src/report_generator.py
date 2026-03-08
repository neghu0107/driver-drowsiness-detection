from fpdf import FPDF
import datetime

def generate_report(driver,risk):

    pdf = FPDF()

    pdf.add_page()

    pdf.set_font("Arial",size=16)

    pdf.cell(200,10,"Driver Monitoring Report",ln=True)

    pdf.cell(200,10,f"Driver : {driver}",ln=True)
    pdf.cell(200,10,f"Risk Score : {risk}",ln=True)

    pdf.cell(200,10,f"Generated : {datetime.datetime.now()}",ln=True)

    file = f"{driver}_report.pdf"

    pdf.output(file)

    return file