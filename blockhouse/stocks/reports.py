import io
import base64
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from .models import StockData
from .ML import generate_visualization  
from reportlab.lib.units import inch

def generate_pdf_report(symbol, days, metrics):
    pdf_filename = f'{symbol}_report.pdf'
    c = canvas.Canvas(pdf_filename, pagesize=letter)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(1 * inch, 10 * inch, f"{symbol} Stock Performance Report")
    y = 9 * inch
    c.setFont("Helvetica", 12)
    for metric, value in metrics.items():
        c.drawString(1 * inch, y, f"{metric}: {value}")
        y -= 0.5 * inch
    c.drawImage('stock_price_comparison.png', 1 * inch, 3 * inch, width=6 * inch, height=4 * inch)
    c.save()
    return pdf_filename