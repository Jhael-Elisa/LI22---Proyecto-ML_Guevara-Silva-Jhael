# -*- coding: utf-8 -*-
"""Convierte el informe y la presentación de Markdown a PDF."""
import pathlib
import re

import markdown
from xhtml2pdf import pisa

DOCS = pathlib.Path(__file__).resolve().parent


def render(md_text, landscape=False):
    html_body = markdown.markdown(md_text, extensions=["tables"])
    page = "A4 landscape" if landscape else "A4 portrait"
    return f"""
    <html>
    <head>
    <style>
        @page {{ size: {page}; margin: 2cm 1.8cm; }}
        body {{ font-family: Helvetica, Arial, sans-serif; font-size: 10.5pt;
               color: #222; line-height: 1.45; }}
        h1 {{ font-size: 20pt; color: #1a3d6d; text-align: center;
              margin-bottom: 10pt; }}
        h2 {{ font-size: 14pt; color: #1a3d6d; border-bottom: 1px solid #1a3d6d;
              padding-bottom: 3pt; margin-top: 16pt; }}
        h3 {{ font-size: 11.5pt; color: #244b7a; }}
        table {{ border-collapse: collapse; width: 100%; margin: 8pt 0; }}
        th {{ background-color: #1a3d6d; color: white; font-size: 9.5pt;
              padding: 5pt; border: 1px solid #1a3d6d; text-align: left; }}
        td {{ border: 1px solid #b0b8c4; padding: 4pt; font-size: 9.5pt;
              vertical-align: top; }}
        code {{ font-family: Courier, monospace; background-color: #f0f2f5;
                font-size: 9.5pt; }}
        li {{ margin-bottom: 3pt; }}
        hr {{ color: #cccccc; }}
        .slide {{ page-break-after: always; }}
        .slide:last-child {{ page-break-after: auto; }}
        .slide h1 {{ font-size: 26pt; }}
        .slide h2 {{ font-size: 22pt; }}
        .slide p, .slide li, .slide td, .slide th {{ font-size: 15pt; }}
    </style>
    </head>
    <body>{html_body}</body>
    </html>
    """


def convert(md_path, pdf_path, landscape=False, slides=False):
    md_text = md_path.read_text(encoding="utf-8")
    if slides:
        partes = re.split(r"\n---\s*\n", md_text)
        html = "".join(
            f'<div class="slide">{markdown.markdown(p, extensions=["tables"])}</div>'
            for p in partes
        )
        html_doc = render("", landscape=True).replace("<body></body>", f"<body>{html}</body>")
    else:
        html_doc = render(md_text, landscape=landscape)

    with open(pdf_path, "wb") as salida:
        estado = pisa.CreatePDF(html_doc, dest=salida, encoding="utf-8")
    if estado.err:
        raise RuntimeError(f"Errores al generar {pdf_path}")
    print("PDF generado:", pdf_path)


convert(DOCS / "informe_final.md", DOCS / "informe_final.pdf")
convert(DOCS / "presentacion.md", DOCS / "presentacion.pdf", slides=True)
