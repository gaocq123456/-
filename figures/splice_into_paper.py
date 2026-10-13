"""Swap the regenerated figures into a compiled paper PDF, without LaTeX.

pdflatex embeds each included figure PDF as a Form XObject.  This replaces
the XObject's content and resources with those of the new figure, so the
figure keeps its exact placement and scale and stays fully vector.

    python figures/splice_into_paper.py paper.pdf paper_mosaic.pdf

FIGURES maps (1-based page, XObject name) to the replacement figure PDF; the
new figure must have the same size as the one it replaces.
"""

import argparse
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from pypdf.generic import DecodedStreamObject, NameObject, ArrayObject, FloatObject

HERE = Path(__file__).resolve().parent

FIGURES = {
    (29, "/Im6"): HERE / "fig6_generators_interventions.pdf",
    (30, "/Im7"): HERE / "fig7_stale_use.pdf",
}


def form_from_page(page, writer):
    """Build a Form XObject in `writer` that draws `page`."""
    form = DecodedStreamObject()
    form.set_data(page.get_contents().get_data())
    x0, y0, x1, y1 = (float(v) for v in page.mediabox)
    form.update({
        NameObject("/Type"): NameObject("/XObject"),
        NameObject("/Subtype"): NameObject("/Form"),
        NameObject("/BBox"): ArrayObject(FloatObject(v) for v in (x0, y0, x1, y1)),
        NameObject("/Resources"): page["/Resources"].clone(writer),
    })
    return writer._add_object(form.flate_encode())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paper", type=Path)
    ap.add_argument("out", type=Path)
    args = ap.parse_args()

    writer = PdfWriter(clone_from=PdfReader(args.paper))
    for (page_no, name), fig_pdf in FIGURES.items():
        xobjects = writer.pages[page_no - 1]["/Resources"]["/XObject"]
        old = xobjects[name].get_object()
        new_page = PdfReader(fig_pdf).pages[0]
        old_size = [round(float(v), 1) for v in old["/BBox"]]
        new_size = [round(float(v), 1) for v in new_page.mediabox]
        if old_size != new_size:
            raise SystemExit(f"page {page_no} {name}: size {old_size} "
                             f"differs from {fig_pdf.name} {new_size}")
        xobjects[NameObject(name)] = form_from_page(new_page, writer)
        print(f"page {page_no} {name} <- {fig_pdf.name}")

    writer.compress_identical_objects(remove_duplicates=True,
                                      remove_unreferenced=True)
    with open(args.out, "wb") as f:
        writer.write(f)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
