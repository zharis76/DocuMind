from pypdf import PdfReader

path = "The Laws of Human Nature.pdf"

reader = PdfReader(path)

for i, page in enumerate(reader.pages, start=1):
    text = page.extract_text()
    print(f"--- Page {i} ---")
    print(text[:200])
    print()