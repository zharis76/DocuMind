from pypdf import PdfReader


def get_pages(path):
    reader = PdfReader(path)
    return [(i, page.extract_text()) for i, page in enumerate(reader.pages, start=1)]


def chunk_page(page_no, text, chunk_words=180, overlap_words=40):
    words = text.split()
    step = chunk_words - overlap_words
    chunks = []
    for start in range(0, len(words), step):
        piece = words[start : start + chunk_words]
        if len(piece) >= 15:
            chunks.append({"page": page_no, "text": " ".join(piece)})
    return chunks


def chunk_all(pages):
    all_chunks = []
    for page_no, text in pages:
        if text:
            all_chunks.extend(chunk_page(page_no, text))
    return all_chunks


if __name__ == "__main__":
    pages = get_pages("The Laws of Human Nature.pdf")
    chunks = chunk_all(pages)
    print(f"Total chunks: {len(chunks)}")
    print(f"First chunk: {chunks[0]}")
    print(f"Second chunk: {chunks[1]}")