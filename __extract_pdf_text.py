from pathlib import Path
from PyPDF2 import PdfReader
path = Path('course_project.pdf')
reader = PdfReader(path)
for i, page in enumerate(reader.pages[:10]):
    text = page.extract_text() or ''
    print(f'=== PAGE {i+1} ===')
    print(text)
    print()
