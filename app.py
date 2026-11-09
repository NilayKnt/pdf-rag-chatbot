from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq

klasor = Path(__file__).parent
db_yolu = str(klasor / "chroma_db")

embedding = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")

# Veritabanı varsa yükle, yoksa PDF'ten oluştur
if (klasor / "chroma_db").exists():
    db = Chroma(persist_directory=db_yolu, embedding_function=embedding)
else:
    sayfalar = PyPDFLoader(str(klasor / "ornek.pdf")).load()
    bolucu = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    parcalar = bolucu.split_documents(sayfalar)
    db = Chroma.from_documents(parcalar, embedding, persist_directory=db_yolu)

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

print("PDF ile sohbet hazır. Çıkmak için 'q' yaz.")
while True:
    soru = input("\nSoru: ")
    if soru.strip().lower() == "q":
        break

    # En yakın 3 parçayı bul
    parcalar = db.similarity_search(soru, k=3)
    baglam = "\n\n".join(p.page_content for p in parcalar)

    prompt = f"""Aşağıdaki metne göre soruyu Türkçe ve kısa cevapla.
Cevap metinde yoksa "Metinde bu bilgi yok" de.

Metin:
{baglam}

Soru: {soru}"""

    cevap = llm.invoke(prompt).content
    sayfalar_no = sorted({p.metadata["page"] + 1 for p in parcalar})
    print("\nCevap:", cevap)
    print("Kaynak sayfa:", sayfalar_no)