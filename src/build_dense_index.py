import os
import csv
import argparse
from tqdm import tqdm
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--passage_file", type=str, required=True)  # DPR psgs_w100.tsv
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="intfloat/e5-base-v2")
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    model = SentenceTransformer(args.model_name)
    dim = model.get_sentence_embedding_dimension()
    index = faiss.IndexFlatIP(dim)

    texts_path = os.path.join(args.output_dir, "texts.txt")
    with open(texts_path, "w") as fout, open(args.passage_file, "r") as fin:
        reader = csv.reader(fin, delimiter="\t")
        _ = next(reader)  # header
        buf = []
        total = 0
        for row in tqdm(reader):
            # row: [id, text, title]
            text, title = row[1], row[2]
            doc = f"{title}. {text}".replace("\n", " ").strip()
            buf.append(doc)
            if len(buf) >= args.batch_size:
                emb = model.encode(buf, normalize_embeddings=True, batch_size=args.batch_size, show_progress_bar=False)
                index.add(np.asarray(emb, dtype="float32"))
                for t in buf:
                    fout.write(t + "\n")
                total += len(buf)
                buf = []
        if buf:
            emb = model.encode(buf, normalize_embeddings=True, batch_size=args.batch_size, show_progress_bar=False)
            index.add(np.asarray(emb, dtype="float32"))
            for t in buf:
                fout.write(t + "\n")
            total += len(buf)

    faiss.write_index(index, os.path.join(args.output_dir, "index.faiss"))
    print(f"Indexed {total} docs into {args.output_dir}")


if __name__ == "__main__":
    main()




