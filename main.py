from dotenv import load_dotenv
from flask import Flask, request, jsonify
import os
import gensim
import MeCab

load_dotenv()

app = Flask(__name__)
mecab = MeCab.Tagger()

model_path = "./model_gensim_norm"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")
model = gensim.models.KeyedVectors.load(model_path, mmap='r')

allowed_domain = "example.com"

@app.route("/parse", methods=["GET"])
def parse():
    referrer = request.referrer

    if not referrer or allowed_domain not in referrer:
        return jsonify({"error": "Access denied"}), 403

    text = request.args.get("text", "")
    parsed = mecab.parse(text)
    result = [{"surface": line.split("\t")[0], "feature": line.split("\t")[1]} for line in parsed.split("\n") if line and "\t" in line]
    return jsonify(result)

@app.route("/distance", methods=["GET"])
def distance():
    referrer = request.referrer

    if not referrer or allowed_domain not in referrer:
        return jsonify({"error": "Access denied"}), 403

    if request.headers.get("Content-Type") != "application/json":
        return jsonify({
            "pairs": [],
            "error": "Content-Type is wrong"
        })

    pairs = request.get_json().get("pairs")
    if pairs is None:
        return jsonify({
            "pairs": [],
            "error": "Missing pairs"
        })

    result = []
    for pair in pairs:
        word1 = pair[0] if pair[0] is not None else ""
        word2 = pair[1] if pair[1] is not None else ""
        if word1 == "" or word2 == "":
            continue

        similarity = ""
        errorMessage = ""
        isWord1InVocab = word1 in model
        isWord2InVocab = word2 in model
        if isWord1InVocab and isWord2InVocab:
            similarity = str(model.similarity(word1, word2))

        if not isWord1InVocab and not isWord2InVocab:
            errorMessage = f"'{word1}' and '{word2}' not found in the vocabulary"
        elif not isWord1InVocab:
            errorMessage = f"'{word1}' not found in the vocabulary"
        elif not isWord2InVocab:
            errorMessage = f"'{word2}' not found in the vocabulary"

        result.append({
            "word1": word1,
            "word2": word2,
            "similarity": similarity,
            "error": errorMessage
        })

    return jsonify({
        "pairs": result,
        "error": ""
    })

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
