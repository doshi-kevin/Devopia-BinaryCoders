from flask import Flask, render_template, request

app = Flask(__name__)

# Dictionary mapping each letter to its corresponding image file
letter_images = {
    "A": {"image": "A.png", "meaning": "Apple"},
    "B": {"image": "B.png", "meaning": "Ball"},
    "C": {"image": "C.png", "meaning": "Cat"},
    "D": {"image": "D.png", "meaning": "Dog"},
    "E": {"image": "E.png", "meaning": "Elephant"},
    "F": {"image": "F.png", "meaning": "Fish"},
    "G": {"image": "G.png", "meaning": "Goat"},
    "H": {"image": "H.png", "meaning": "House"},
    "I": {"image": "I.png", "meaning": "Ice Cream"},
    "J": {"image": "J.png", "meaning": "Jug"},
    "K": {"image": "K.png", "meaning": "Kite"},
    "L": {"image": "L.png", "meaning": "Lion"},
    "M": {"image": "M.png", "meaning": "Monkey"},
    "N": {"image": "N.png", "meaning": "Nest"},
    "O": {"image": "O.png", "meaning": "Orange"},
    "P": {"image": "P.png", "meaning": "Penguin"},
    "Q": {"image": "Q.png", "meaning": "Queen"},
    "R": {"image": "R.png", "meaning": "Rabbit"},
    "S": {"image": "S.png", "meaning": "Snake"},
    "T": {"image": "T.png", "meaning": "Tiger"},
    "U": {"image": "U.png", "meaning": "Umbrella"},
    "V": {"image": "V.png", "meaning": "Violin"},
    "W": {"image": "W.png", "meaning": "Watch"},
    "X": {"image": "X.png", "meaning": "Xylophone"},
    "Y": {"image": "Y.png", "meaning": "Yak"},
    "Z": {"image": "Z.png", "meaning": "Zebra"},
}


def map_text_to_images(text):
    words_with_meanings = []
    for word in text.split():
        word_data = {"word": word}
        for letter in word:
            # Get the image filename and meaning for each letter
            letter_data = letter_images.get(letter.upper())
            if letter_data:
                word_data.setdefault("images", []).append(letter_data["image"])
                word_data.setdefault("meanings", []).append(letter_data["meaning"])
        words_with_meanings.append(word_data)
    return words_with_meanings


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_input = request.form["user_input"]
        words_with_meanings = map_text_to_images(user_input)
        return render_template("sign.html", words_with_meanings=words_with_meanings)
    return render_template("sign.html")


if __name__ == "__main__":
    app.run(debug=True, port=8080)
