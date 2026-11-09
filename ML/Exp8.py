import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.layers import Dense, Dropout, Embedding, Input, SimpleRNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences


def main():
    tf.keras.utils.set_random_seed(42)
    vocabulary_size = 10000
    max_length = 200
    (x_train, y_train), (x_test, y_test) = imdb.load_data(
        num_words=vocabulary_size
    )
    print("Training reviews:", len(x_train))
    print("Testing reviews:", len(x_test))
    print("First review:", x_train[0][:20])
    print("First review sentiment:", y_train[0])

    x_train = pad_sequences(
        x_train,
        maxlen=max_length,
        padding="post",
        truncating="post",
    )
    x_test = pad_sequences(
        x_test,
        maxlen=max_length,
        padding="post",
        truncating="post",
    )

    model = Sequential(
        [
            Input(shape=(max_length,)),
            Embedding(
                input_dim=vocabulary_size,
                output_dim=32,
                mask_zero=True,
            ),
            SimpleRNN(32),
            Dropout(0.3),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()
    model.fit(
        x_train,
        y_train,
        epochs=3,
        batch_size=64,
        validation_split=0.2,
        verbose=1,
    )

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print("Test loss:", test_loss)
    print(f"Test accuracy: {test_accuracy * 100:.2f}%")

    word_index = imdb.get_word_index()

    def preprocess_review(review):
        sequence = [1]
        for word in review.lower().split():
            index = word_index.get(word)
            if index is None or index + 3 >= vocabulary_size:
                sequence.append(2)
            else:
                sequence.append(index + 3)
        return pad_sequences(
            [sequence],
            maxlen=max_length,
            padding="post",
            truncating="post",
        )

    def predict_sentiment(review):
        processed = preprocess_review(review)
        probability = float(model.predict(processed, verbose=0)[0, 0])
        sentiment = "POSITIVE" if probability >= 0.5 else "NEGATIVE"
        print("Review:", review)
        print("Positive probability:", round(probability, 4))
        print("Predicted sentiment:", sentiment)

    predict_sentiment("This movie was excellent and enjoyable")


if __name__ == "__main__":
    main()
