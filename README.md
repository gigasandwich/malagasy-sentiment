# Goal
Fetch reviews of products from random sources to do sentimental analysis so `we may predict if a comment of a review is POSITIVE or NEGATIVE` in `MALAGASY` language

# Steps
- [x] Format the reviews in `data/original.txt` into a more raeadable format (CSV) ONLY with positive/negative reviews
- [x] Load the CSV (to train the model)
- [x] Train and test the model
- [x] Traduct the language to Malagasy
- [X] Train and test the model for Malagasy language
- [] Do an e-commerce like website to add comments on a product

# How to run
```bash
py run.py
```

# Modules
To install non native modules of python-3:

```bash
pip install pandas sklearn gensim openpyxl
```

# Note
Best combo so far:
Model saved at: trained_models/logisticregressionmodel-tfidf.pkl
```
Accuracy: 0.7616926503340757
Classification report:                precision    recall  f1-score   support

          -1       0.73      0.91      0.81       250
           1       0.84      0.57      0.68       199

    accuracy                           0.76       449
   macro avg       0.78      0.74      0.75       449
weighted avg       0.78      0.76      0.75       449

Ity no vokatra tsara indrindra novidiko: 1
Tena halako ilay izy, serivisy ratsy be!: -1
Milay izy izany: -1
Hividy hafa koa aho amin ny manaraka: -1
Vokatra ratsy indrindra novidiko hatrizay: -1
Omeko 10/10 izany, tena tsara: 1
Tsy tsara: 1
Tsara: 1
Vokatra ara-barotra tena tsara: 1
Aza mividy ity vokatra ity: -1
Nahoana ny olona no mivarotra an ity, ity no vokatra ratsy indrindra eto: -1
```

# Model size comparison
![Comparison](assets/comparison.png)