# Plain Stock Prediction

![Authour](https://img.shields.io/badge/Author-Zhang%20Hao%20(Isaac%20Changhau)-blue.svg) ![](https://img.shields.io/badge/Java-1.8-brightgreen.svg) ![](https://img.shields.io/badge/DeepLearning4J-0.9.1-yellowgreen.svg) ![](https://img.shields.io/badge/ND4J-0.9.1-yellowgreen.svg) ![](https://img.shields.io/badge/Guava-23.0-yellowgreen.svg) ![](https://img.shields.io/badge/OpenCSV-3.9-yellowgreen.svg) ![](https://img.shields.io/badge/Spark-2.1.0-yellowgreen.svg)

Plain Stock Price Prediction via RNNs with Graves LSTM unit.

Training and Predicting a specific feature by setting `PriceCategory` in `com.isaac.stock.predict.StockPricePrediction.java` as:
```java
PriceCategory category = PriceCategory.CLOSE; // CLOSE: train and predict close price
// or
PriceCategory category = PriceCategory.OPEN; // OPEN: train and predict open price
// ...
```
The `PriceCategory` enum:
```java
public enum PriceCategory {
    OPEN, CLOSE, LOW, HIGH, VOLUME, ALL
}
```
Predicting all features as:
```java
PriceCategory category = PriceCategory.ALL; // ALL: train and predict all features
```

**Demo Result**

<img src="predict.png" align=center />

**A Useful GitHub Repository**: [timestocome/Test-stock-prediction-algorithms](https://github.com/timestocome/Test-stock-prediction-algorithms), which contains much information, methods and sources about predict stock and market movements.
