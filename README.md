# Plain Stock Prediction
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