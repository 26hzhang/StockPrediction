package com.isaac.stock.utils;

import org.apache.spark.ml.feature.MinMaxScaler;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.functions;
import org.datavec.api.util.ClassPathResource;

import java.io.IOException;

// java.lang.ExceptionInInitializerError  ??????
public class DataPreview {
    public static void main (String[] args) throws IOException {
        SparkSession spark = SparkSession.builder().master("local").appName("DataProcess").getOrCreate();
        String filename = "prices-split-adjusted.csv";
        String symbol = "GOOG";
        // load data from csv file
        Dataset<Row> data = spark.read().format("csv").option("header", true)
                .load(new ClassPathResource(filename).getFile().getAbsolutePath())
                //.filter(functions.col("symbol").equalTo(symbol))
                //.drop("date").drop("symbol")
                .withColumn("openPrice", functions.col("open").cast("double")).drop("open")
                .withColumn("closePrice", functions.col("close").cast("double")).drop("close")
                .withColumn("lowPrice", functions.col("low").cast("double")).drop("low")
                .withColumn("highPrice", functions.col("high").cast("double")).drop("high")
                .withColumn("volumeTmp", functions.col("volume").cast("double")).drop("volume")
                .toDF("date", "symbol", "open", "close", "low", "high", "volume");

        data.show();

        Dataset<Row> symbols = data.select("date", "symbol").groupBy("symbol").agg(functions.count("date").as("count"));
        System.out.println("Number of Symbols: " + symbols.count());
        symbols.show();

        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(new String[] {"open", "low", "high", "volume", "close"})
                .setOutputCol("features");

        data = assembler.transform(data).drop("open", "low", "high", "volume", "close");

        data = new MinMaxScaler().setMin(0).setMax(1)
                .setInputCol("features").setOutputCol("normalizedFeatures")
                .fit(data).transform(data)
                .drop("features").toDF("features");
    }
}
