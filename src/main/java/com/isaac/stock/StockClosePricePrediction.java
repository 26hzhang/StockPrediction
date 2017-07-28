package com.isaac.stock;

import javafx.util.Pair;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.io.ClassPathResource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.List;

/**
 * Created by zhanghao on 26/7/17.
 * @author ZHANG HAO
 */
public class StockClosePricePrediction {
	
	private static final Logger log = LoggerFactory.getLogger(StockClosePricePrediction.class);
	
    public static void main (String[] args) throws IOException {
    	
        String filename = new ClassPathResource("prices-split-adjusted.csv").getFile().getAbsolutePath();
        String symbol = "GOOG"; // stock name
        int batchSize = 64; // mini-batch size
        int exampleLength = 22; // time series length, assume 22 working days per month
        double splitRatio = 0.9; // 0.9 for training, 0.1 for testing
        int epochs = 100; // epochs for training
        
        // create dataset iterator
        log.info("create stock dataSet iterator...");
        StockDataSetIterator iterator = new StockDataSetIterator(filename, symbol, batchSize, exampleLength, splitRatio);
        log.info("load test dataset...");
        List<Pair<INDArray, Double>> test = iterator.getTestDataSet();
        
        // build lstm network
        log.info("build lstm networks...");
        MultiLayerNetwork net = LSTMNetwork.buildLstmNetworks(iterator.inputColumns(), iterator.totalOutcomes());
        
        // training
        log.info("training...");
        for (int i = 0; i < epochs; i++) {
            DataSet dataSet;
            while (iterator.hasNext()) {
                dataSet = iterator.next();
                net.fit(dataSet);
            }
            iterator.reset(); // reset iterator
            net.rnnClearPreviousState(); // clear previous state
        }
        
        // save model
        log.info("saving model...");
        File locationToSave = new File("src/main/resources/StockPriceLSTM.zip");
        //saveUpdater: i.e., the state for Momentum, RMSProp, Adagrad etc. Save this if you want to train your network more in the future
        ModelSerializer.writeModel(net, locationToSave, true);
        
        // load model
        //log.info("load model...");
        //MultiLayerNetwork restoredNet = ModelSerializer.restoreMultiLayerNetwork(locationToSave);
        
        // Testing
        log.info("testing...");
        double max = iterator.getMaxNum()[1];
        double min = iterator.getMinNum()[1];
        double[] predicts = new double[test.size()];
        double[] actuals = new double[test.size()];
        for (int i = 0; i < test.size(); i++) {
        	predicts[i] = net.rnnTimeStep(test.get(i).getKey()).getDouble(exampleLength - 1) * (max - min) + min;
        	actuals[i] = test.get(i).getValue();
        }

        // print out
        System.out.println("Predict" + "," + "Actual");
        for (int i = 0; i < predicts.length; i++) System.out.println(predicts[i] + "," + actuals[i]);

        // plot predicts and actual values
        log.info("plot...");
        PlotUtil.plot(predicts, actuals);
    }
}
