package com.isaac.stock.representation;

import com.opencsv.CSVReader;
import javafx.util.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.NoSuchElementException;

/**
 * Created by zhanghao on 26/7/17.
 * Modified by zhanghao on 28/9/17.
 * @author ZHANG HAO
 */
public class StockDataSetIterator implements DataSetIterator {

    private final int VECTOR_SIZE = 5;
    private int miniBatchSize;
    private int exampleLength;

    private double[] minNum = new double[VECTOR_SIZE];
    private double[] maxNum = new double[VECTOR_SIZE];

    private PriceCategory category;

    private LinkedList<Integer> exampleStartOffsets = new LinkedList<>();

    private List<StockData> train;
    private List<Pair<INDArray, INDArray>> test;

    public StockDataSetIterator(String filename, String symbol, int miniBatchSize, int exampleLength, double splitRatio,
                         PriceCategory category) {
        List<StockData> stockDataList = readStockDataFromFile(filename, symbol);
        this.miniBatchSize = miniBatchSize;
        this.exampleLength = exampleLength;
        this.category = category;
        int split = (int) Math.round(stockDataList.size() * splitRatio);
        train = stockDataList.subList(0, split);
        test = generateTestDataSet(stockDataList.subList(split, stockDataList.size()));
        initializeOffsets();
    }

    private void initializeOffsets () {
        exampleStartOffsets.clear();
        int window = exampleLength + 1;
        for (int i = 0; i < train.size() - window; i++) {
            exampleStartOffsets.add(i);
        }
    }

    public List<Pair<INDArray, INDArray>> getTestDataSet() { return test; }

    public double[] getMaxNum() { return maxNum; }

    public double[] getMinNum() { return minNum; }

    @Override
    public DataSet next(int num) {
        if (exampleStartOffsets.size() == 0) throw new NoSuchElementException();
        int actualMiniBatchSize = Math.min(num, exampleStartOffsets.size());
        INDArray input = Nd4j.create(new int[] {actualMiniBatchSize, VECTOR_SIZE, exampleLength}, 'f');
        INDArray label;
        if (category.equals(PriceCategory.ALL))
            label = Nd4j.create(new int[] {actualMiniBatchSize, VECTOR_SIZE, exampleLength}, 'f');
        else
            label = Nd4j.create(new int[] {actualMiniBatchSize, 1, exampleLength}, 'f');
        for (int index = 0; index < actualMiniBatchSize; index++) {
            int startIdx = exampleStartOffsets.removeFirst();
            int endIdx = startIdx + exampleLength;
            StockData curData = train.get(startIdx);
            StockData nextData;
            for (int i = startIdx; i < endIdx; i++) {
                nextData = train.get(i + 1);
                int c = i - startIdx;
                input.putScalar(new int[] {index, 0, c}, (curData.getOpen() - minNum[0]) / (maxNum[0] - minNum[0]));
                input.putScalar(new int[] {index, 1, c}, (curData.getClose() - minNum[1]) / (maxNum[1] - minNum[1]));
                input.putScalar(new int[] {index, 2, c}, (curData.getLow() - minNum[2]) / (maxNum[2] - minNum[2]));
                input.putScalar(new int[] {index, 3, c}, (curData.getHigh() - minNum[3]) / (maxNum[3] - minNum[3]));
                input.putScalar(new int[] {index, 4, c}, (curData.getVolume() - minNum[4]) / (maxNum[4] - minNum[4]));
                if (category.equals(PriceCategory.ALL)) {
                    label.putScalar(new int[] {index, 0, c}, (nextData.getOpen() - minNum[1]) / (maxNum[1] - minNum[1]));
                    label.putScalar(new int[] {index, 1, c}, (nextData.getClose() - minNum[1]) / (maxNum[1] - minNum[1]));
                    label.putScalar(new int[] {index, 2, c}, (nextData.getLow() - minNum[2]) / (maxNum[2] - minNum[2]));
                    label.putScalar(new int[] {index, 3, c}, (nextData.getHigh() - minNum[3]) / (maxNum[3] - minNum[3]));
                    label.putScalar(new int[] {index, 4, c}, (nextData.getVolume() - minNum[4]) / (maxNum[4] - minNum[4]));
                } else {
                    label.putScalar(new int[]{index, 0, c}, feedLabel(nextData));
                }
                curData = nextData;
            }
            if (exampleStartOffsets.size() == 0) break;
        }
        return new DataSet(input, label);
    }

    private double feedLabel(StockData data) {
        double value;
        switch (category) {
            case OPEN: value = (data.getOpen() - minNum[0]) / (maxNum[0] - minNum[0]); break;
            case CLOSE: value = (data.getClose() - minNum[1]) / (maxNum[1] - minNum[1]); break;
            case LOW: value = (data.getLow() - minNum[2]) / (maxNum[2] - minNum[2]); break;
            case HIGH: value = (data.getHigh() - minNum[3]) / (maxNum[3] - minNum[3]); break;
            case VOLUME: value = (data.getVolume() - minNum[4]) / (maxNum[4] - minNum[4]); break;
            default: throw new NoSuchElementException();
        }
        return value;
    }

    @Override
    public int totalExamples() { return train.size() - exampleLength - 1; }

    @Override
    public int inputColumns() { return VECTOR_SIZE; }

    @Override
    public int totalOutcomes() {
        if (this.category.equals(PriceCategory.ALL)) return VECTOR_SIZE;
        else return 1;
    }

    @Override
    public boolean resetSupported() { return false; }

    @Override
    public boolean asyncSupported() { return false; }

    @Override
    public void reset() { initializeOffsets(); }

    @Override
    public int batch() { return miniBatchSize; }

    @Override
    public int cursor() { return totalExamples() - exampleStartOffsets.size(); }

    @Override
    public int numExamples() { return totalExamples(); }

    @Override
    public void setPreProcessor(DataSetPreProcessor dataSetPreProcessor) {
        throw new UnsupportedOperationException("Not Implemented");
    }

    @Override
    public DataSetPreProcessor getPreProcessor() { throw new UnsupportedOperationException("Not Implemented"); }

    @Override
    public List<String> getLabels() { throw new UnsupportedOperationException("Not Implemented"); }

    @Override
    public boolean hasNext() { return exampleStartOffsets.size() > 0; }

    @Override
    public DataSet next() { return next(miniBatchSize); }
    
    private List<Pair<INDArray, INDArray>> generateTestDataSet (List<StockData> stockDataList) {
    	int window = exampleLength + 1;
    	List<Pair<INDArray, INDArray>> test = new ArrayList<>();
    	for (int i = 0; i < stockDataList.size() - window; i++) {
    		INDArray input = Nd4j.create(new int[] {exampleLength, VECTOR_SIZE}, 'f');
    		for (int j = i; j < i + exampleLength; j++) {
    			StockData stock = stockDataList.get(j);
    			input.putScalar(new int[] {j - i, 0}, (stock.getOpen() - minNum[0]) / (maxNum[0] - minNum[0]));
    			input.putScalar(new int[] {j - i, 1}, (stock.getClose() - minNum[1]) / (maxNum[1] - minNum[1]));
    			input.putScalar(new int[] {j - i, 2}, (stock.getLow() - minNum[2]) / (maxNum[2] - minNum[2]));
    			input.putScalar(new int[] {j - i, 3}, (stock.getHigh() - minNum[3]) / (maxNum[3] - minNum[3]));
    			input.putScalar(new int[] {j - i, 4}, (stock.getVolume() - minNum[4]) / (maxNum[4] - minNum[4]));
    		}
            StockData stock = stockDataList.get(i + exampleLength);
            INDArray label;
            if (category.equals(PriceCategory.ALL)) {
                label = Nd4j.create(new int[]{VECTOR_SIZE}, 'f');
                label.putScalar(new int[] {0}, stock.getOpen());
                label.putScalar(new int[] {1}, stock.getClose());
                label.putScalar(new int[] {2}, stock.getLow());
                label.putScalar(new int[] {3}, stock.getHigh());
                label.putScalar(new int[] {4}, stock.getVolume());
            } else {
                label = Nd4j.create(new int[] {1}, 'f');
                switch (category) {
                    case OPEN: label.putScalar(new int[] {0}, stock.getOpen()); break;
                    case CLOSE: label.putScalar(new int[] {0}, stock.getClose()); break;
                    case LOW: label.putScalar(new int[] {0}, stock.getLow()); break;
                    case HIGH: label.putScalar(new int[] {0}, stock.getHigh()); break;
                    case VOLUME: label.putScalar(new int[] {0}, stock.getVolume()); break;
                    default: throw new NoSuchElementException();
                }
            }
    		test.add(new Pair<>(input, label));
    	}
    	return test;
    }

	private List<StockData> readStockDataFromFile (String filename, String symbol) {
        List<StockData> stockDataList = new ArrayList<>();
        try {
            List<String[]> list = new CSVReader(new FileReader(filename)).readAll();
            for (int i = 0; i < maxNum.length; i++) {
                maxNum[i] = Double.MIN_VALUE;
                minNum[i] = Double.MAX_VALUE;
            }
            for (String[] arr : list) {
                if (!arr[1].equals(symbol)) continue;
                double[] nums = new double[VECTOR_SIZE];
                for (int i = 0; i < arr.length - 2; i++) {
                    nums[i] = Double.valueOf(arr[i + 2]);
                    if (nums[i] > maxNum[i]) maxNum[i] = nums[i];
                    if (nums[i] < minNum[i]) minNum[i] = nums[i];
                }
                stockDataList.add(new StockData(arr[0], arr[1], nums[0], nums[1], nums[2], nums[3], nums[4]));
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return stockDataList;
    }
}
