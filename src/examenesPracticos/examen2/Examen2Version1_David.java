package examenesPracticos.examen2;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;

public class Examen2Version1_David {

    public static void main(String[] args) {
        String trainPath = args[0];
        String outModelPath = args[1];
        String outResultPath = args[2];
        DataSource ds;
        Instances trainInstances = null;
        try {
            ds = new DataSource(trainPath);
            trainInstances = ds.getDataSet();
            trainInstances.setClassIndex(trainInstances.numAttributes()-1);
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }

        RandomForest classifier = optimizeRandomForest(trainInstances);
        saveModel(classifier, outModelPath);
        writeToFile(testClassifier(classifier, trainInstances), outResultPath);
    }

    private static RandomForest optimizeRandomForest(Instances pInstances) {
        RandomForest classifier = new RandomForest();
        int minNumTrees = 2;
        int maxNumTrees = pInstances.numAttributes()/2;
        int indexMinClass = getIndexMinorityClass(pInstances);
        double bestFMeasure = -1;
        int bestNumTrees = -1;

        for(int numTrees = minNumTrees; numTrees <= maxNumTrees; numTrees++) {
            try {
                Evaluation evaluation = new Evaluation(pInstances);
                classifier.setNumIterations(numTrees);
                evaluation.crossValidateModel(classifier, pInstances, 4, new Random(3));
                double fMeasure = evaluation.fMeasure(indexMinClass);
                System.out.println(numTrees + "-> " + fMeasure);
                if (bestFMeasure < fMeasure) {
                    // crédito total a Guzmán (https://github.com/6uzm4n) por al código que sigue
                    bestFMeasure = fMeasure;
                    // fin del crédito a Guzmán
                    bestNumTrees = numTrees;
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

        System.out.println("bestNumTrees: " + bestNumTrees + " -> " + bestFMeasure);
        try {
            classifier.setNumIterations(bestNumTrees);
            classifier.buildClassifier(pInstances);
        } catch (Exception e) {
            e.printStackTrace();
        }

        return classifier;
    }

    private static String testClassifier(Classifier pClassifier, Instances pInstances) {
        StringBuilder result = new StringBuilder();

        // HOLD OUT
        int numIterations = 100;
        double trainPercent = 70.0;
        int numTrain = (int) (pInstances.numInstances()*trainPercent/100);
        int numTest = pInstances.numInstances() - numTrain;
        int indexMinClass = getIndexMinorityClass(pInstances);
        double[] fMeasures = new double[numIterations];
        double avgFMeasure = 0;
        double stDevFMeasure = 0;
        String confMatrix = "";
        for (int i = 0; i < numIterations; i++) {
            try {
                pInstances.randomize(new Random(1));
                Instances train = new Instances(pInstances, 0, numTrain);
                Instances test = new Instances(pInstances, numTrain, numTest);
                pClassifier.buildClassifier(train);
                Evaluation evaluation = new Evaluation(train);
                evaluation.evaluateModel(pClassifier, test);
                fMeasures[i] = evaluation.fMeasure(indexMinClass);
                avgFMeasure += fMeasures[i];
                if (i+1 == numIterations)
                    confMatrix = evaluation.toMatrixString();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
        avgFMeasure = avgFMeasure/numIterations;
        for(int i = 0; i < numIterations; i++) {
            stDevFMeasure += Math.pow((fMeasures[i] - avgFMeasure), 2);
        }
        stDevFMeasure = Math.sqrt(stDevFMeasure/numIterations);

        result.append("HOLD-OUT " + trainPercent + "% (" + numIterations + " iteraciones)\n");
        result.append("Media de f-measures de la clase minoritaria: " + avgFMeasure + "\n");
        result.append("Desviación estándar de f-measures de la clase minoritaria: " + stDevFMeasure + "\n");
        result.append("Matriz de confusión de la última iteración:\n" + confMatrix);

        // EVAL. NO HONESTA
        try {
            pClassifier.buildClassifier(pInstances);
            Evaluation evaluation = new Evaluation(pInstances);
            evaluation.evaluateModel(pClassifier, pInstances);

            result.append("\n\nEVALUACIÓN NO HONESTA\n");
            result.append(evaluation.toClassDetailsString() + "\n");
            result.append(evaluation.toMatrixString());
        } catch (Exception e) {
            e.printStackTrace();
        }

        return result.toString();
    }

    private static int getIndexMinorityClass(Instances pInstances) {
        int[] nomCounts = pInstances.attributeStats(pInstances.classIndex()).nominalCounts;
        int indexMin = -1;
        for (int i = 0; i < nomCounts.length; i++) {
            if (indexMin < 0 || nomCounts[i] < nomCounts[indexMin])
                indexMin = i;
        }
        return indexMin;
    }

    private static void saveModel(Classifier pClassifier, String pPath) {
        try {
            SerializationHelper.write(pPath, pClassifier);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static void writeToFile(String pText, String pPath) {
        try {
            BufferedWriter bw = new BufferedWriter(new FileWriter(pPath));
            bw.write(pText);
            bw.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

}
