package examenesPracticos.examen2;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;

public class Examen2Version1_David {

    public static void main(String[] args) {
        String trainPath = args[0];
        String outModelPath = args[1];
        String outResultPath = args[2];

        Instances trainInstances = loadInstances(trainPath);
        RandomForest classifier = optimizeRandomForest(trainInstances);
        saveModel(classifier, outModelPath);
        String results = testClassifier(classifier, trainInstances);
        writeToFile(results, outResultPath);
    }

    private static RandomForest optimizeRandomForest(Instances pInstances) {
        RandomForest classifier = new RandomForest();

        int minNumTrees = 2;
        int maxNumTrees = pInstances.numAttributes()/2;
        int bestNumTrees = -1;
        double bestFMeasure = -1;
        int minClassIndex = getMinorityClassIndex(pInstances);

        for(int numTrees = minNumTrees; numTrees <= maxNumTrees; numTrees++) {
            try {
                Evaluation evaluation = new Evaluation(pInstances);
                classifier.setNumIterations(numTrees);
                evaluation.crossValidateModel(classifier, pInstances, 4, new Random(3));
                double fMeasure = evaluation.fMeasure(minClassIndex);
                if (fMeasure > bestFMeasure) {
                    bestFMeasure = fMeasure;
                    bestNumTrees = numTrees;
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

        try {
            classifier.setNumIterations(bestNumTrees);
        } catch (Exception e) {
            e.printStackTrace();
        }

        return classifier;
    }

    private static String testClassifier(Classifier pClassifier, Instances pInstances) {
        return testByHoldOut(pClassifier, pInstances, 70) +
               "\n" +
               testWithTrainSet(pClassifier, pInstances);
    }

    private static String testByHoldOut(Classifier pClassifier, Instances pInstances, double pPercent) {
        StringBuilder result = new StringBuilder();
        try {
            int numTrain = (int) (pInstances.numInstances()*pPercent/100);
            int numTest = pInstances.numInstances() - numTrain;
            int iterations = 100;
            int minClassIndex = getMinorityClassIndex(pInstances);
            double avgFMeasure = 0;
            double stDevFMeasure = 0;
            double [] fMeasures = new double[iterations];
            String confusionMatrix = "";

            for(int i = 0; i < iterations; i++) {
                pInstances.randomize(new Random(1));
                Instances train = new Instances(pInstances, 0, numTrain);
                Instances test = new Instances(pInstances, numTrain, numTest);

                Evaluation evaluation = new Evaluation(pInstances);
                pClassifier.buildClassifier(train);
                evaluation.evaluateModel(pClassifier, test);

                double fMeasure = evaluation.fMeasure(minClassIndex);
                fMeasures[i] = fMeasure;
                avgFMeasure += fMeasure;

                if (i + 1 == iterations) {
                    confusionMatrix = evaluation.toMatrixString();
                }
            }

            avgFMeasure = avgFMeasure/iterations;

            for(int j = 0; j < iterations; j++) {
                stDevFMeasure += Math.pow((fMeasures[j] - avgFMeasure), 2);
            }
            stDevFMeasure = Math.sqrt(stDevFMeasure/iterations);

            result.append(String.format("EVALUACIÓN HOLD-OUT %f%% (%d iteraciones)\n", pPercent, iterations));
            result.append(String.format("Media del f-measure de la clase minoritaria: %f\n", avgFMeasure));
            result.append(String.format("Desviación estándar del f-measure de la clase minoritaria: %f\n", stDevFMeasure));
            result.append(String.format("Matriz de confusión de la última iteración:\n%s\n", confusionMatrix));
        } catch (Exception e) {
            e.printStackTrace();
        }

        return result.toString();
    }

    private static String testWithTrainSet(Classifier pClassifier, Instances pInstances) {
        StringBuilder result = new StringBuilder();
        try {
            pInstances.randomize(new Random(1));
            Evaluation evaluation = new Evaluation(pInstances);
            pClassifier.buildClassifier(pInstances);
            evaluation.evaluateModel(pClassifier, pInstances);

            result.append("EVALUACIÓN NO HONESTA\n");
            result.append(evaluation.toClassDetailsString());
            result.append("\n");
            result.append(evaluation.toMatrixString());
        } catch (Exception e) {
            e.printStackTrace();
        }

        return result.toString();
    }

    private static Instances loadInstances(String pPath) {
        Instances instances = null;
        try {
            ConverterUtils.DataSource ds = new ConverterUtils.DataSource(pPath);
            instances = ds.getDataSet();
            instances.setClassIndex(instances.numAttributes()-1);
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }
        return instances;
    }

    private static int getMinorityClassIndex(Instances pInstances) {
        int[] nomCounts = pInstances.attributeStats(pInstances.classIndex()).nominalCounts;
        int minClassAmount = -1;
        int minClassIndex = -1;
        for(int i = 0; i < nomCounts.length; i++) {
            if (minClassAmount < 0 || nomCounts[i] < minClassAmount) {
                minClassAmount = nomCounts[i];
                minClassIndex = i;
            }
        }
        return minClassIndex;
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
            BufferedWriter bf = new BufferedWriter(new FileWriter(pPath));
            bf.write(pText);
            bf.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

}
