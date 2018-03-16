package examenesPracticos.examen2;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;

public class Examen2Version2_David {

	// este es el codigo que entregé en mi examen
	
	public static void main(String[] args) {
		String pathTrain = args[0];
		String outPathModel= args[1];
		String outPathResults = args[2];
		
		DataSource ds;
		Instances instances = null;
		try {
			ds = new DataSource(pathTrain);
			instances = ds.getDataSet();
			instances.setClassIndex(instances.numAttributes()-1);
		} catch (Exception e) {
			e.printStackTrace();
			System.exit(1);
		}
		
		SMO classifier = optimizeClassifier(instances);
		saveModel(classifier, outPathModel);
		String results = evalHoldOut(classifier, instances, 66);
		results += "\n\n" + evalTrainSet(classifier, instances);
		writeToFile(results, outPathResults);
		
	}
	
	private static SMO optimizeClassifier(Instances pInstances) {
		SMO classifier = new SMO();
		int it = 5;
		int minClassIndex = getMinorityClassIndex(pInstances);
		double bestFMEasure = -1;
		int bestExponent = -1;
		for(int i = 0; i <= it; i++) {
			try {
				PolyKernel pK = new PolyKernel();
				pK.setExponent(i);
				classifier.setKernel(pK);
				Evaluation eval;
				eval = new Evaluation(pInstances);
				eval.crossValidateModel(classifier, pInstances, 3, new Random(4));
				double fMeasure = eval.fMeasure(minClassIndex);
				if (fMeasure > bestFMEasure) {
					bestFMEasure = fMeasure;
					bestExponent = i;
				}
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
		try {
			classifier = new SMO();
			PolyKernel pK = new PolyKernel();
			pK.setExponent(bestExponent);
			classifier.setKernel(pK);
			classifier.buildClassifier(pInstances);
		} catch (Exception e) {
			e.printStackTrace();
		}
		return classifier;
	}
	
	private static String evalHoldOut(Classifier pClassifier, Instances pInstances, double pTrainPercent) {
		StringBuilder result = new StringBuilder();
		int trainSize = (int)(pInstances.numInstances()*pTrainPercent/100);
		int testSize = pInstances.numInstances()-trainSize;
		int minClassIndex = getMinorityClassIndex(pInstances);
		int it = 100;
		double[] fMeasures = new double[it];
		double avgFMeasure = 0;
		double stDevFMeasure = 0;
		String matrix = "";
		for(int i = 0; i < it; i++) {
			try {
				pInstances.randomize(new Random(1));
				Instances train = new Instances(pInstances, 0, trainSize);
				Instances test = new Instances(pInstances, trainSize, testSize);
				Evaluation eval;
				eval = new Evaluation(train);
				pClassifier.buildClassifier(train);
				eval.evaluateModel(pClassifier, test);
				double fM = eval.fMeasure(minClassIndex);
				fMeasures[i] = fM;
				avgFMeasure += fM;
				if (i+1 == it) {
					matrix = eval.toMatrixString();
				}
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
		
		avgFMeasure = avgFMeasure/it;
		for(int i = 0; i < it; i++) {
			stDevFMeasure += Math.pow((fMeasures[i]-avgFMeasure), 2);
		}
		stDevFMeasure = Math.sqrt(stDevFMeasure/it);
		
		result.append("HOLD OUT:\n");
		result.append(String.format("media f-measure: %f\n", avgFMeasure));
		result.append(String.format("desviacion estandar f-measure: %f\n", stDevFMeasure));
		result.append(String.format("matriz confusion ultima iteracion:\n%s", matrix));
		
		return result.toString();
	}
	
	private static String evalTrainSet(Classifier pClassifier, Instances pInstances) {
		StringBuilder result = new StringBuilder();
		try { 
			Evaluation eval = new Evaluation(pInstances);
			pClassifier.buildClassifier(pInstances);
			eval.evaluateModel(pClassifier, pInstances);
			result.append("EVAL. NO HONESTA:\n");
			result.append(eval.toClassDetailsString());
			result.append("\n");
			result.append(eval.toMatrixString());
		} catch (Exception e) {
			e.printStackTrace();
		}
		return result.toString();
	}
	
	private static void saveModel(Classifier pModel, String pPath) {
		try {
			SerializationHelper.write(pPath, pModel);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	private static void writeToFile(String pText, String pPath) {
		try {
			BufferedWriter bf = new BufferedWriter(new FileWriter(pPath));
			bf.write(pText);
			bf.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
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
}
