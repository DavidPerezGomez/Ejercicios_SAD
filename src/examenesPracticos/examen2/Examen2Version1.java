package examenesPracticos.examen2;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.BestFirst;
import weka.classifiers.Classifier;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.Evaluation;

public class Examen2Version1 {
	
	/*
	 * El ejecutable necesita 3 argumentos:
	 * 1º args[0] path donde coger los datos -> ej: /some/where/breast-cancer.arff
	 * 2º args[1] path donde guardar los resultados -> ej: /some/where/results.txt
	 * 3º args[2] path donde cargar o guardar el modelo -> ej: /some/where/oner.model
	 * 
	 */ 

	public static void main(String[] args) throws Exception {
		if (args.length != 3) {
			System.out.println("Error: 3 argumentos esperados");
			System.exit(1);
		}
		
		// Obtención entradas
		String dataPath = args[0];
		String modelPath = args[1];
		String outputPath = args[2];

		
		// Lectura de datos
		Instances data = null;
		try {
			DataSource source = new DataSource(dataPath);
			data = source.getDataSet();
			if (data.classIndex() == -1)
				data.setClassIndex(data.numAttributes()-1);
		} catch (Exception e) {
			System.out.println("Error al leer los datos de " + dataPath);
			e.printStackTrace();
			System.exit(1);
		}
		
		// Filtrado de datos
		AttributeSelection filter = new AttributeSelection();
		CfsSubsetEval eval = new CfsSubsetEval();
		BestFirst search = new BestFirst();
		filter.setEvaluator(eval);
		filter.setSearch(search);
		filter.setInputFormat(data);
		Instances filteredData = Filter.useFilter(data, filter);
		
		/*
		 * CONSTRUCCIÓN Y GUARDADO DEL MODELO
		 */
		// Instaciado del clasificador
		RandomForest rf = new RandomForest();
		Evaluation evaluator = new Evaluation(filteredData);
		double topFMeasure = 0;
		int bestI = 0;
		for (int i = 2; i<= data.numInstances()/2; i++) {
			rf.setNumIterations(i);
			evaluator.crossValidateModel(rf, filteredData, 4, new Random(3));
			if (topFMeasure <= evaluator.weightedFMeasure()) {
				topFMeasure = evaluator.weightedFMeasure();
				bestI = i;
			}
		}
		rf.setNumIterations(bestI);
		evaluator.crossValidateModel(rf, filteredData, 4, new Random(3));
		// Guardado del modelo
		rf.buildClassifier(filteredData);
		System.out.println(rf.toString());
		saveModel(rf, modelPath);
		
		// Evaluación del clasificador
		String results1 = "EVALUACIÓN HOLD-OUT\n";
		results1 += evalHoldOut(rf, filteredData, 70);
		String results2 = "EVALUACIÓN NO HONESTA\n";
		results2 += evalNoHonesta(rf, filteredData);
		
		String results = results1 + results2;
		
		// Escritura de resultados en archivo	
		writeToFile(outputPath, results);
		System.out.println(results);	
	}
	
	private static String evalHoldOut(Classifier pClassfier, Instances pData, double pTrainPercent) throws Exception {
		// Instanciado del evaluador
		Evaluation evaluator = new Evaluation(pData);
		
		// Cálculo del número de instancias de entrenamiento y de test en base al porcentaje
		int numInstances = pData.numInstances();
		int numTrain = (int) (numInstances * pTrainPercent / 100);
		int numTest = numInstances - numTrain;
		
		// Obtención los conjuntos de entrenamiento y de test
		pData.randomize(new Random(1));
		Instances trainData = new Instances(pData, 0, numTrain);
		Instances testData = new Instances(pData, numTrain, numTest);
		
		// Entrenamiento del clasificador
		pClassfier.buildClassifier(trainData);
		
		// Evaluación del clasificador
		evaluator.evaluateModel(pClassfier, testData);
		
		// Devolución de resultados
		return getResults(evaluator);
	}
	
	private static String evalNoHonesta(Classifier pClassifier, Instances pData) throws Exception {
		// Instanciado del evaluador
		Evaluation evaluator = new Evaluation(pData);
		
		// Entrenamiento del clasificador con el set completo de datos
		pClassifier.buildClassifier(pData);
		
		// Evaluación del clasificador con el set completo de datos
		evaluator.evaluateModel(pClassifier, pData);
		
		//Construcción del resultado
		String result = "";
		result += "Calidad: " + evaluator.pctCorrect();
		result += "Precisión: " + evaluator.precision(0);
		
		// Devolución de resultados
		return result;
	}
	
	private static String getResults(Evaluation pEvaluator) throws Exception {
		StringBuilder results = new StringBuilder();
		results.append(pEvaluator.toSummaryString());
		results.append("\n");
		results.append(pEvaluator.toClassDetailsString());
		results.append("\n");
		results.append(pEvaluator.toMatrixString());
		
		results.append("EVALUACIÓN NO HONESTA:");
		// Calidad del clasificador, apuntado por si pide guardar solo este dato
		// pEvaluator.pctCorrect());
		return results.toString();
	}
	
	private static void writeToFile(String pPath, String pText) {
		BufferedWriter bw;
		try {
			bw = new BufferedWriter(new FileWriter(pPath));
			bw.write(pText);
			bw.close();
		} catch (IOException e) {
			System.out.println("Error al escribir los resultados en " + pPath);
			e.printStackTrace();
		}
	}
	
	private static void saveModel(Classifier pClassifier, String pPath) {
		try {
			weka.core.SerializationHelper.write(pPath, pClassifier);
		} catch (Exception e) {
			System.out.println("Error al guardar el modelo");
			e.printStackTrace();
		}
	}
}