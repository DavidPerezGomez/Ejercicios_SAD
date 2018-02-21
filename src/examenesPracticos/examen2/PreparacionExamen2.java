/**
 * Alicia en el laboratorio ha dicho que va a entrar lo siguiente:
 * 
 * Hacer 2 ejecutables:
 * Ejecutable 1:
 *   [x] Estimar la calidad de un modelo
 *   [-] Guardar modelo y su calidad
 * Ejecutable 2:
 *   [-] Cargar modelo
 *   [ ] Usar el modelo cargado para predecir la clase (sacar esto de: evaluate Javadoc)
 *   
 * NOTA: 
 * 	[x] -> es lo que se supone que ya hemos dado en la etapa anterior.
 *  [ ] -> las funciones nuevas que faltan por implementar en esta etapa.
 *  [-] -> las funciones nuevas que hemos conseguido implementar en esta etapa.
 */
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
import weka.classifiers.lazy.IBk;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.rules.ZeroR;
import weka.classifiers.rules.OneR;
import weka.classifiers.Evaluation;

public class PreparacionExamen2 {
	
	/*
	 * El ejecutable necesita 3 argumentos:
	 * 1º args[0] path donde coger los datos -> ej: /some/where/breast-cancer.arff
	 * 2º args[1] path donde guardar los resultados -> ej: /some/where/results.txt
	 * 3º args[2] path donde cargar o guardar el modelo -> ej: /some/where/oner.model
	 * 
	 * Puedes poner los valores en "run configurations -> pestaña arguments"
	 */
	public static void main(String[] args) throws Exception {
		if (args.length != 3) {
			System.out.println("Error: 3 argumentos esperados");
			System.exit(1);
		}
		
		// obtener direcciones del input y output
		String dataPath = args[0];
		String outputPath = args[1];
		String modelPath = args[2];
		
		// leer los datos
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
		
		// filtar los datos
		AttributeSelection filter = new AttributeSelection();
		CfsSubsetEval eval = new CfsSubsetEval();
		BestFirst search = new BestFirst();
		filter.setEvaluator(eval);
		filter.setSearch(search);
		filter.setInputFormat(data);
		Instances filteredData = Filter.useFilter(data, filter);
		
	// ##################################################################################
	// OPCION 1: CONSTRUIR UN CLASIFICADOR DESDE 0 ######################################
		// instanciar clasificador
		OneR oneR = new OneR();
//		NaiveBayes naiveBayes = new NaiveBayes();
//		IBk ibk = new IBk(5);
//		ZeroR zeroR = new ZeroR();
		
		// evaluar clasificador
//		String results = evalNoHonesta(oneR, filteredData);
		String results = evalHoldOut(oneR, filteredData, 70);
//		String results = evalKFoldCrossValidation(oneR, filteredData, 10);
//		String results = evalLeaveOneOut(ibk, filteredData);

		// en el paso anterior se han entrenado los clasificadores ahora se pueden guardar
		saveModel(oneR, modelPath);
//		saveModel(zeroR, modelPath);
//		saveModel(ibk, modelPath);
//		saveModel(naiveBayes, modelPath);
	// FIN OPCION 1 #####################################################################
	// ##################################################################################
		

	// ##################################################################################
	// OPCION 2: CARGAR UN CLASIFICADOR YA GUARDADO #####################################
//		OneR oneR = (OneR) loadModel(modelPath); // modelo entrenado
//		/*
//		 *  Si hacemos, por ejemplo, una evalNoHonesta con este clasificador que hemos
//		 *  cargado, los resultados serán iguales que el que nos da el que hacemos desde 0.
//		 *  Así que está bien.
//		 */
//		Evaluation evaluator = new Evaluation(filteredData);
//		evaluator.evaluateModel(oneR, filteredData);
//		String results = getResults(evaluator);
	// FIN OPCION 2 ######################################################################
	// ###################################################################################
		
		// se escriben los resultados
		writeToFile(outputPath, results);
		System.out.println(results);
	}
	
	private static String evalNoHonesta(Classifier pClassifier, Instances pData) throws Exception {
		// se instancia el evaluador
		Evaluation evaluator = new Evaluation(pData);
		
		// se entrena el clasificador con el set entero de datos
		pClassifier.buildClassifier(pData);
		
		// se evalua el clasificador con el set entero de datos
		evaluator.evaluateModel(pClassifier, pData);
		
		// se devuelven los resultados
		return getResults(evaluator);
	}
	
	private static String evalHoldOut(Classifier pClassfier, Instances pData, double pTrainPercent) throws Exception {
		// se instancia el evaliador
		Evaluation evaluator = new Evaluation(pData);
		
		// se calcula el nÃºmero de instancias de entrenamiento y de test en base al porcentaje
		int numInstances = pData.numInstances();
		int numTrain = (int) (numInstances * pTrainPercent / 100);
		int numTest = numInstances - numTrain;
		
		// se obtienen los conjuntos de entrenamiento y de test
		pData.randomize(new Random(1));
		Instances trainData = new Instances(pData, 0, numTrain);
		Instances testData = new Instances(pData, numTrain, numTest);
		
		// se entrena el clasificador
		pClassfier.buildClassifier(trainData);
		
		// se evalua el clasificador
		evaluator.evaluateModel(pClassfier, testData);
		
		//se devuelven los resultados
		return getResults(evaluator);
	}
	
	private static String evalKFoldCrossValidation(Classifier pClassifier, Instances pData, int pK) throws Exception {
		// se instancia el evaluador
		Evaluation evaluator = new Evaluation(pData);
		
		// se evalua el clasificador
		evaluator.crossValidateModel(pClassifier, pData, pK, new Random(1));
		
		// se devuelven los resultados
		return getResults(evaluator);
	}
	
	private static String evalLeaveOneOut(Classifier pClassifier, Instances pData) throws Exception {
		// se instancia el evaluador
		Evaluation evaluator = new Evaluation(pData);
		
		// se obtienen el nÃºmero de instancias del set de datos
		int numInstances = pData.numInstances();
		
		// se realiza en k-fold cross validation con k = num. instancias
		return evalKFoldCrossValidation(pClassifier, pData, numInstances);
	}
	
	private static String getResults(Evaluation pEvaluator) throws Exception {
		StringBuilder results = new StringBuilder();
		results.append(pEvaluator.toSummaryString());
		results.append("\n");
		results.append(pEvaluator.toMatrixString());
		
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
	
	private static Classifier loadModel(String pPath) {
		Classifier cls = null;
		try {
			cls = (Classifier) weka.core.SerializationHelper.read(pPath);
		} catch (Exception e) {
			System.out.println("Error al cargar el modelo");
			e.printStackTrace();
		}
		return cls;
	}
}
