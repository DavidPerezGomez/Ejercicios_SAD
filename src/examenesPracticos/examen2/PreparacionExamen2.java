/**
 * Alicia en el laboratorio ha dicho que va a entrar lo siguiente:
 * 
 * Hacer 2 ejecutables:
 * Ejecutable 1:
 *   [x] Estimar la calidad de un modelo
 *   [-] Guardar modelo y su calidad
 * Ejecutable 2:
 *   [-] Cargar modelo
 *   [-] Usar el modelo cargado para predecir la clase (sacar esto de: evaluate Javadoc)
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

import weka.core.Instance;
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
	/**
	 * @param args
	 * @throws Exception
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
		
		// hace un print de los atributos y de la clase
		printAttributesAndClass(data);
		
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
		

	// ###################################################################################
	// ANEXO 1 : ¿COMO PREDECIR UNA CLASE? ###############################################
		/*
		 * FUENTE: https://stackoverflow.com/questions/28123913/print-out-prediction-with-weka-in-java
		 * 
		 * Imaginemos que hemos entrenado un clasificador con el archivo "breast-cancer.arff" y
		 * que ahora tenemos otro con una sola instancia del que no sabemos la clase (tiene una ? en la clase)
		 * 
		 * ----------------------------------------------------------
		 * Archivo inventado "breast-cancer-one-instance.arff"
		 * ----------------------------------------------------------
		 * @relation breast-cancer
		 * 
		 * @attribute age {'10-19','20-29','30-39','40-49','50-59','60-69','70-79','80-89','90-99'}
		 * @attribute menopause {'lt40','ge40','premeno'}
		 * @attribute tumor-size {'0-4','5-9','10-14','15-19','20-24','25-29','30-34','35-39','40-44','45-49','50-54','55-59'}
		 * @attribute inv-nodes {'0-2','3-5','6-8','9-11','12-14','15-17','18-20','21-23','24-26','27-29','30-32','33-35','36-39'}
		 * @attribute node-caps {'yes','no'}
		 * @attribute deg-malig {'1','2','3'}
		 * @attribute breast {'left','right'}
		 * @attribute breast-quad {'left_up','left_low','right_up','right_low','central'}
		 * @attribute 'irradiat' {'yes','no'}
		 * @attribute 'Class' {'no-recurrence-events','recurrence-events'}
		 * 
		 * @data
		 * '40-49','premeno','15-19','0-2','yes','3','right','left_up','no',?
		 * ----------------------------------------------------------
		 */
//		DataSource source = new DataSource("/home/ander/breast-cancer-one-instance.arff");
//        Instances dataToPredict = source.getDataSet();
//        // setting class attribute if the data format does not provide this information
//        // For example, the XRFF format saves the class attribute information as well
//        if (dataToPredict.classIndex() == -1)
//            dataToPredict.setClassIndex(dataToPredict.numAttributes() - 1);
//
//		predictClass(oneR, dataToPredict);
	// FIN ANEXO 1 #######################################################################
	// ###################################################################################
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
	
	private static void predictClass(Classifier pClassifier, Instances pDatos) {
		try {
			// cogemos la primera linea de nuestro archivo inventado,
			// que tiene un ? en la instancia (mirar ANEXO 1)
			Instance instance = pDatos.instance(0);
			// clasificamos la instancia y guardamos el double que devuelve
			double label = pClassifier.classifyInstance(instance);
			// guardamos el resultado en la instancia
			instance.setClassValue(label);
			// mostramos el resultado por pantalla
			System.out.println(instance.stringValue(instance.classIndex()));
			// si hemos seguido el ejemplo que he explicado el resultado es "recurrence-events"
		} catch (Exception e) {
			System.out.println("Error al clasificar una instancia");
			e.printStackTrace();
		}

	}
	

	private static void printAttributeCount(Instances pDatos, int index) {
		weka.core.AttributeStats at = pDatos.attributeStats(index);
		
		int[] array = at.nominalCounts;
		
		System.out.println(" Nombre " + pDatos.attribute(pDatos.classIndex()).name());
		for (int i = 0; i < array.length; i++) {
			System.out.print(pDatos.attribute(pDatos.classIndex()).value(i) + ": ");
			System.out.println(array[i]);
		}
	}
	

	private static void printAttributesAndClass(Instances pDatos) {
		// lo preguntó en el examen del grupo 1
		int numAtttributes = pDatos.numAttributes();
		
		printAttributeCount(pDatos, pDatos.classIndex());
		
		System.out.println("Hay " + numAtttributes + " atributos:");

		for (int i=0; i < numAtttributes; i++) {
			System.out.println(" - " + pDatos.attribute(i).name());
		}
		
		System.out.println("La clase es: " + pDatos.attribute(pDatos.classIndex()).name());
	}
}
