package examenesPracticos.examen1;

import java.util.Random;

import weka.classifiers.Classifier;
import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
// clasificadores ZeroR, OneR y IBk
import weka.classifiers.rules.ZeroR;
import weka.classifiers.rules.OneR;
import weka.classifiers.lazy.IBk;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.core.converters.ConverterUtils.DataSource;


///////////////////////////////////////////////////////
// Observa:
// http://weka.wikispaces.com/Use+Weka+in+your+Java+code
// http://weka.sourceforge.net/doc.stable/weka/classifiers/Evaluation.html
///////////////////////////////////////////////////////
public class DataMiningExample1 {
	
    @SuppressWarnings("unused")
	public static void main(String[] args) throws Exception {
		////////////////////////////////////////////////////////////
    	// Cargando los datos de las instancias
        String folder = "/home/david/Documentos/Universidad/3º/2º Cuatrimestre/Sistemas de Apoyo a la Decisión/arff_files/";
    	String file = "breast-cancer.arff";
    	String path = folder + file;
    	DataSource source = new DataSource(path);
    	Instances data = null;
    	try {
    	    // no hace falta Reader/BufferedReader
            // esto lee directamente los datos del archivo
    		data = source.getDataSet();
	    	if (data.classIndex() == -1)
                // si no hay indice para la clase se indica
                // que la clase es el último atributo
				data.setClassIndex(data.numAttributes() - 1);
    	} catch (NullPointerException e) {
            // e.printStackTrace();
    		System.out.println("Problema al cargar los datos de " + path);
    		System.exit(1);
    	}
    	
		
		/////////////////////////////////////////////////////////////	
    	// Filtro para preprocesar las instancias
        // filtro
		AttributeSelection filter = new AttributeSelection();
    	// evaluador de los atributos
		CfsSubsetEval eval = new CfsSubsetEval();
		// no estoy seguro...
		BestFirst search = new BestFirst();
		// se ponen el evaluador y el buscador
		filter.setEvaluator(eval);
		filter.setSearch(search);
		// se introducen datos para que el filtro reconozca el formato
		filter.setInputFormat(data);

		// se filtran los datos
		Instances newData = Filter.useFilter(data, filter);

		/////////////////////////////////////////////////////////////
		// Clasificadores
        // instanciados clasificadores para posible uso
        // naive bayes
		NaiveBayes estimadorNB = new NaiveBayes();
        // estos son los clasificadores que se mencionan en la Práctica 2
        // zeroR
		ZeroR estimadorZR = new ZeroR();
		// oneR
		OneR estimadorOR = new OneR();
        // IBK con k = 1
		IBk estimadorIBk = new IBk(3);

		// atajo para tener que cambiar solo una variable
		Classifier estimadorAUsar = estimadorIBk;

		// las cuatro distinas evaluaciones que hemos dado
        // comentar las que no se vayan a usar para no hacerlas todas
//        evalNoHonesta(estimadorAUsar, newData);
//        evalHoldOut(estimadorAUsar, newData, 60);
        evalKFoldCrossValidation(estimadorAUsar, newData, 7);
//        evalLeaveOneOut(estimadorAUsar, newData);

    }

    private static void evalNoHonesta(Classifier pClassifier, Instances pData) throws Exception {
        // se inicializa el evaluador con los datos para que reconozca el formato
        Evaluation evaluator = new Evaluation(pData);
        // se entrena el clasificador con el set entero de datos
        pClassifier.buildClassifier(pData);
        // se evalua el clasificador con el set entero de datos
        evaluator.evaluateModel(pClassifier, pData);
        // se escriben los resultados
        printEvalResults(evaluator);
    }

    private static void evalHoldOut(Classifier pClassifier, Instances pData, double pPercent) throws Exception {
        // se randomizan los datos
        pData.randomize(new Random(1));
        // se calcula el número de instancias de cada set (train/test)
        int sizeTrain = (int) (pData.numInstances() * pPercent / 100);
        int sizeTest = pData.numInstances() - sizeTrain;
        // se dividen los datos en los sets train y test
        Instances dataTrain = new Instances(pData, 0, sizeTrain);
        Instances dataTest = new Instances(pData, sizeTrain, sizeTest);
        // se inicializa el evaluador con los datos para que reconozca el formato
        Evaluation evaluator = new Evaluation(pData);
        // se entrena el clasificador con el set train
        pClassifier.buildClassifier(dataTrain);
        // se evalua el clasificador con el set test
        evaluator.evaluateModel(pClassifier, dataTest);
        // se escriben los resultados
        printEvalResults(evaluator);
    }

    private static void evalKFoldCrossValidation(Classifier pClassifier, Instances pData, int pK) throws Exception {
        // se inicializa el evaluador con los datos para que reconozca el formato
        Evaluation evaluator = new Evaluation(pData);
        // se evalua el clasificador con el metodo k-fold cross validation
        evaluator.crossValidateModel(pClassifier, pData, pK, new Random(1));
        // se escriben los resultados
        printEvalResults(evaluator);
    }

    private static void evalLeaveOneOut(Classifier pClassifier, Instances pData) throws Exception {
        // se obtiene el número de instancias de los datos
        int k = pData.numInstances();
        // se hace el k-fold cross validation con k = num. instancias
        evalKFoldCrossValidation(pClassifier, pData, k);
    }

    private static void printEvalResults(Evaluation pEvaluator) throws Exception {
        printEvalResults(pEvaluator, true);
    }

    private static void printEvalResults(Evaluation pEvaluator, boolean pExit) throws Exception {
        System.out.println(pEvaluator.toSummaryString());
        System.out.println(pEvaluator.toMatrixString());
        if (pExit)
            System.exit(0);
    }
}
